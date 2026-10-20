import os
import sys
import json
import traceback
import git
import re
from datetime import datetime, timezone
from knowledge_graph import KnowledgeGraph
from term import Stage, stage as tstage, ok as tok, fail as tfail, info as tinfo, rule as trule, step as tstep
from utils import (
    extract_methods_from_traceback, 
    get_source_files_by_extensions,
    get_reference_functions_from_text, 
    read_file,
    normalize_repo_path,
    TextAnalyzer
)
from links import PatchLinkExpander
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from config import (
    GITHUB_TOKEN,
    NEO4J_URI, 
    NEO4J_USER,
    NEO4J_PASSWORD,
    MAX_CANDIDATE_METHODS,
    MAX_SEARCH_DEPTH,
    SEARCH_SPACE,
    WEAK_CONNECTION,
    NORMAL_CONNECTION,
    STRONG_CONNECTION,
)
from language_factory import LanguageConfigFactory, ParserFactory, language_by_extension, EXT_LANG_MAP
from benchmark import get_target_sample
from rtl_source_discovery import discover_rtl_source_files
from functools import lru_cache


def _entity_path_text(entity, max_hops=5):
    path = entity.get('path') or []
    steps = []
    for step in path[:max_hops]:
        if not isinstance(step, dict):
            continue
        start = str(step.get('start_node') or '').strip()
        end = str(step.get('end_node') or '').strip()
        relation = str(step.get('type') or step.get('relation_kind') or 'RELATED').strip()
        description = str(step.get('description') or '').strip()
        if start or end:
            label = f"{start} -[{relation}]-> {end}"
            if description:
                label = f"{start} -[{relation}: {description}]-> {end}"
            steps.append(label)
    if not steps:
        file_path = _repoish_path(entity.get('file_path'))
        target = str(entity.get('signature') or entity.get('name') or entity.get('signal_name') or '').strip()
        if file_path and target:
            steps.append(f"Issue:root -[MENTIONS]-> {file_path} -[CONTAINS]-> {target}")
        elif file_path:
            steps.append(f"Issue:root -[MENTIONS]-> {file_path}")
        elif target:
            steps.append(f"Issue:root -[MENTIONS]-> {target}")
    return " | ".join(steps)


def _fallback_issue_path(entity):
    file_path = _repoish_path(entity.get('file_path'))
    target = str(entity.get('signature') or entity.get('name') or entity.get('signal_name') or '').strip()
    rows = []
    if file_path:
        rows.append({
            'start_node': 'Issue:root',
            'end_node': file_path,
            'type': 'MENTIONS',
            'description': 'issue mentions candidate file',
        })
    if file_path and target:
        rows.append({
            'start_node': file_path,
            'end_node': target,
            'type': 'CONTAINS',
            'description': 'candidate file contains entity',
        })
    elif target:
        rows.append({
            'start_node': 'Issue:root',
            'end_node': target,
            'type': 'MENTIONS',
            'description': 'issue mentions candidate entity',
        })
    return rows


def _repoish_path(path):
    text = str(path or '').replace('\\', '/').strip()
    if not text:
        return ''
    matches = [(idx, marker) for marker in ('/src/', '/hw/', '/core/', '/rtl/', '/include/', '/common/') if (idx := text.find(marker)) >= 0]
    if matches:
        idx, _ = min(matches, key=lambda item: item[0])
        return text[idx + 1:]
    return text.lstrip('./')


def _build_issue_to_entity_paths(related_entities, limit=24):
    rows = []
    for group_name in (
        'patchable_spans',
        'direct_drivers',
        'top_level_wiring',
        'config_registers',
        'edit_targets',
        'evidence_entities',
        'rtl_entities',
        'methods',
    ):
        for entity in related_entities.get(group_name) or []:
            if not isinstance(entity, dict):
                continue
            path_text = _entity_path_text(entity)
            if not path_text:
                continue
            rows.append({
                'group': group_name,
                'target': entity.get('signature') or entity.get('name') or entity.get('signal_name') or '',
                'type': entity.get('type') or entity.get('verilog_kind') or entity.get('label') or '',
                'file_path': _repoish_path(entity.get('file_path')) or '',
                'start_line': entity.get('start_line'),
                'end_line': entity.get('end_line'),
                'module_name': entity.get('module_name') or '',
                'similarity': entity.get('similarity') or 0,
                'global_rank': entity.get('global_rank'),
                'mode_rank': entity.get('mode_rank'),
                'distance_mode': entity.get('distance_mode'),
                'distance': entity.get('distance'),
                'path': entity.get('path') or _fallback_issue_path(entity),
                'path_text': path_text,
            })
    rows.sort(key=lambda item: (
        int(item.get('global_rank')) if item.get('global_rank') is not None else 10**9,
        -float(item.get('similarity') or 0),
    ))
    seen = set()
    deduped = []
    for row in rows:
        key = (row.get('target'), row.get('file_path'), row.get('start_line'), row.get('path_text'))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    # Keep the path artifact repair-oriented: reserve most of its fixed budget
    # for ranked PatchableSpan targets, then add semantic evidence paths.
    patch_rows = [
        row for row in deduped
        if row.get('group') in {'patchable_spans', 'edit_targets'}
        or str(row.get('type') or '').lower() in {'patchable_span', 'patchablespan'}
    ]
    result = patch_rows[:min(16, limit)]
    selected = {
        (row.get('target'), row.get('file_path'), row.get('start_line'), row.get('path_text'))
        for row in result
    }
    for row in deduped:
        key = (row.get('target'), row.get('file_path'), row.get('start_line'), row.get('path_text'))
        if key in selected:
            continue
        result.append(row)
        selected.add(key)
        if len(result) >= limit:
            break
    return result

class CodeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.language_config = LanguageConfigFactory.get_config(config.get('language', 'python'))
        self.parser = ParserFactory.create_parser(self.language_config.language)
        self.repo_path = config['repo_path']
        self._set_parser_repo_root(self.parser)
        self.repo = git.Repo(self.repo_path)
        self.rtl_source_discovery = []
        self.github_token = GITHUB_TOKEN
        self.max_search_depth = MAX_SEARCH_DEPTH
        self.kg = KnowledgeGraph(
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            sys.argv[1] # instance_id
        )
        self.kg.clear_graph()
        self.kg._create_indexes()
        patch_expander_token = None if config.get('benchmark_name') in {'local', 'verilog-local'} else GITHUB_TOKEN
        self.patch_link_expander = PatchLinkExpander(patch_expander_token, config['repo_name'])
        self.method_search_cache = {}
        self.issue_cache = {}
        self.MAX_CANDIDATE_METHODS = MAX_CANDIDATE_METHODS
        self.processed_prs = set()
        self.processed_files = set()
        self.linked_issues = set()
        self.linked_issue_contents = set()
        self.searched_methods = set()
        self.artifact_stats = {"skipped_due_to_time": 0, "valid_related_items": 0}
        self.counted_valid_artifact_ids = set()
        self.counted_skipped_artifact_ids = set()

    def _clean_path(self, file_path: str) -> str:
        return normalize_repo_path(file_path, self.config.get('repo_path', ''))

    def _is_verilog_noise_path(self, path: str) -> bool:
        """Return True for paths that should not guide RTL repair localization."""
        if self.language_config.language != 'verilog':
            return False
        repo_root = os.path.abspath(self.config.get('repo_path', self.repo_path))
        abs_path = os.path.abspath(path)
        try:
            rel = os.path.relpath(abs_path, repo_root).replace('\\', '/').lower()
        except ValueError:
            rel = str(path).replace('\\', '/').lower()
        parts = set(rel.split('/'))
        if parts & {
            '.git', '__pycache__', 'node_modules', 'build', 'dist', 'out',
            'bazel-out', '.cache', 'coverage', 'generated', 'autogen', 'gen', 'docs',
            'doc', 'test', 'tests', 'tb', 'dv', 'sim', 'verif', 'vendor', 'asserts',
        }:
            return True
        return (
            rel.startswith(('pd/synth/', 'scratch/', 'tmp/'))
            or '/deprecated/' in rel
            or re.search(r'(^|/)(?:.*_tb|tb_.*|.*_test|test_.*)\.(?:sv|svh|v|vh)$', rel) is not None
        )

    def _check_and_count_artifact_time(self, artifact_timestamp, artifact_unique_id: str) -> bool:
        """
        Checks if artifact_timestamp is not later than self.created_at.
        Updates artifact_stats and counted sets. Returns True if valid, False if skipped.
        """
        unique_id = str(artifact_unique_id) # Ensure string
        if artifact_timestamp > self.created_at:
            if unique_id not in self.counted_skipped_artifact_ids:
                self.artifact_stats["skipped_due_to_time"] += 1
                self.counted_skipped_artifact_ids.add(unique_id)
            return False # Invalid due to time (too late)
        else:
            if unique_id not in self.counted_valid_artifact_ids:
                self.artifact_stats["valid_related_items"] += 1
                self.counted_valid_artifact_ids.add(unique_id)
            return True # Valid by time

    @lru_cache(maxsize=None)
    def _parser_for_file(self, file_path: str):
        lang = language_by_extension(file_path)
        if not lang:
            return None
        parser = ParserFactory.create_parser(lang)
        self._set_parser_repo_root(parser)
        return parser

    def _set_parser_repo_root(self, parser):
        if parser is not None and hasattr(parser, 'set_repo_root'):
            parser.set_repo_root(self.config.get('repo_path', self.repo_path))

    def analyze(self):
        """Execute complete analysis flow"""
        try:
            target_sample = self._get_target_sample()
            if not target_sample:
                return
            self._process_repository(target_sample)
            issue_text = target_sample.get('problem_statement', '') or target_sample.get('issue', '') or ''
            issue_mentioned_files = self._extract_verilog_issue_file_hints(issue_text) if self.language_config.language == 'verilog' else []
            if self.language_config.language == 'verilog' and issue_mentioned_files:
                for file_hint in issue_mentioned_files[:8]:
                    try:
                        self.kg.link_issue_to_file('root', file_hint, weight=0.25)
                    except Exception as exc:
                        tinfo(f"Unable to link issue hint file {file_hint}: {exc}")
                tinfo(f"Issue-derived RTL file hints: {', '.join(issue_mentioned_files[:8])}")
            
            # Get related entities
            related_entities = self.kg.get_all_similarities_to_root(
                limit=SEARCH_SPACE,
                max_hops=3,
                sort=True,
                issue_text=issue_text,
                issue_mentioned_files=issue_mentioned_files,
            )

            for group_name, entities in related_entities.items():
                if isinstance(entities, list):
                    if group_name in {'patchable_spans', 'edit_targets'} and any(
                        isinstance(item, dict) and item.get('global_rank') is not None
                        for item in entities
                    ):
                        entities.sort(key=lambda item: int(item.get('global_rank') or 10**9))
                    else:
                        entities.sort(key=lambda x: x.get('similarity', 0) if isinstance(x, dict) else 0, reverse=True)
            
            tinfo(f"Found {sum(len(v) for v in related_entities.values())} related entities")
            for entity_type, entities in sorted(related_entities.items()):
                if entities:
                    tinfo(f"  {entity_type}: {len(entities)}")

            issue_to_entity_paths = _build_issue_to_entity_paths(related_entities)
            if issue_to_entity_paths:
                tinfo(
                    f"Persisting {len(issue_to_entity_paths)} "
                    "Issue-to-PatchableSpan KG inference paths"
                )
            
            return {
                'related_entities': related_entities,
                'issue_to_entity_paths': issue_to_entity_paths,
                # Canonical name for the new methodology; retain the legacy
                # key above for existing repair artifacts and readers.
                'issue_to_patchable_span_paths': issue_to_entity_paths,
                'kg_query_mode': {
                    'persisted_json': 'location snapshot and path cache',
                    'neo4j': 'authoritative graph for dynamic cross-file signal propagation queries',
                },
                'kg_schema': self.kg.schema_summary(),
                'ranking_version': (self.kg.last_localization_metadata or {}).get('ranking_version', 'kgcompass-b0'),
                'localization_metadata': self.kg.last_localization_metadata or {},
                'artifact_stats': self.artifact_stats,
                'rtl_source_discovery': self.rtl_source_discovery,
                'issue': issue_text,
                'issue_mentioned_files': issue_mentioned_files,
                'language': self.language_config.language,
                'instance_id': self.config.get('instance_id'),
                'repo': self.config.get('repo_name'),
            }
            
        except Exception as e:
            print(f"Analysis process error: {str(e)}")
            print(traceback.format_exc())
        finally:
            self._cleanup()

    def _extract_verilog_issue_file_hints(self, issue_text: str):
        issue_text = issue_text or ''
        hints = []
        seen = set()

        def normalize_hint(value):
            value = str(value or '').strip().strip('`"\'').replace('\\', '/')
            if not value or not re.search(r'\.(?:sv|svh|v|vh)$', value, flags=re.IGNORECASE):
                return ''
            if os.path.isabs(value):
                try:
                    value = os.path.relpath(value, self.repo_path).replace('\\', '/')
                except ValueError:
                    value = value.lstrip('/')
            if value.startswith('./'):
                value = value[2:]
            return value

        def is_source_candidate(value):
            lowered = value.lower()
            return not (
                re.search(r'(^|/)(\.git|tb|test|tests|dv|sim|verif|formal|benchmark|benchmarks|out|build|bazel-out|autogen)(/|$)', lowered)
                or re.search(r'(^|/)(verification[^/]*|uvm[^/]*|uvmf[^/]*|environment_packages)(/|$)', lowered)
                or re.search(r'(^|/)(vendor|generated|gen|docs|doc|pd/synth)(/|$)', lowered)
                or self._is_verilog_noise_path(os.path.join(self.repo_path, value))
            )

        def add_hint(value):
            value = normalize_hint(value)
            if value and is_source_candidate(value) and value not in seen:
                hints.append(value)
                seen.add(value)

        for match in re.finditer(r"`([^`]+\.(?:sv|svh|v|vh))`", issue_text, flags=re.IGNORECASE):
            add_hint(match.group(1))
        for match in re.finditer(r"(?<![\w./-])([A-Za-z0-9_./-]+\.(?:sv|svh|v|vh))\b", issue_text, flags=re.IGNORECASE):
            add_hint(match.group(1))

        repo_candidates = []
        if os.path.isdir(self.repo_path):
            for root, dirs, files in os.walk(self.repo_path):
                dirs[:] = [
                    d for d in dirs
                    if not re.search(r'(^|/)(\.git|tb|test|tests|dv|sim|verif|formal|verification[^/]*|uvm[^/]*|uvmf[^/]*|environment_packages|benchmark|benchmarks|out|build|bazel-out|vendor|generated|autogen|gen|docs|doc)(/|$)', os.path.relpath(os.path.join(root, d), self.repo_path).replace('\\', '/').lower())
                    and not self._is_verilog_noise_path(os.path.join(root, d))
                ]
                for filename in files:
                    if not re.search(r'\.(?:sv|svh|v|vh)$', filename, flags=re.IGNORECASE):
                        continue
                    rel_path = os.path.relpath(os.path.join(root, filename), self.repo_path).replace('\\', '/')
                    if is_source_candidate(rel_path):
                        repo_candidates.append(rel_path)

        stopwords = {
            'module', 'signal', 'always', 'assign', 'reset', 'clock', 'state',
            'register', 'error', 'issue', 'logic', 'input', 'output', 'wire',
            'reg', 'bit', 'byte', 'test', 'case', 'should', 'would', 'could',
            'there', 'their', 'with', 'from', 'this', 'that', 'have', 'has',
            'when', 'then', 'expected', 'actual', 'instruction', 'instructions',
            'failing', 'failure', 'verilog', 'repo', 'description', 'condition',
            'conditions', 'function', 'affected', 'trigger', 'memory', 'physical',
            'protection', 'scenario', 'observed', 'result', 'results', 'problem',
            'summary', 'steps', 'reproduce', 'baseline', 'buggy', 'fixed',
            'implementation', 'configuration', 'config', 'feature', 'option',
            'value', 'values', 'field', 'fields', 'ibex', 'cva6', 'risc', 'riscv',
            'caliptra', 'opentitan'
        }
        issue_tokens = []
        for raw in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b", issue_text):
            token = raw.lower()
            if token not in stopwords:
                issue_tokens.append(token)
        issue_tokens = list(dict.fromkeys(issue_tokens))[:24]
        csr_issue = bool(re.search(r'\b(?:csr|csrs|control\s+and\s+status|trap)\b', issue_text, flags=re.IGNORECASE))
        csr_terms = {
            'mstatus', 'misa', 'mtvec', 'mscratch', 'mepc', 'mcause', 'mtval',
            'mhartid', 'mcountinhibit', 'mcycle', 'minstret', 'dcsr', 'dpc',
            'dscratch0', 'dscratch1'
        }

        ranked = []
        for candidate in repo_candidates:
            path_lower = candidate.lower()
            base_lower = os.path.splitext(os.path.basename(candidate))[0].lower()
            split_tokens = set(re.split(r'[^a-z0-9]+', path_lower))
            score = 0
            for token in issue_tokens:
                if token in split_tokens:
                    score += 4
                elif token in base_lower:
                    score += 3
                elif token in path_lower:
                    score += 1
                if csr_issue and base_lower in {'ibex_cs_registers', 'cs_registers'}:
                    if token in {'csr', 'csrs', 'register', 'registers'} or token in csr_terms:
                        score += 8
                    elif token in {'trap', 'machine', 'privileged'}:
                        score += 3
            for first, second in zip(issue_tokens, issue_tokens[1:]):
                if first in base_lower and second in base_lower:
                    score += 3
            if csr_issue and base_lower in {'ibex_cs_registers', 'cs_registers'}:
                score += 10
            if 'rtl/' in path_lower:
                score += 1
            if score:
                ranked.append((score, candidate))
        for _, candidate in sorted(ranked, key=lambda item: (-item[0], len(item[1]), item[1]))[:8]:
            add_hint(candidate)
        return hints

    # Use GitPython to get file commit information
    def get_commit_info(self, file_path):
        try:
            # Get relative file path to repository root
            repo_root = self.repo.working_tree_dir
            relative_file_path = os.path.relpath(file_path, repo_root).replace('\\', '/')
            
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist")
                return []

            commits = list(self.repo.iter_commits(paths=relative_file_path, max_count=1))
            if not commits:
                print(f"Warning: File {file_path} has no commit history")
                return []
            
            last_commit = commits[0]
            commit_id = last_commit.hexsha
            commit_message = last_commit.message.strip()

            try:
                # Use relative path to run git blame to get commit information for each line
                blame_info = self.repo.git.blame('HEAD', '--', relative_file_path).splitlines()
            except git.exc.GitCommandError as e:
                print(f"Warning: Unable to get blame information for file {file_path}: {e}")
                return []

            commit_data = []
            total_lines = len(blame_info)
            current_line = 1
            
            print(f"Start processing git blame info for {os.path.basename(file_path)} ({total_lines} lines)")
            
            for i, line in enumerate(blame_info, 1):
                if i % 100 == 0:
                    print(f"Progress: {i}/{total_lines} lines ({(i/total_lines*100):.1f}%)")
                
                try:
                    parts = line.split(')', 1)
                    if len(parts) < 2:
                        continue
                    blame_info = parts[0].split('(', 1)[1].strip()
                    blame_parts = blame_info.rsplit(' ', 2)
                    if len(blame_parts) < 3:
                        continue
                    line_commit_id = parts[0].split()[0]
                    commit_data.append((current_line, line_commit_id, commit_message))
                    current_line += 1
                except Exception as e:
                    print(f"Warning: Error processing line {i}: {e}")
                    continue
            
            print(f"Completed processing git blame info: {total_lines} lines")
            return commit_data
        except Exception as e:
            print(f"git blame error: {e}")
            print(traceback.format_exc())
            return []
    
    def _get_target_sample(self):
        """Get target sample based on configured benchmark."""
        benchmark_name = self.config.get('benchmark_name', 'verilog-local')
        final_target_sample = None

        print(f"Attempting to load dataset for benchmark: {benchmark_name}")

        if benchmark_name in {'local', 'verilog-local'}:
            final_target_sample = get_target_sample(
                self.config['instance_id'],
                repo_name=self.config.get('repo_name'),
                benchmark_name=benchmark_name,
            )
        else:
            print(f"Unsupported benchmark_name: {benchmark_name}")
            return None

        if not final_target_sample:
            print(f"No sample found for instance_id '{self.config['instance_id']}' in repo '{self.config['repo_name']}' for benchmark '{benchmark_name}'")
            return None

        print(f"Found target sample for benchmark {benchmark_name}:")
        print(f"  Repo: {final_target_sample.get('repo')}")
        print(f"  Instance ID: {final_target_sample.get('instance_id')}")
        print(f"  Base commit: {final_target_sample.get('base_commit')}")
        # Print only a snippet of the problem statement
        problem_statement_snippet = (final_target_sample.get('problem_statement') or "")[:200]
        print(f"  Problem: {problem_statement_snippet}...")
        print(f"  Created At: {final_target_sample.get('created_at')}")
        if final_target_sample.get('repo'):
            self.config['repo_name'] = final_target_sample.get('repo')
        
        return final_target_sample

    def _build_file_class_methods(self, file_path):
        parser = self._parser_for_file(file_path)
        if parser is None:
            return

        # 检查文件扩展名
        if not any(file_path.endswith(ext) for ext in self.language_config.config['file_extensions']):
            return
            
        if re.search(r'(?:^|[/\\])test[/\\]|[/\\]tests[/\\]|_(?:tb|test|smoke|unit_tb|regression_tb|tb_top)\b', file_path, re.IGNORECASE) and not file_path.endswith(self.language_config.config['test_file_pattern']):
            print(f"Skip processing test file: {file_path}")
            return
            
        if file_path in self.processed_files:
            print(f"File {file_path} already processed, skipping")
            return
            
        print(f"Processing file: {file_path}")
        self.processed_files.add(file_path)
        
        # 使用语言特定的解析器
        classes = parser.extract_classes(file_path)
        
        for class_info in classes:
            class_name = class_info['name'] if class_info['name'] else '__'
            
            # 创建类实体
            self.kg.create_class_entity(
                class_name,
                class_info['file_path'],
                class_info['start_line'],
                class_info['end_line'],
                class_info.get('source_code', ''),
                class_info.get('doc_string', ''),
                STRONG_CONNECTION,
                verilog_kind=class_info.get('verilog_kind'),
                semantic_summary=class_info.get('semantic_summary'),
                parse_source=class_info.get('parse_source'),
                parse_confidence=class_info.get('parse_confidence'),
                repair_role=class_info.get('repair_role'),
                timing_tags=class_info.get('timing_tags'),
                timing_priority=class_info.get('timing_priority'),
                timing_summary=class_info.get('timing_summary'),
            )
            self.kg.link_class_to_file(class_name, class_info['file_path'], STRONG_CONNECTION)
            
            # 处理方法
            for method in class_info.get('methods', []):
                method_name = f"{method['name']}"
                
                self.kg.create_method_entity(
                    method_name,
                    method['signature'],
                    method['file_path'],
                    method['start_line'],
                    method['end_line'],
                    method['source_code'],
                    method.get('doc_string', ''),
                    STRONG_CONNECTION,
                    verilog_kind=method.get('verilog_kind'),
                    semantic_summary=method.get('semantic_summary'),
                    parse_source=method.get('parse_source'),
                    parse_confidence=method.get('parse_confidence'),
                    repair_role=method.get('repair_role'),
                    timing_tags=method.get('timing_tags'),
                    timing_priority=method.get('timing_priority'),
                    timing_summary=method.get('timing_summary'),
                )
                
                self.kg.link_class_to_method(
                    class_name,
                    class_info['file_path'],
                    method_name,
                    method['signature'],
                    STRONG_CONNECTION
                )

        if getattr(parser.language_config, 'language', '') == 'verilog' and hasattr(parser, 'extract_rtl_graph'):
            self._build_verilog_rtl_entities(file_path, parser)

    def _build_verilog_rtl_entities(self, file_path, parser):
        try:
            rtl_graph = parser.extract_rtl_graph(file_path)
        except Exception as e:
            print(f"Error extracting Verilog RTL graph for {file_path}: {e}")
            return

        def _edge_weight(parse_confidence, default_weight=NORMAL_CONNECTION):
            try:
                confidence = float(parse_confidence)
            except (TypeError, ValueError):
                return default_weight
            confidence = max(0.0, min(1.0, confidence))
            return max(0.15, round(1.05 - confidence, 2))

        for entity in rtl_graph.get('entities', []):
            self.kg.create_hdl_entity(
                entity['label'],
                entity['name'],
                entity['file_path'],
                module_name=entity.get('module_name', ''),
                rtl_kind=entity.get('rtl_kind', ''),
                signal_name=entity.get('signal_name', ''),
                direction=entity.get('direction', ''),
                width=entity.get('width', ''),
                declaration=entity.get('declaration', ''),
                start_line=entity.get('start_line'),
                end_line=entity.get('end_line'),
                start_column=entity.get('start_column'),
                end_column=entity.get('end_column'),
                source_code=entity.get('source_code', ''),
                semantic_summary=entity.get('semantic_summary', ''),
                parse_source=entity.get('parse_source'),
                parse_confidence=entity.get('parse_confidence'),
                repair_role=entity.get('repair_role'),
                timing_tags=entity.get('timing_tags'),
                timing_priority=entity.get('timing_priority'),
                timing_summary=entity.get('timing_summary'),
                span_kind=entity.get('span_kind', ''),
                patch_granularity=entity.get('patch_granularity', ''),
                parent_span=entity.get('parent_span', ''),
                reads_signals=entity.get('reads_signals'),
                writes_signals=entity.get('writes_signals'),
                drives_signals=entity.get('drives_signals'),
                connects_signals=entity.get('connects_signals'),
                guards_signals=entity.get('guards_signals'),
                port_bindings=entity.get('port_bindings'),
                projection_source=entity.get('projection_source', ''),
                weight=STRONG_CONNECTION,
            )

            module_name = entity.get('module_name')
            if module_name:
                self.kg.link_class_to_hdl_entity(
                    module_name,
                    entity['file_path'],
                    entity['label'],
                    entity['name'],
                    entity['file_path'],
                    entity_module_name=module_name,
                    description=f"contains {entity['label'].lower()}",
                    reverse_description='contained in module',
                    relation_kind='contains',
                    parse_source=entity.get('parse_source'),
                    parse_confidence=entity.get('parse_confidence'),
                    weight=STRONG_CONNECTION,
                )

        for edge in [
            *(rtl_graph.get('method_signal_edges', []) or []),
            *(rtl_graph.get('method_span_edges', []) or []),
        ]:
            self.kg.link_method_to_hdl_entity(
                edge['method_name'],
                edge['method_signature'],
                edge['method_file_path'],
                edge['target_label'],
                edge['target_name'],
                edge['target_file_path'],
                entity_module_name=edge.get('target_module_name', ''),
                description=edge.get('description', 'uses rtl entity'),
                reverse_description=edge.get('reverse_description', 'used by method'),
                relation_kind=edge.get('relation_kind', 'uses'),
                parse_source=edge.get('parse_source'),
                parse_confidence=edge.get('parse_confidence'),
                weight=_edge_weight(edge.get('parse_confidence')),
            )

        for edge in rtl_graph.get('signal_edges', []):
            self.kg.link_hdl_entities(
                edge['source_label'],
                edge['source_name'],
                edge['source_file_path'],
                edge.get('source_module_name', ''),
                edge['target_label'],
                edge['target_name'],
                edge['target_file_path'],
                edge.get('target_module_name', ''),
                description=edge.get('description', 'feeds signal'),
                reverse_description=edge.get('reverse_description', 'fed by signal'),
                relation_kind=edge.get('relation_kind', 'feeds'),
                parse_source=edge.get('parse_source'),
                parse_confidence=edge.get('parse_confidence'),
                weight=_edge_weight(edge.get('parse_confidence')),
            )

        for edge in rtl_graph.get('span_edges', []):
            self.kg.link_hdl_entities(
                edge['source_label'],
                edge['source_name'],
                edge['source_file_path'],
                edge.get('source_module_name', ''),
                edge['target_label'],
                edge['target_name'],
                edge['target_file_path'],
                edge.get('target_module_name', ''),
                description=edge.get('description', 'projects to patchable span'),
                reverse_description=edge.get('reverse_description', 'projected from semantic entity'),
                relation_kind=edge.get('relation_kind', 'projects_to'),
                parse_source=edge.get('parse_source'),
                parse_confidence=edge.get('parse_confidence'),
                weight=_edge_weight(edge.get('parse_confidence')),
            )

        for edge in rtl_graph.get('instance_edges', []):
            self.kg.link_method_to_class(
                edge['method_name'],
                edge['method_signature'],
                edge['method_file_path'],
                edge['class_name'],
                description=edge.get('description', 'instantiates module'),
                reverse_description=edge.get('reverse_description', 'instantiated by method'),
                relation_kind=edge.get('relation_kind', 'instantiates'),
                parse_source=edge.get('parse_source'),
                parse_confidence=edge.get('parse_confidence'),
                weight=_edge_weight(edge.get('parse_confidence')),
            )

    def _process_reference(self, issue_id, ref_data, multipler = 1, issue_refs=None):
        ref_type, full_path = ref_data
        if full_path in self.searched_methods:
            return
        
        with self.lock: # Assuming self.lock is initialized elsewhere (e.g., in _link_reference_to_issue_faster)
            if full_path in self.searched_methods:
                return
            self.searched_methods.add(full_path)
        
        module_parts = full_path.split('.')
        target_name = module_parts[-1]
        if target_name == 'py' and len(module_parts) > 1: # Python-specific heuristic
            target_name = module_parts[-2]
        elif f".{target_name}" in getattr(self.language_config, 'config', {}).get('file_extensions', []):
            target_name = os.path.splitext(os.path.basename(full_path.replace('\\', '/')))[0]
        is_contextual_verilog_signal = self._is_contextual_verilog_signal_reference(ref_type, target_name)
        
        base_path = self.config['repo_path']
        initial_path_resolver_config = self.parser.language_config

        possible_paths_generated = []
        possible_paths_generated.extend(initial_path_resolver_config.resolve_qualified_name_to_file_paths(base_path, module_parts))
        if len(module_parts) > 1:
            possible_paths_generated.extend(initial_path_resolver_config.resolve_qualified_name_to_file_paths(base_path, module_parts[:-1]))

        possible_paths = []
        seen_paths = set()
        for type_hint, path_str in possible_paths_generated:
            if path_str not in seen_paths:
                possible_paths.append((type_hint, path_str))
                seen_paths.add(path_str)

        find_by_kg = [] if is_contextual_verilog_signal else self.kg.search_file_by_path(target_name)
        if find_by_kg:
            for file_node in find_by_kg:
                kg_file_path = file_node['file']['path']
                if kg_file_path not in seen_paths and not self._is_verilog_noise_path(kg_file_path):
                    possible_paths.append(('file', kg_file_path))
                    seen_paths.add(kg_file_path)
        
        found_specific_entity = False
        processed_files_for_this_ref = set()

        for path_type_hint, file_path_candidate in possible_paths:
            if not os.path.exists(file_path_candidate):
                continue
            if self._is_verilog_noise_path(file_path_candidate):
                continue

            actual_parser = self._parser_for_file(file_path_candidate)
            if not actual_parser:
                continue

            if os.path.isdir(file_path_candidate):
                if path_type_hint == 'package': 
                    print(f"Processing directory import/reference: {file_path_candidate}")
                    self.kg.create_directory_structure(file_path_candidate, self, True, multipler * STRONG_CONNECTION)
                    found_specific_entity = True 
                    current_lang_extensions = actual_parser.language_config.config['file_extensions']
                    for item_name in os.listdir(file_path_candidate):
                        if any(item_name.endswith(ext) for ext in current_lang_extensions):
                            dir_file_path = os.path.join(file_path_candidate, item_name)
                            if os.path.isfile(dir_file_path) and dir_file_path not in processed_files_for_this_ref:
                                print(f"Associated directory file: {dir_file_path}")
                                clean_dir_file_path = self._clean_path(dir_file_path)
                                self.kg.create_file_entity(clean_dir_file_path)
                                self.kg.link_issue_to_file(issue_id, clean_dir_file_path, multipler * NORMAL_CONNECTION)
                                self._build_file_class_methods(dir_file_path)
                                processed_files_for_this_ref.add(dir_file_path)
                continue 
            
            if file_path_candidate in processed_files_for_this_ref:
                continue

            print(f"Processing file candidate for reference '{full_path}': {file_path_candidate} using {actual_parser.language_config.language} parser")
            clean_file_path_candidate = self._clean_path(file_path_candidate)
            classes = actual_parser.extract_classes(file_path_candidate)
            methods = actual_parser.get_global_methods(file_path_candidate, self.config['repo_root'])
            methods.extend(actual_parser.get_global_variables(file_path_candidate, self.config['repo_root']))

            self.kg.link_issue_to_file(issue_id, clean_file_path_candidate, multipler * STRONG_CONNECTION)
            self._build_file_class_methods(file_path_candidate)
            processed_files_for_this_ref.add(file_path_candidate)
            # Marking found_specific_entity = True here because we successfully processed a directly resolved file path.
            # Specific entity linking below is a bonus.
            found_specific_entity = True 

            entity_linked_in_file = False
            for class_info in classes:
                if class_info['name'] == target_name: 
                    print(f"Matched class by target_name: {class_info['name']}")
                    self.kg.link_class_to_issue(class_info['name'], class_info['file_path'], issue_id, multipler * NORMAL_CONNECTION)
                    entity_linked_in_file = True
                for method_info in class_info.get('methods', []):
                    if method_info['name'] == target_name:
                        print(f"Matched method by target_name: {method_info['name']}")
                        self.kg.link_method_to_issue(method_info['name'], method_info['signature'], method_info['file_path'], issue_id, multipler * NORMAL_CONNECTION)
                        entity_linked_in_file = True
            
            for method_info in methods:
                if method_info['name'] == target_name:
                    print(f"Matched global method/variable by target_name: {method_info['name']}")
                    self.kg.link_method_to_issue(method_info['name'], method_info['signature'], method_info['file_path'], issue_id, multipler * NORMAL_CONNECTION)
                    entity_linked_in_file = True
            
            # No need to set found_specific_entity again if entity_linked_in_file is true, as it's already true from file processing.

        if not found_specific_entity:
            if is_contextual_verilog_signal:
                print(
                    f"No direct path for contextual signal reference '{full_path}'. "
                    f"Searching only within issue-context RTL files for '{target_name}'."
                )
                found_files_by_name_search = self._search_short_signal_by_issue_context(
                    self.config['repo_path'],
                    target_name,
                    issue_refs or [],
                )
                if not found_files_by_name_search:
                    print(
                        f"Contextual signal reference '{full_path}' was not resolved in issue-context files; "
                        "skipping global name search to avoid noisy RTL localization."
                    )
            else:
                print(f"No specific entity found by direct path resolution for '{full_path}'. Falling back to name search for '{target_name}'.")
                found_files_by_name_search = self._search_method_by_name(self.config['repo_path'], target_name)
            
            for file_match_info in found_files_by_name_search:
                file_path_from_search = file_match_info['path']
                match_type_from_search = file_match_info['type']

                if file_path_from_search in processed_files_for_this_ref:
                    continue
                if self._is_verilog_noise_path(file_path_from_search):
                    continue

                actual_parser_for_searched_file = self._parser_for_file(file_path_from_search)
                if not actual_parser_for_searched_file:
                    continue

                print(f"Found '{target_name}' via name search in file: {file_path_from_search} (match type: {match_type_from_search})")
                clean_file_path_from_search = self._clean_path(file_path_from_search)
                self.kg.link_issue_to_file(issue_id, clean_file_path_from_search, multipler * WEAK_CONNECTION)
                self._build_file_class_methods(file_path_from_search)
                processed_files_for_this_ref.add(file_path_from_search)
                found_specific_entity = True 

                s_classes = actual_parser_for_searched_file.extract_classes(file_path_from_search)
                s_methods = actual_parser_for_searched_file.get_global_methods(file_path_from_search, self.config['repo_root'])
                s_methods.extend(actual_parser_for_searched_file.get_global_variables(file_path_from_search, self.config['repo_root']))

                for class_info in s_classes:
                    if class_info['name'] == target_name: 
                        self.kg.link_class_to_issue(class_info['name'], class_info['file_path'], issue_id, multipler * WEAK_CONNECTION)
                    for method_info in class_info.get('methods', []):
                        if method_info['name'] == target_name:
                            self.kg.link_method_to_issue(method_info['name'], method_info['signature'], method_info['file_path'], issue_id, multipler * WEAK_CONNECTION)
                for method_info in s_methods:
                    if method_info['name'] == target_name:
                        self.kg.link_method_to_issue(method_info['name'], method_info['signature'], method_info['file_path'], issue_id, multipler * WEAK_CONNECTION)

        if not found_specific_entity:
            # Use double quotes for the main f-string to allow single quotes inside for variable values
            print(f"Reference '{full_path}' (target: '{target_name}') could not be resolved to a specific file or entity after all attempts.")

    def _link_reference_to_issue_faster(self, issue_id, issue_content, multipler = 1):
        if issue_id in self.linked_issues or issue_content in self.linked_issue_contents:
            print(f"Issue/PR #{issue_id} already processed, skipping")
            return
        self.linked_issues.add(issue_id)
        self.linked_issue_contents.add(issue_content)
        print(f"Processing Issue/PR #{issue_id} references")
        ref_list = get_reference_functions_from_text(self.config['repo_path'], issue_content, self.parser, self.method_search_cache)
        if self.language_config.language == 'verilog':
            signal_values = {str(value or '') for kind, value in ref_list if kind == 'signal'}
            if signal_values:
                ref_list = [
                    (kind, value) for kind, value in ref_list
                    if not (str(value or '') in signal_values and kind != 'signal')
                ]
        print(ref_list)
        if not ref_list:
            print(f"No code-like references found in Issue/PR #{issue_id}")
            return
        if not hasattr(self, 'lock'):
            self.lock = threading.Lock()
        primary_refs = []
        signal_refs = []
        for ref_data in ref_list:
            ref_type, ref_value = ref_data
            target_name = str(ref_value or '').split('.')[-1]
            if self._is_contextual_verilog_signal_reference(ref_type, target_name):
                signal_refs.append(ref_data)
            else:
                primary_refs.append(ref_data)

        def process_batch(batch, offset=0):
            if not batch:
                return
            with ThreadPoolExecutor(max_workers=min(16, len(batch))) as executor:
                future_to_ref = {
                    executor.submit(self._process_reference, issue_id, ref_data, multipler, ref_list): (ref_data, issue_id)
                    for ref_data in batch
                }
                for idx, future in enumerate(as_completed(future_to_ref), 1):
                    ref_data = future_to_ref[future]
                    try:
                        future.result()
                        print(f"Completed processing reference [{offset + idx}/{len(ref_list)}]: {ref_data[0]} -> {ref_data[1]}")
                    except Exception as e:
                        print(f"Error processing reference {ref_data[0]} -> {ref_data[1]}: {str(e)}")

        # Resolve module/file references before signal tokens. Signals are only
        # meaningful when anchored to issue-local RTL context.
        process_batch(primary_refs, 0)
        process_batch(signal_refs, len(primary_refs))

    def _link_stacktrace_to_issue(self, issue_id, issue_content):
        # Extract method information from stack trace
        stack_methods = extract_methods_from_traceback(self.repo.working_tree_dir, self.config['repo_root'], issue_content, self.kg, self.parser)
        for method_info in stack_methods:
            # method_info['file_path'] is now cleaned by the extractor
            full_path_for_check = os.path.join(self.config['repo_path'], os.path.normpath(method_info['file_path']))
            # A simpler way to reconstruct, assuming repo_path is like 'playground/repo_name'
            full_path_for_check_alt = os.path.join('playground', self._clean_path(method_info['file_path']))

            print(f"Found method from stack trace: {method_info}")

            # Reconstruct the full path to check for existence
            # This is necessary because method_info['file_path'] is now the cleaned, relative path.
            # self.repo_path is 'playground/<repo_name>/'
            full_path_for_check = os.path.join(self.config['repo_path'], os.path.normpath(method_info['file_path']))

            # A simpler way might be to join 'playground' with the cleaned path if the structure is fixed
            # For now, let's trust that the parser gives a path relative to repo root (without repo name)
            # And self.repo_path is the full path to the checkout dir.
            
            # The previous logic in `_clean_path` makes it relative to cwd.
            # A better way to reconstruct:
            # The cleaned path is relative to 'playground'. So we join 'playground' and the cleaned path.
            reconstructed_path = os.path.join('playground', method_info['file_path'])

            if os.path.exists(reconstructed_path):
                # Create method entity
                self.kg.create_method_entity(
                    method_info['name'],
                    method_info.get('signature', ''),
                    method_info['file_path'],
                    method_info.get('line_number', 0),
                    method_info.get('line_number', 0),
                    method_info.get('source_code', ''),
                    method_info.get('doc_string', ''),
                    STRONG_CONNECTION
                )
                
                # Establish method-root_issue association
                self.kg.link_method_to_issue(
                    method_info['name'],
                    method_info.get('signature', ''),
                    method_info['file_path'],
                    issue_id,
                    STRONG_CONNECTION
                )

    def _link_source_files_to_issue(self, issue_id: str, issue_content: str):
        """Links source files mentioned in the issue content to the issue node."""
        if not issue_content:
            return

        # Correctly initialize TextAnalyzer and use its extract_matches method
        text_analyzer = TextAnalyzer(self.github_token) 
        # Assuming extract_matches takes the text and repo_name (or repo_path if more appropriate)
        # And that its return format is suitable for source_files
        source_files = text_analyzer.extract_matches(issue_content, self.config['repo_name']) 
        
        # Logic to handle multiple language extensions
        # If self.language_config is set (meaning a primary language is defined for the run)
        if hasattr(self, 'language_config') and self.language_config:
            try:
                # Access the 'config' attribute which holds the loaded configuration
                current_lang_extensions = self.language_config.config['file_extensions']
            except KeyError:
                print(f"Warning: 'file_extensions' not found in language_config for {self.language_config.language}")
                current_lang_extensions = []
        else: # Fallback or if no specific language config is primary
            current_lang_extensions = [ext for lang_config in EXT_LANG_MAP.values() 
                                       for ext in LanguageConfigFactory.get_config(lang_config).config['file_extensions']]
            current_lang_extensions = list(set(current_lang_extensions)) # Unique extensions


        linked_files_count = 0
        # Ensure source_files is iterable (list) before the loop
        if not isinstance(source_files, list):
            print(f"Warning: source_files is {type(source_files)} before final loop. Attempting to use keys if dict, else treating as empty.")
            if isinstance(source_files, dict):
                # Assuming keys of the dict are the file paths if it's a dict
                # This is a guess; actual structure of dict from extract_matches matters
                source_files = list(source_files.keys()) 
            else:
                source_files = [] # Treat as empty if not list or dict

        for file_path_ref in source_files:
            if any(file_path_ref.endswith(ext) for ext in current_lang_extensions):
                candidate_paths = [
                    file_path_ref,
                    os.path.join(self.config['repo_path'], file_path_ref),
                ]
                basename = os.path.basename(file_path_ref)
                if basename:
                    for root, _, files in os.walk(self.config['repo_path']):
                        if basename in files:
                            candidate_paths.append(os.path.join(root, basename))

                for candidate_path in candidate_paths:
                    if os.path.exists(candidate_path):
                        clean_file_path = self._clean_path(candidate_path)
                        print(f"Found file reference in Issue #{issue_id}: {clean_file_path}")
                        # Create file entity and establish association
                        self.kg.create_file_entity(clean_file_path)
                        self.kg.link_issue_to_file(issue_id, clean_file_path)
                        self._build_file_class_methods(candidate_path) # This uses self.parser
                        linked_files_count += 1
                        break

        print(f"Found source file references in Issue #{issue_id} (filtered by current language): {linked_files_count}")

    def _search_method_by_name(self, repo_root, method_name):
        cache_key = f"{repo_root}:{method_name}"
        if cache_key in self.method_search_cache:
            print(f"Retrieved method {method_name} search results from cache")
            return self.method_search_cache[cache_key]

        matching_files = []
        print(f"Searching methods or global variables {method_name} in repository {repo_root}")

        # Get all supported extensions from the language factory
        all_supported_extensions = list(EXT_LANG_MAP.keys())

        cnt = 0
        max_matches = 8 if self.language_config.language == 'verilog' else 20
        scan_limit = 80 if self.language_config.language == 'verilog' else 20
        # Iterate through all files in the repository
        for root, dirs, files in os.walk(repo_root):
            # Skip .git directory and other common non-code directories if needed
            dirs[:] = [
                d for d in dirs
                if d not in ['.git', '__pycache__', 'node_modules', 'build', 'dist']
                and not self._is_verilog_noise_path(os.path.join(root, d))
            ]
            for file_name in files:
                # Check if the file extension is one of the supported ones
                if not any(file_name.endswith(ext) for ext in all_supported_extensions):
                    continue

                file_path = os.path.join(root, file_name)
                if self._is_verilog_noise_path(file_path):
                    continue
                
                # Skip test files if not relevant (can be made more configurable)
                # This simple check might need to be adapted based on language-specific test file patterns
                if 'test' in file_path.lower() and 'pytest' not in file_path.lower(): 
                    # More robust check would use parser.language_config.get_config().get('test_file_pattern')
                    # but self.parser might not be the right one for the current file_path here.
                    # For now, a simple heuristic.
                    continue

                parser = self._parser_for_file(file_path)
                if not parser:
                    # print(f"No parser for {file_path}, skipping in _search_method_by_name")
                    continue

                # Get language-specific search patterns
                search_patterns = parser.language_config.get_search_patterns(method_name)
                if not search_patterns:
                    continue
                
                try:
                    content = read_file(file_path) # Assuming read_file reads the entire file as a string
                    # Quick case-insensitive check to avoid regex on irrelevant files
                    if method_name.lower() not in content.lower():
                        continue

                    # Check each rule for the current language
                    found_in_file = False
                    for rule_type, pattern in search_patterns.items():
                        if self.language_config.language == 'verilog' and rule_type == 'comment':
                            continue
                        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                            print(f"Found matching {rule_type} rule: {pattern}, file: {file_path}")
                            matching_files.append({
                                'path': file_path,
                                'type': rule_type, # This type is from the perspective of the regex rule
                                '_score': self._search_match_score(file_path, method_name, rule_type),
                            })
                            cnt += 1
                            found_in_file = True
                            break # Found a match in this file, move to the next file
                    
                except Exception as e:
                    print(f"Error reading or searching file {file_path}: {e}")
                    continue
                
                if cnt >= scan_limit:
                    print(f"Search limit of {cnt} reached, stopping search.")
                    break  # Break from inner loop (files in current directory)
            if cnt >= scan_limit:
                break # Break from outer loop (os.walk)

        matching_files.sort(key=lambda item: (-int(item.get('_score') or 0), len(item.get('path') or ''), item.get('path') or ''))
        matching_files = matching_files[:max_matches]
        for item in matching_files:
            item.pop('_score', None)
        print(f"Completed searching methods or global variables {method_name} in repository {repo_root}, found {len(matching_files)} files after processing {cnt} matches.")
        self.method_search_cache[cache_key] = matching_files
        return matching_files

    def _search_match_score(self, file_path, method_name, rule_type):
        rel = os.path.relpath(file_path, self.config.get('repo_path', self.repo_path)).replace('\\', '/').lower()
        base = os.path.splitext(os.path.basename(rel))[0].lower()
        target = str(method_name or '').lower()
        score = 0
        if base == target:
            score += 120
        elif target and target in base:
            score += 45
        if f"/{target}." in f"/{rel}" or f"/{target}/" in f"/{rel}/":
            score += 60
        if '/rtl/' in rel or rel.startswith('rtl/'):
            score += 15
        if '/libs/rtl/' in rel or '/lib/rtl/' in rel:
            score += 20
        if '/src/' in rel:
            score += 5
        score += {
            'module': 90,
            'interface': 90,
            'package': 70,
            'signal_declaration': 45,
            'signal_connection': 40,
            'instance': 35,
            'procedural_block': 20,
            'function': 15,
        }.get(rule_type, 0)
        return score

    def _is_short_verilog_reference(self, ref_type, target_name):
        return self._is_contextual_verilog_signal_reference(ref_type, target_name)

    def _is_contextual_verilog_signal_reference(self, ref_type, target_name):
        if getattr(self.language_config, 'language', '') != 'verilog':
            return False
        target = str(target_name or '').strip()
        if ref_type == 'signal' and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', target):
            return True
        return (
            len(target) <= 3
            and '_' not in target
            and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', target or '')
            and ref_type in {'variable', 'global'}
        )

    def _search_short_signal_by_issue_context(self, repo_root, signal_name, issue_refs):
        context_files = self._issue_context_files_for_short_signal(repo_root, signal_name, issue_refs)
        matches = []
        seen = set()
        token_pattern = re.compile(rf'\b{re.escape(signal_name)}\b')
        for rank, file_path in enumerate(context_files):
            if file_path in seen or not os.path.exists(file_path):
                continue
            seen.add(file_path)
            if self._is_verilog_noise_path(file_path):
                continue
            try:
                content = read_file(file_path)
            except Exception:
                continue
            clean_content = re.sub(r"/\*.*?\*/", " ", content, flags=re.DOTALL)
            clean_content = re.sub(r"//.*", " ", clean_content)
            if not token_pattern.search(clean_content):
                continue
            match_type = self._classify_verilog_name_match(clean_content, signal_name)
            if match_type == 'symbol_use':
                # A short token use without a declaration is usually too weak for a
                # localization edge.  It can still enter through a nearby declared
                # signal in another context file.
                continue
            matches.append({
                'path': file_path,
                'type': match_type,
                '_score': self._search_match_score(file_path, signal_name, match_type) + max(0, 80 - rank),
            })
        matches.sort(key=lambda item: (-int(item.get('_score') or 0), len(item.get('path') or ''), item.get('path') or ''))
        matches = matches[:6]
        for item in matches:
            item.pop('_score', None)
        print(
            f"Context search for short signal {signal_name} considered {len(context_files)} files "
            f"and kept {len(matches)} matches."
        )
        return matches

    def _issue_context_files_for_short_signal(self, repo_root, signal_name, issue_refs):
        context_files = []
        seen = set()

        def add_file(path):
            if not path:
                return
            path = os.path.normpath(path)
            if not os.path.isabs(path):
                if not os.path.exists(path):
                    path = os.path.join(repo_root, path)
            path = os.path.normpath(path)
            if path in seen or not os.path.exists(path) or os.path.isdir(path):
                return
            if self._is_verilog_noise_path(path):
                return
            seen.add(path)
            context_files.append(path)

        extensions = tuple(getattr(self.language_config, 'config', {}).get('file_extensions', []) or [])
        for ref_type, ref_value in issue_refs or []:
            ref_value = str(ref_value or '').strip()
            if not ref_value or ref_value == signal_name:
                continue
            low = ref_value.lower()
            if extensions and low.endswith(extensions):
                add_file(ref_value)
                add_file(os.path.join(repo_root, ref_value))
                continue
            if self._is_contextual_verilog_signal_reference(ref_type, ref_value):
                continue
            if (
                len(ref_value) <= 3
                and '_' not in ref_value
                and '/' not in ref_value
                and '.' not in ref_value
            ):
                continue
            for file_match_info in self._search_method_by_name(repo_root, ref_value)[:4]:
                add_file(file_match_info.get('path'))
        return context_files[:16]

    def _classify_verilog_name_match(self, content, name):
        escaped = re.escape(name)
        if re.search(rf'\b(?:module|interface|package|program)\s+{escaped}\b', content):
            return 'module'
        if re.search(rf'\b(?:wire|reg|logic|input|output|inout|parameter|localparam|bit|int|integer)\b[^;]*\b{escaped}\b', content):
            return 'signal_declaration'
        if re.search(rf'\.{escaped}\s*\(', content):
            return 'signal_connection'
        if re.search(rf'\b{escaped}\b\s*(?:<=|=)', content):
            return 'procedural_block'
        if re.search(rf'\b{escaped}\b', content):
            return 'symbol_use'
        return 'unknown'

    def _process_repository(self, target_sample):
        """Process code repository"""
        # Before processing problem description, first parse and add patch content
        problem_statement = self.patch_link_expander._expand_patch_links(target_sample['problem_statement'])
        
        # Update target_sample content
        target_sample['problem_statement'] = problem_statement
        
        # Switch to specified commit
        tinfo(f"Checkout commit {target_sample['base_commit'][:8]}")
        self._checkout_commit(target_sample['base_commit'])
        # Convert creation time to UTC timestamp
        self.created_at = (
            datetime.strptime(target_sample['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
        ).timestamp()

        # 0. Create directory structure
        tstage(Stage.KG, "Building directory structure...")
        self.kg.create_directory_structure(self.config['repo_path'], self, False, STRONG_CONNECTION)

        # 1. Create root node, containing problem description and hint information
        root_id = 'root'
        root_content = f"{target_sample['problem_statement']}"

        # Extract title from first line of problem description
        title = target_sample['problem_statement'].split('\n')[0].strip()
        tstage(Stage.KG, f"Root issue: {title[:60]}")
        
        # Create root node
        self.kg.create_issue_entity(
            root_id,
            title,
            '\n'.join(root_content.split('\n')[1:]),        # content
            self.created_at,  # created_at
            #datetime.now(timezone.utc).timestamp(),  # created_at
            "open",             # state
            False,              # is_pr
            "root"        # name
        )
        # The root task itself is always considered a valid item for context.
        if "root_task" not in self.counted_valid_artifact_ids:
             self.artifact_stats["valid_related_items"] += 1
             self.counted_valid_artifact_ids.add("root_task") # Special ID for the root task itself
        
        localization_mode = os.getenv('KGVERI_LOCALIZATION_MODE', 'rtl_causal_b2').strip().lower()
        use_rtl_anchor_discovery = (
            self.language_config.language == 'verilog'
            and localization_mode not in {'kgcompass_b0', 'b0', 'legacy'}
        )
        if use_rtl_anchor_discovery:
            discovery_limit = int(os.getenv('KGVERI_DISCOVERY_FILES', '8'))
            discovered = discover_rtl_source_files(
                self.config['repo_path'],
                root_content,
                limit=discovery_limit,
            )
            self.rtl_source_discovery = [item.to_dict() for item in discovered]
            if discovered:
                tinfo(
                    "RTL anchor discovery: "
                    + ", ".join(item.file_path for item in discovered)
                )
            for item in discovered:
                absolute_path = os.path.join(self.config['repo_path'], item.file_path)
                self._build_file_class_methods(absolute_path)
                try:
                    self.kg.link_issue_to_file(root_id, item.file_path, weight=STRONG_CONNECTION)
                except Exception as exc:
                    tinfo(f"Unable to link discovered RTL file {item.file_path}: {exc}")

        # Extract file references and methods from root node content
        if root_content:
            # Extract reference information
            self._link_source_files_to_issue(root_id, root_content)
            self._link_stacktrace_to_issue(root_id, root_content)
            if use_rtl_anchor_discovery and self.rtl_source_discovery:
                tinfo("RTL anchor source discovery succeeded; skipping legacy name-search expansion")
            else:
                self._link_reference_to_issue_faster(root_id, root_content, 1)

        # 2-5. Local benchmark: skip external GitHub issue search
        tinfo("Local benchmark — skipping external issue search")

        # 6. Process method call relationships
        if self.language_config.language == 'verilog' and localization_mode not in {'kgcompass_b0', 'b0', 'legacy'}:
            tstage(Stage.LOCATE, "Using AST/static RTL relationships")
        else:
            tstage(Stage.LOCATE, "Scanning method call relationships...")
            self._scan_project_for_related_methods(self.kg.get_all_methods(self.MAX_CANDIDATE_METHODS))
        tok("KG construction complete")

    def _has_explicit_verilog_file_reference(self, text):
        extensions = tuple(
            str(ext).lower()
            for ext in (getattr(self.language_config, 'config', {}).get('file_extensions', []) or [])
        )
        if not extensions:
            return False
        pattern = r"(?:[A-Za-z0-9_.-]+[/\\])*[A-Za-z0-9_.-]+(?:" + "|".join(
            re.escape(ext) for ext in extensions
        ) + r")(?::\d+)?"
        return bool(re.search(pattern, str(text or ''), re.IGNORECASE))

    def _scan_project_for_related_methods(self, all_methods):
        """Scan project to find methods with call relationships to existing methods"""
        
        # Use all supported extensions from EXT_LANG_MAP
        all_supported_extensions = list(EXT_LANG_MAP.keys())
        source_files = get_source_files_by_extensions(self.config['repo_path'], all_supported_extensions)
        repo_root = os.path.abspath(self.config['repo_path'])
        skip_parts = {
            '.git', '__pycache__', 'node_modules', 'build', 'dist', 'out',
            'bazel-out', '.cache', 'coverage', 'generated', 'gen',
        }
        skip_prefixes = (
            'tests/', 'test/', 'tb/', 'dv/uvm/', 'dv/tests/', 'dv/cov/',
            'vendor/google_riscv-dv/', 'docs/', 'doc/',
        )
        filtered_files = []
        for file_path in source_files:
            abs_path = os.path.abspath(file_path)
            rel_path = os.path.relpath(abs_path, repo_root).replace('\\', '/')
            parts = {part.lower() for part in rel_path.split('/')}
            rel_lower = rel_path.lower()
            if parts & skip_parts:
                continue
            if any(rel_lower.startswith(prefix) for prefix in skip_prefixes):
                continue
            filtered_files.append(abs_path)
        source_files = filtered_files
        total_files = len(source_files)
        
        for idx, file_path in enumerate(source_files, 1):
            parser = self._parser_for_file(file_path)
            if parser is None:
                continue
            
            # Example of skipping specific paths - make this more configurable if needed
            # if '/rubi/rules/' in file_path: 
            #     continue
            try:
                imports = parser.get_imports(file_path)
                local_methods = []
                global_methods = parser.get_global_methods(file_path, self.config['repo_root'])
                local_methods.extend(global_methods)
                
                classes = parser.extract_classes(file_path)
                for class_info in classes:
                    for method in class_info.get('methods', []):
                        local_methods.append(method)
                
                for local_method_info in local_methods:
                    # The analyze_method_calls_in_method is part of the parser interface
                    # and should be implemented by each language-specific parser.
                    parser.analyze_method_calls_in_method(
                        local_method_info,
                        all_methods, 
                        self.kg,
                        imports,
                        self.config['repo_root'] # repo_root might be specific to Python's module naming
                    )
            except Exception as e:
                print(f"Error processing file {file_path} for method calls: {str(e)}")
                # import traceback
                # print(traceback.format_exc())
                continue

    def _cleanup(self):
        """Clean up resources"""
        # First clean working directory
        try:
            print("Cleaning working directory...")
            # Force clean untracked files
            self.repo.git.clean('-fd')
            # Abandon local modifications
            self.repo.git.reset('--hard')
            print("Working directory cleaned")                    
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print('Cleanup completed')

    def _checkout_commit(self, commit_hash):
        """
        Switch to specified commit
        
        Args:
            commit_hash (str): Commit hash to switch to
        """
        try:
            print(f"Switching to commit: {commit_hash}")
            # Use -f parameter to force switch, will discard all local changes
            self.repo.git.checkout('-f', commit_hash)
            print(f"Successfully switched to commit: {commit_hash}")
            
        except Exception as e:
            print(f"Error switching commit: {e}")
            print(traceback.format_exc())
            raise


def main():
    if len(sys.argv) < 4:
        print("Usage: fl.py <instance_id> <repo_path> <output_dir> [benchmark_name]")
        sys.exit(1)
    instance_id = sys.argv[1]
    repo_path = sys.argv[2]
    output_dir = sys.argv[3]
    benchmark_name = sys.argv[4] if len(sys.argv) > 4 else "verilog-local"

    config = {
        "repo_path": repo_path,
        "repo_root": repo_path,
        "repo_name": "",
        "instance_id": instance_id,
        "benchmark_name": benchmark_name,
        "language": "verilog",
    }
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze()
    if result:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{instance_id}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    analyzer._cleanup()


if __name__ == "__main__":
    main()
