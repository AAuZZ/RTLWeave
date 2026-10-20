import ast
import difflib
import json
import os
import traceback
import chardet
import re
from collections import defaultdict, OrderedDict
from github import Github
import tokenize
from io import BytesIO
import tempfile
import shutil
import uuid
import threading
from collections import defaultdict
import subprocess
import io
import tokenize
try:
    from term import warn as twarn
    from config import SEARCH_SPACE
except Exception:  # pragma: no cover - package import path
    from .term import warn as twarn
    from .config import SEARCH_SPACE

try:
    from verilog_timing import classify_verilog_entity_timing
except Exception:  # pragma: no cover - fallback for package-relative imports.
    from .verilog_timing import classify_verilog_entity_timing

repo_locks = defaultdict(threading.Lock)


def _compact_text(text: str, limit: int = 1200) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."

def normalize_repo_path(file_path: str, repo_root: str = "") -> str:
    """Normalize a repo path to a stable repository-relative form."""
    path = os.path.normpath(str(file_path or "")).replace('\\', '/')
    if not path:
        return ""

    if repo_root:
        try:
            repo_root_abs = os.path.abspath(repo_root).replace('\\', '/')
            path_abs = os.path.abspath(path).replace('\\', '/')
            if path_abs.startswith(repo_root_abs.rstrip('/') + '/'):
                path = os.path.relpath(path_abs, repo_root_abs).replace('\\', '/')
        except ValueError:
            pass
    elif os.path.isabs(path):
        try:
            path = os.path.relpath(path, os.getcwd()).replace('\\', '/')
        except ValueError:
            pass

    if path.startswith('workdirs/'):
        parts = path.split('/')
        if len(parts) > 4 and parts[2] == 'repos':
            return '/'.join(parts[4:])

    for marker in ('playground', 'verilog_repair_cases'):
        prefix = marker + '/'
        if path.startswith(prefix):
            parts = path.split('/')
            if len(parts) > 2:
                return '/'.join(parts[2:])
            return ''
        marker_idx = path.find('/' + prefix)
        if marker_idx >= 0:
            parts = path[marker_idx + 1:].split('/')
            if len(parts) > 2:
                return '/'.join(parts[2:])
            return ''

    return path


def _clean_path(file_path: str) -> str:
    """Backward-compatible wrapper for repository-relative path normalization."""
    rel_path = normalize_repo_path(file_path)
    if rel_path.startswith('workdirs/'):
        parts = rel_path.split('/')
        if len(parts) > 4 and parts[2] == 'repos':
            return '/'.join(parts[4:])
    for marker in ('playground', 'verilog_repair_cases'):
        prefix = marker + '/'
        if rel_path.startswith(prefix):
            parts = rel_path.split('/')
            if len(parts) > 2:
                return '/'.join(parts[2:])
    return rel_path


VERILOG_EDIT_TARGET_KINDS = {
    'module_body',
    'always',
    'always_ff',
    'always_comb',
    'always_latch',
    'initial',
    'assign',
    'function',
    'task',
    'instance',
}

VERILOG_EVIDENCE_LABELS = {
    'signal',
    'port',
    'parameter',
    'macro',
    'state',
    'branch',
    'condition',
    'generateblock',
    'conditionalcompilationscope',
    'testbench',
    'assertion',
}


def context_entity_role(item, fallback_group=None):
    explicit = str(
        item.get('entity_role')
        or item.get('repair_role')
        or item.get('role')
        or ''
    ).strip().lower()
    if explicit:
        return explicit

    group = str(fallback_group or '').strip().lower()
    if group in {'methods', 'patchable_spans', 'edit_targets'}:
        return 'edit_target'
    if group in {'rtl_entities', 'evidence_entities'}:
        return 'evidence'
    if group == 'issues':
        return 'issue'

    item_type = str(item.get('type') or item.get('label') or '').strip().lower()
    verilog_kind = str(item.get('verilog_kind') or '').strip().lower()
    if item_type == 'method' or verilog_kind in VERILOG_EDIT_TARGET_KINDS:
        return 'edit_target'
    if item_type in VERILOG_EVIDENCE_LABELS or verilog_kind in VERILOG_EVIDENCE_LABELS:
        return 'evidence'
    if item_type == 'issue':
        return 'issue'
    return 'context'


def context_entity_priority(item, fallback_group=None):
    try:
        explicit = item.get('priority_score')
        if explicit is not None:
            return float(explicit)
    except (TypeError, ValueError):
        pass

    base = item.get('similarity')
    try:
        base_score = float(base if base is not None else 0.0)
    except (TypeError, ValueError):
        base_score = 0.0

    role = context_entity_role(item, fallback_group)
    item_type = str(item.get('type') or item.get('label') or '').strip().lower()
    verilog_kind = str(item.get('verilog_kind') or '').strip().lower()
    repair_role = str(item.get('repair_role') or '').strip().lower()
    parse_confidence = item.get('parse_confidence')
    distance = item.get('distance')
    path = item.get('path') or []

    if repair_role in {'direct_driver', 'top_level_wiring', 'config_register', 'side_evidence'}:
        role = repair_role

    if role == 'edit_target':
        kind_boost = {
            'module_body': 1.40,
            'always_comb': 1.24,
            'always_ff': 1.22,
            'always_latch': 1.20,
            'always': 1.18,
            'function': 1.14,
            'task': 1.10,
            'instance': 1.06,
            'assign': 1.02,
        }.get(verilog_kind, 1.0)
    elif role == 'evidence':
        kind_boost = {
            'signal': 1.18,
            'port': 1.16,
            'state': 1.14,
            'branch': 1.08,
            'condition': 1.08,
            'parameter': 1.10,
            'macro': 1.04,
            'generateblock': 0.96,
            'conditionalcompilationscope': 0.94,
            'testbench': 0.88,
            'assertion': 0.92,
        }.get(item_type or verilog_kind, 1.0)
    elif role == 'issue':
        kind_boost = 0.82
    elif role == 'direct_driver':
        kind_boost = 1.10
    elif role == 'top_level_wiring':
        kind_boost = 1.15
    elif role == 'config_register':
        kind_boost = 1.08
    elif role == 'side_evidence':
        kind_boost = 0.92
    else:
        kind_boost = 1.0

    confidence_boost = 1.0
    try:
        confidence_boost += min(max(float(parse_confidence), 0.0), 1.0) * 0.12
    except (TypeError, ValueError):
        pass

    distance_penalty = 1.0
    try:
        distance_penalty = 1.0 / (1.0 + max(float(distance), 0.0) * 0.08)
    except (TypeError, ValueError):
        pass

    path_boost = 1.0
    if path:
        path_boost += min(len(path), 4) * 0.03

    timing_boost = 1.0
    try:
        timing_priority = item.get('timing_priority')
        if timing_priority is not None:
            timing_boost += min(max(float(timing_priority), 0.0), 10.0) * 0.01
    except (TypeError, ValueError):
        pass
    timing_tags = item.get('timing_tags') or []
    if isinstance(timing_tags, str):
        timing_tags = [tag.strip() for tag in timing_tags.split(',') if tag.strip()]
    if 'immediate_observable' in timing_tags:
        timing_boost += 0.10
    if 'registered_output' in timing_tags:
        timing_boost += 0.08
    if 'edge_sensitive' in timing_tags:
        timing_boost += 0.05

    return base_score * kind_boost * confidence_boost * distance_penalty * path_boost * timing_boost


def context_entity_sort_key(item, fallback_group=None):
    item_type = str(item.get('type') or item.get('label') or '').strip().lower()
    if str(fallback_group or '').lower() == 'patchable_spans' or item_type in {'patchable_span', 'patchablespan'}:
        try:
            similarity = float(item.get('similarity') or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        try:
            distance = float(item.get('distance') or 0.0)
        except (TypeError, ValueError):
            distance = 0.0
        return (
            -similarity,
            distance,
            str(item.get('file_path') or ''),
            int(item.get('start_line') or 0),
            str(item.get('name') or ''),
        )
    priority = context_entity_priority(item, fallback_group)
    try:
        parse_confidence = float(item.get('parse_confidence') or 0.0)
    except (TypeError, ValueError):
        parse_confidence = 0.0
    try:
        distance = float(item.get('distance') or 0.0)
    except (TypeError, ValueError):
        distance = 0.0
    file_path = str(item.get('file_path') or '')
    try:
        start_line = int(item.get('start_line') or 0)
    except (TypeError, ValueError):
        start_line = 0
    name = str(item.get('name') or item.get('signature') or '')
    return (-priority, -parse_confidence, distance, file_path, start_line, name)

class TextAnalyzer:
    def __init__(self, github_token):
        self.g = Github(github_token)
        self.cache = defaultdict(dict)
        self.patterns = {
            'issue_numbers': r'#(\d+)',
            'github_files': r'(?:blob|tree)/[^/\s]+/[^/\s]+',
            'github_links': r'https://github\.com\S+',
            'django_links': r'https://code\.djangoproject\.com\S+',
            'source_files': self._get_source_file_pattern(),
            'python_files': self._get_python_file_pattern()
        }
        self.pattern_counts = defaultdict(int)

    def _get_source_file_pattern(self):
        source_file_patterns = [
            r'(?:\.{0,2}/)?(?:[\w.-]+/)*[\w.-]+\.(?:py|java|cpp|cc|cxx|h|hpp|v|sv|vh|svh)\b',
            r'(?:src|rtl|include|lib|hdl|tb|test|tests)/(?:[\w.-]+/)*[\w.-]+\.(?:v|sv|vh|svh)\b',
        ]
        return '|'.join(f'({p})' for p in source_file_patterns)

    def _get_python_file_pattern(self):
        python_file_patterns = [
            r'[_\w-]+\.py\b',
            r'(?:\.{0,2}/)?(?:[\w-]+/)*[\w-]+\.py\b',
            r'(?:tests?|src|lib|apps?|modules?|examples?|scripts?)/(?:[_\w-]+/)*[\w-]+\.py\b',
            r'__\w+__\.py\b'
        ]
        return '|'.join(f'({p})' for p in python_file_patterns)

    def extract_matches(self, text, repo_name):
        file_matches = []
        
        for pattern_name, pattern in self.patterns.items():
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                if pattern_name in {'python_files', 'source_files'}:
                    found = [v for tup in found for v in tup if v.strip()]
                if found:
                    self.pattern_counts[pattern_name] += 1
                    if pattern_name in {'python_files', 'source_files'}:
                        file_matches.extend(found)
        
        seen = set()
        ordered = []
        for item in file_matches:
            normalized = item.strip('` "\'').replace('\\', '/')
            if normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
        return ordered

def get_pr_file_line_belongs(pull_request, repo_root, file_path, start_line, end_line, parser=None):
    belongs_to = {
        'classes': [],
        'methods': []
    }
    if '/tests/' in file_path:
        return belongs_to
    try:
        commit = pull_request.head.sha
        repo = pull_request.base.repo
        relative_path = os.path.relpath(file_path, repo_root)
        file_content = repo.get_contents(relative_path, ref=commit).decoded_content.decode('utf-8')
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(file_path))
        try:
            try:
                ast.parse(file_content)
            except SyntaxError as e:
                twarn(f"Python 2 syntax ({file_path}), attempting conversion")
                patterns = [
                    (r'(?m)^(\s*)print\s+([^(].*?)$', r'\1print(\2)'),
                    (r'(?m)^(\s*)print\s*$', r'\1print()'),
                    (r'(?m)^(\s*)print\s+"([^"]*?)"\s*%\s*(.+?)$', r'\1print("\2" % \3)'),
                    (r"(?m)^(\s*)print\s+'([^']*?)'\s*%\s*(.+?)$", r"\1print('\2' % \3)"),
                    (r'(?m)^(\s*)print\s+([^(].*?),\s*$', r'\1print(\2, end=" ")'),
                    (r'(?m)^(\s*)print\s+([^("].*?),\s*(.*?)$', r'\1print(\2, \3)'),
                    (r'`([^`]+)`', r'str(\1)'),
                    (r'\s*<>\s*', ' != '),
                    (r'(\w+)\.has_key\((.*?)\)', r'\1 in \2'),
                ]
                for pattern, replacement in patterns:
                    file_content = re.sub(pattern, replacement, file_content)
                file_content = re.sub(
                    r'(?m)^(\s*)print\s+(["\'].*?["\'])\s*%\s*(.+?)$',
                    r'\1print(\2 % \3)',
                    file_content
                )
                file_content = re.sub(
                    r'assert\s+([^,]+),\s*`([^`]+)`',
                    r'assert \1, str(\2)',
                    file_content
                )
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(file_content)
            if parser is not None:
                classes = parser.extract_classes(temp_file)
                methods = parser.get_global_methods(temp_file, repo.name.split('/')[-1])
                methods.extend(parser.get_global_variables(temp_file, repo.name.split('/')[-1]))
            else:
                classes = get_classes_from_file(temp_file, repo.name.split('/')[-1])
                methods = get_global_methods_from_file(temp_file, repo.name.split('/')[-1])
                methods.extend(get_global_variables_from_file(temp_file, repo.name.split('/')[-1]))
            
            clean_file_path = _clean_path(file_path)
            module_path = clean_file_path.replace(os.sep, '.').replace('.py', '')
            
            for class_info in classes:
                if (class_info['start_line'] <= end_line and 
                    class_info['end_line'] >= start_line):
                    
                    simple_class_name = class_info['name'].split('.')[-1]
                    qualified_class_name = f"{module_path}.{simple_class_name}"

                    belongs_to['classes'].append({
                        'name': qualified_class_name,
                        'file_path': clean_file_path,
                        'start_line': class_info['start_line'],
                        'end_line': class_info['end_line'],
                        'source_code': class_info['source_code'],
                        'doc_string': class_info['doc_string'],
                    })
                    for method in class_info.get('methods', []):
                        if (method['start_line'] <= end_line and 
                            method['end_line'] >= start_line):
                            
                            simple_method_name = method['name'].split('.')[-1]
                            qualified_method_name = f"{qualified_class_name}.{simple_method_name}"

                            # Reconstruct signature with the correct qualified name
                            params = re.search(r'\((.*)\)', method['signature'])
                            param_str = params.group(1) if params else ''
                            new_signature = f"{qualified_method_name}({param_str})"

                            belongs_to['methods'].append({
                                'name': qualified_method_name,
                                'file_path': clean_file_path,
                                'signature': new_signature,
                                'start_line': method['start_line'],
                                'end_line': method['end_line'],
                                'source_code': method['source_code'],
                                'doc_string': method['doc_string'],
                            })
            for method in methods:
                if (method['start_line'] <= end_line and 
                    method['end_line'] >= start_line):
                    
                    simple_method_name = method['name'].split('.')[-1]
                    qualified_method_name = f"{module_path}.{simple_method_name}"

                    params = re.search(r'\((.*)\)', method['signature'])
                    param_str = params.group(1) if params else ''
                    
                    new_signature = f"{qualified_method_name}({param_str})"
                    
                    # Special handling for global variables where signature is an assignment
                    if '=' in method['signature']:
                        value_part = method['signature'].split('=', 1)[1].strip()
                        new_signature = f"{qualified_method_name} = {value_part}"

                    belongs_to['methods'].append({
                        'name': qualified_method_name,
                        'file_path': clean_file_path,
                        'signature': new_signature,
                        'start_line': method['start_line'],
                        'end_line': method['end_line'],
                        'source_code': method['source_code'],
                        'doc_string': method['doc_string'],
                    })
            
            return belongs_to
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        twarn(f"获取文件行归属时出错: {e}")
        return belongs_to

_commit_file_cache = {}

def get_commit_file(repo, commit, file_path):
    cache_key = f"{commit.sha}:{file_path}"
    if cache_key in _commit_file_cache:
        return _commit_file_cache[cache_key]    
    file_content = repo.get_contents(file_path, ref=commit.sha)
    content = file_content.decoded_content.decode('utf-8')
    _commit_file_cache[cache_key] = content
    return content

def get_encoding(file):
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']

def read_file(file_name):
    try:
        ec = get_encoding(file_name)
        f = open(file_name, 'r', encoding=ec, errors='ignore')
        codes = ''.join(f.readlines())
        return codes
    except Exception as e:
        raise Exception(e)

def get_source_files_by_extensions(directory, extensions):
    source_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                source_files.append(os.path.join(root, file))
    return source_files

def get_python_files_from_content(repo_path, content, repo_name):
    python_files = re.findall(r'`?([a-zA-Z0-9_/\\]+\.py)`?', content)
    related_files = set()
    for file_name in python_files:
        file_parts = file_name.split('/')
        possible_paths = []
        possible_paths.append(os.path.join(repo_path, file_name))
        possible_paths.append(os.path.join(repo_path, repo_name, file_name))
        find_times = 0
        find_full_path = []
        for i in range(min(3, len(file_parts)), 0, -1):
            partial_parts = file_parts[-i:]
            for root, _, files in os.walk(repo_path):
                if partial_parts[-1] in files:
                    full_path = os.path.join(root, partial_parts[-1])
                    rel_path = os.path.relpath(full_path, repo_path)
                    rel_parts = rel_path.split(os.sep)
                    if (len(rel_parts) >= len(partial_parts) and 
                        rel_parts[-len(partial_parts):] == partial_parts):
                        if find_times == 0 or ('test' in find_full_path and 'test' not in full_path) or ('docs' in find_full_path and 'docs' not in full_path):
                            find_full_path.append(full_path)
                        find_times += 1
        if len(find_full_path) > 0:
            start_index = 0
            if len(find_full_path) > 1 and 'test' in find_full_path[0] and 'docs' in find_full_path[0]:
                start_index = 1
            for i in range(start_index, len(find_full_path)):
                possible_paths.append(find_full_path[i])
        if os.path.isabs(file_name):
            possible_paths.append(file_name)
        possible_paths = list(set(possible_paths))
        for possible_path in possible_paths:
            if os.path.exists(possible_path):
                related_files.add(possible_path)
    return list(related_files)[:SEARCH_SPACE]

def get_global_methods_from_file(file_path, repo_name):
    clean_file_path = _clean_path(file_path)
    content = read_file(file_path)
    tree = ast.parse(content)
    module_path = clean_file_path.replace(os.sep, '.').replace('.py', '')
    methods = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            method_name = node.name
            params = [a.arg for a in node.args.args]
            method_signature = f"{module_path}.{method_name}({', '.join(params)})"
            doc_string = ast.get_docstring(node) or ''
            methods.append({
                "name": f"{module_path}.{method_name}",
                "signature": method_signature,
                'file_path': clean_file_path,
                "start_line": node.lineno,
                "source_code": ast.get_source_segment(content, node),
                "end_line": node.end_lineno if hasattr(node, 'end_lineno') else None,
                "doc_string": doc_string,
            })
    return methods

def format_path_info(item):
        if 'path' not in item:
            return ''
        path_info = '- path_info : '
        first = True
        for path in reversed(item['path']):
            if first:
                path_info += f"{path['end_node']}"
                first = False
            relation_type = path.get('type') or path.get('relation_kind') or 'RELATED'
            description = path.get('description') or path.get('relation_kind') or ''
            path_info += f" -> {relation_type}:{description} -> {path['start_node']}"
        return path_info

def format_entity_content(item, show_path=False):
    """
    Format complete content for a single entity
    """
    # simplified from reference, can be expanded if path_info is needed
    path_info = format_path_info(item)
    signature = item.get('signature') or item.get('name', '')
    kind = item.get('verilog_kind') or item.get('type') or ''
    semantic_summary = item.get('semantic_summary') or ''
    parse_source = item.get('parse_source') or ''
    parse_confidence = item.get('parse_confidence')
    repair_role = item.get('repair_role') or ''
    timing_tags = item.get('timing_tags') or []
    timing_priority = item.get('timing_priority')
    timing_summary = item.get('timing_summary') or ''
    source_code = item.get('source_code') or item.get('declaration') or ''
    if kind == 'module_body' and source_code:
        source_code = _compact_text(source_code, 1200)
    metadata = ""
    if kind:
        metadata += f"- verilog_kind : {kind}\n"
    if repair_role:
        metadata += f"- repair_role : {repair_role}\n"
    if timing_tags:
        if isinstance(timing_tags, str):
            timing_tags = [tag.strip() for tag in timing_tags.split(',') if tag.strip()]
        metadata += f"- timing_tags : {', '.join(timing_tags)}\n"
    if timing_priority is not None:
        metadata += f"- timing_priority : {timing_priority}\n"
    if timing_summary:
        metadata += f"- timing_summary : {timing_summary}\n"
    if parse_source:
        metadata += f"- parse_source : {parse_source}\n"
    if parse_confidence is not None:
        metadata += f"- parse_confidence : {parse_confidence}\n"
    if semantic_summary:
        metadata += f"- semantic_summary : {semantic_summary}\n"
    if not show_path:
        return (f"### {item.get('file_path', '')}\n"
                f"- signature : {signature}\n"
                f"{metadata}"
                f"- start_line : {item.get('start_line')}\n"
                f"- end_line : {item.get('end_line')}\n"
                f"...\n{source_code}\n...\n\n")
    else:
        return (f"### {item.get('file_path', '')}\n"
                f"- signature : {signature}\n"
                f"{metadata}"
                f"{path_info}\n"
                f"- start_line : {item.get('start_line')}\n"
                f"- end_line : {item.get('end_line')}\n"
                f"...\n{source_code}\n...\n\n")


def get_global_variables_from_file(file_path, repo_name):
    clean_file_path = _clean_path(file_path)
    content = read_file(file_path)
    tree = ast.parse(content)
    module_path = clean_file_path.replace(os.sep, '.').replace('.py', '')
    variables = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        value = ast.get_source_segment(content, node.value)
                    variables.append({
                        "name": f"{module_path}.{target.id}",
                        "signature": f"{module_path}.{target.id} = {value}",
                        "file_path": clean_file_path,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno if hasattr(node, 'end_lineno') else None,
                        "source_code": ast.get_source_segment(content, node),
                        "doc_string": "",
                    })
    return variables

def get_class_and_method_from_content(content, file_path, repo_name):
    repo_name = repo_name.split('/')[-1]
    temp_file = f'playground/{repo_name}/{file_path}'
    if not os.path.exists(temp_file):
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    with open(temp_file, 'w') as f:
        f.write(content)
    classes = get_classes_from_file(temp_file, repo_name)
    methods = []
    for class_info in classes:
        for method in class_info['methods']:
            methods.append(method)
        class_info['methods'] = []

    methods.extend(get_global_methods_from_file(temp_file, repo_name))
    methods.extend(get_global_variables_from_file(temp_file, repo_name))
    return [classes, methods]

def get_classes_from_file(file_path, repo_name):
    try:
        clean_file_path = _clean_path(file_path)
        content = read_file(file_path)
        comments_map = {}
        current_comments = []
        tokens = tokenize.tokenize(BytesIO(content.encode('utf-8')).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment_text = token.string[1:].strip()
                if current_comments and token.start[0] > current_comments[-1][0] + 1:
                    current_comments = []
                current_comments.append((token.start[0], comment_text))
            elif token.type == tokenize.NL or token.type == tokenize.NEWLINE:
                if current_comments:
                    merged_comment = ' '.join(comment[1] for comment in current_comments)
                    comments_map[current_comments[0][0]] = merged_comment
                    comments_map[current_comments[-1][0] + 1] = merged_comment
                    current_comments = []
            else:
                current_comments = []
        
        path_parts = file_path.split(os.sep)
        try:
            repo_root_index = len(path_parts) - 1 - path_parts[::-1].index(repo_name)
            module_path = '.'.join(path_parts[repo_root_index:]).replace('.py', '')
        except ValueError:
            module_path = os.path.relpath(file_path).replace('/', '.').replace('\\', '.').replace('.py', '')

        tree = ast.parse(content)
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                full_class_name = f"{module_path}.{node.name}"
                class_doc = ast.get_docstring(node) or ''
                if not class_doc:
                    class_doc = comments_map.get(node.lineno, '') or comments_map.get(node.lineno - 1, '')
                class_source_code = ast.get_source_segment(content, node)
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_doc = ''
                        doc_node = ast.get_docstring(item)
                        if doc_node:
                            method_doc = doc_node
                        if not method_doc:
                            method_doc = comments_map.get(item.lineno, '')
                        function_source_code = ast.get_source_segment(content, item)
                        full_method_name = f"{full_class_name}.{item.name}"
                        method_info = {
                            'name': full_method_name,
                            'file_path': clean_file_path,
                            'start_line': item.lineno,
                            'end_line': item.end_lineno,
                            'source_code': function_source_code,
                            'signature': f"{full_method_name}({', '.join(arg.arg for arg in item.args.args)})",
                            'doc_string': method_doc,
                        }
                        methods.append(method_info)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                try:
                                    value = ast.literal_eval(item.value)
                                except (ValueError, SyntaxError):
                                    value = ast.get_source_segment(content, item.value)
                                full_var_name = f"{full_class_name}.{target.id}"
                                var_info = {
                                    'name': full_var_name,
                                    'file_path': clean_file_path,
                                    'start_line': item.lineno,
                                    'end_line': item.end_lineno if hasattr(item, 'end_lineno') else None,
                                    'source_code': ast.get_source_segment(content, item),
                                    'signature': f"{full_var_name} = {value}",
                                    'doc_string': comments_map.get(item.lineno, '') or comments_map.get(item.lineno - 1, ''),
                                }
                                methods.append(var_info)
                class_info = {
                    'name': full_class_name,
                    'file_path': clean_file_path,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno,
                    'doc_string': class_doc,
                    'source_code': class_source_code,
                    'methods': methods,
                }
                classes.append(class_info)
        
        return classes
    except Exception as e:
        twarn(f"Error parsing {file_path}: {e}")
        return []

def extract_code_blocks(markdown_text):
    codes = []
    now_code = None
    for lines in markdown_text.split('\n'):
        if lines.startswith('```'):
            if now_code is not None:
                codes.append(now_code)
                now_code = None
            else:
                now_code = ''
        elif now_code is not None:
            now_code += lines + '\n'
    return codes

def get_reference_functions_from_text(repo_name, text, parser, exclude_set=set()):
    code_references = []
    parser_language = getattr(getattr(parser, 'language_config', None), 'language', '')
    # Extract code blocks (language-agnostic part)
    # Markdown code blocks might specify a language, e.g., ```python ... ``` or ```cpp ... ```
    # For now, we assume all code blocks are of the configured language if not specified otherwise,
    # or that the parser's analyze_snippet_for_references can handle/ignore other languages.
    extracted_blocks = extract_code_blocks(text)
    
    for code_block in extracted_blocks:
        try:
            # Use the passed parser to analyze the snippet
            # This delegates language-specific snippet parsing to the appropriate parser
            snippet_refs = parser.analyze_snippet_for_references(code_block)
            if snippet_refs: # Ensure it's not None or empty
                code_references.extend(snippet_refs)
        except Exception as e:
            # import traceback # Keep commented unless debugging
            # print(traceback.format_exc())
            twarn(f"Error processing code block references: {e}")
            continue
    
    # The rest of this function uses regex and general string matching.
    # Verilog/SystemVerilog issue text is often prose-heavy Markdown.  The
    # Python-oriented fallback below is too broad for HDL repositories because
    # it treats section titles such as "Description" and "Actual" as imports.
    # Use a stricter HDL-specific extractor before the general fallback.
    try:
        EXCLUDE_PATTERNS = {
            'the', 'this', 'that', 'readme', 'todo', 'note', 'warning', 'error', 'pr', 'rfc', 'python', 'py', 'pyc', 'pyo', 'pyd', 'os', 'sys', 'io', 'json', 'self', 'import', 'def', 'try', 'except', 'finally', 'with', 'as', 'if', 'else', 'elif', 'while', 'for', 'in', 'is', 'and', 'or', 'not', 'none', 'true', 'false', 'none', 'null', 'google', 'github', 'community', 'com', 'org', 'www', 'http', 'https', 'hh', 'mm', 'dd', 'uuuuuu', 'do', 'does', 'should', 'please', 'thanks', 'thank', 'wanted', 'want', 'however', 'instead', 'what', 'how', 'when', 'where', 'seems', 'seem', 'patch', 'both', 'name', 'have', 'to', 'be', 'can', 'will', 'may', 'might', 'could', 'would', 'should', 'must', 'need', 'want', 'try', 'use', 'using', 'get', 'take', 'look', 'root', 'google.com', 'github.com', 'docs.djangoproject.com', 'developer', 'already', 'pending', 'looking', 'several', 'java', 'cpp', 'verilog', 'systemverilog', 'module', 'endmodule', 'wire', 'reg', 'logic', 'assign', 'always', 'initial', 'input', 'output', 'clock', 'reset', 'set', 'dict', 'int', 'str', 'float', 'list', 'tuple', 'here', 'you', 'your', '', 'a', 'an', 'i', 'he', 'it', 'they', 'she', 's', 'in', 'out', 'fix', 'of', 'open', 'on', 'off', 
        }
        if parser_language == 'verilog':
            EXCLUDE_PATTERNS.update({
                'firmware', 'expected', 'relevant', 'behavior', 'documented', 'register',
                'interrupt', 'asserted', 'deassert', 'clear', 'cleared', 'clears',
                'overflow', 'sticky', 'control', 'current', 'local', 'path', 'bit',
                'uart', 'fifo', 'soc', 'rtl', 'rx', 'tx', 'description', 'condition',
                'conditions', 'function', 'affected', 'trigger', 'actual', 'memory',
                'physical', 'protection', 'section', 'scenario', 'observed', 'result',
                'results', 'problem', 'summary', 'steps', 'reproduce', 'expected',
                'baseline', 'buggy', 'fixed', 'failure', 'fails', 'failed', 'pass',
                'passes', 'passing', 'implementation', 'configuration', 'config',
                'feature', 'option', 'value', 'values', 'field', 'fields',
                'ibex', 'cva6', 'risc', 'riscv', 'caliptra', 'opentitan',
            })
        def keep_reference(ref):
            ref = str(ref or '').strip()
            if ref.lower() in EXCLUDE_PATTERNS:
                return False
            if parser_language == 'verilog':
                low = ref.lower()
                if re.fullmatch(r'(?:x|f|a|s|t)\d+', low) or low in {'zero', 'ra', 'sp', 'gp', 'tp', 'fp'}:
                    return False
                if low.endswith(('.sv', '.svh', '.v', '.vh')):
                    return True
                if re.fullmatch(r'[A-Z][a-z]+', ref):
                    return False
                if re.fullmatch(r'[A-Z]{2,5}', ref) and '_' not in ref and not re.search(r'\d', ref):
                    return False
                if ref.islower() and '_' not in ref and '/' not in ref and '.' not in ref and not re.search(r'\d', ref):
                    return False
                if len(ref) <= 3 and '_' not in ref and '/' not in ref and '.' not in ref and not re.search(r'\d', ref):
                    return False
                return bool(
                    '_' in ref
                    or '/' in ref
                    or '.' in ref
                    or re.search(r'\d', ref)
                    or re.search(r'[a-z][A-Z]|[A-Z][a-z].*[A-Z]', ref)
                )
            if parser_language == 'verilog' and ref.islower() and '_' not in ref and '/' not in ref and '.' not in ref:
                return False
            if parser_language == 'verilog' and len(ref) <= 3 and '_' not in ref and '/' not in ref and '.' not in ref:
                return False
            return True

        if parser_language == 'verilog':
            hdl_refs = []

            def add_ref(kind, value, allow_short=False):
                value = str(value or '').strip().strip('`"\'')
                if not value:
                    return
                low = value.lower()
                if re.fullmatch(r'(?:x|f|a|s|t)\d+', low) or low in {'zero', 'ra', 'sp', 'gp', 'tp', 'fp'}:
                    return
                if allow_short and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', value):
                    hdl_refs.append((kind, value))
                elif keep_reference(value):
                    hdl_refs.append((kind, value))

            for match in re.finditer(r"`([^`]+)`", text or ""):
                token = match.group(1).strip()
                if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$./-]*', token):
                    if re.search(r'\.(?:sv|svh|v|vh)$', token, re.IGNORECASE):
                        kind = 'import'
                    elif (
                        re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', token)
                        and (
                            (len(token) <= 3 and '_' not in token)
                            or re.search(r'_(?:i|o|q|d|n|ni|no)$', token)
                        )
                    ):
                        kind = 'signal'
                    else:
                        kind = 'call'
                    add_ref(kind, token, allow_short=True)

            for match in re.finditer(r"(?<![\w./-])([A-Za-z0-9_./-]+\.(?:sv|svh|v|vh))\b", text or "", re.IGNORECASE):
                add_ref('import', match.group(1))

            for match in re.finditer(r"\b([A-Z][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z][a-z0-9]+_[A-Za-z0-9_]+|[A-Za-z]+[0-9][A-Za-z0-9_]*|[A-Za-z]+[a-z][A-Z][A-Za-z0-9_]*)\b", text or ""):
                add_ref('call', match.group(1))

            code_references.extend(hdl_refs)
            unique_refs = set(code_references)
            signal_values = {value for kind, value in unique_refs if kind == 'signal'}
            if signal_values:
                unique_refs = {
                    (kind, value) for kind, value in unique_refs
                    if not (value in signal_values and kind != 'signal')
                }
            return list(sorted(
                unique_refs,
                key=lambda x: (
                    x[1].count('/') * 12 + x[1].count('.') * 8 + x[1].count('_') * 3 + len(x[1]),
                    x[1],
                ),
                reverse=True,
            ))[:SEARCH_SPACE]

        module_pattern = fr'\b({repo_name}(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b|\`({repo_name}(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\`'
        module_refs = re.findall(module_pattern, text)
        method_pattern = r'[\.\`\[\'\:]([a-zA-Z_][a-zA-Z0-9_\-]*)'
        method_refs = re.findall(method_pattern, text)
        method_refs = [ref for ref in method_refs if keep_reference(ref)]
        global_pattern = r'\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b'
        global_refs = re.findall(global_pattern, text) 
        global_refs = [ref for ref in global_refs if keep_reference(ref)]
        self_pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
        self_refs = re.findall(self_pattern, text)
        module_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.[\w]'
        new_module_refs = [ref for ref in re.findall(module_pattern, text) 
                          if keep_reference(ref)]
        module_refs.extend(new_module_refs)
        full_module_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
        full_module_refs = [ref for ref in re.findall(full_module_pattern, text) 
                          if keep_reference(ref)]
        module_refs.extend(full_module_refs)
        for self_ref in self_refs:
            if self_ref in exclude_set:
                continue
            code_references.append(('variable', self_ref))
        for method_ref in method_refs:
            if method_ref in exclude_set:
                continue
            code_references.append(('call', method_ref))
        for global_ref in global_refs:
            if global_ref in exclude_set:
                continue
            code_references.append(('global', global_ref))
        for module_ref in module_refs:
            if module_ref in exclude_set:
                continue
            code_references.append(('import', module_ref))
        for module_ref in full_module_refs:
            if module_ref in exclude_set:
                continue
            code_references.append(('import', module_ref))
        if len(code_references) == 0:
            class_pattern = r'([A-Z][a-zA-Z_]*)'
            class_refs = [ref for ref in re.findall(class_pattern, text) 
                                if ref.lower() not in EXCLUDE_PATTERNS]
            for class_ref in class_refs:
                code_references.append(('call', class_ref))
    except Exception as e:
        twarn(f"Error parsing module references: {e}")
    return list(sorted(set(code_references), key=lambda x: len(x[1]) + x[1].count('.') * 5 + x[1].count('.py') * 10, reverse=True))[:SEARCH_SPACE]

def extract_methods_from_traceback(working_tree_dir, repo_root, content, kg, parser=None):
    methods = []
    # repo_root = working_tree_dir # This line seems to override the passed repo_root, which might be an issue. Consider reviewing.
    # For now, let's keep the original logic of using working_tree_dir as effective repo_root for path construction inside.
    # However, for parser.get_global_methods, the original repo_root (self.config['repo_root']) is used.

    # Regex patterns for stack trace, largely Python-specific.
    # TODO: Generalize these patterns or make them language-specific if trace formats differ significantly.
    stack_patterns = [
        r'File\\s+(?:"([^"]+?\\.py)"|([^:"\\s]+?\\.py))(?:"|,|\\s|:)\\s*(?:line\\s+)?(\\d+)(?:,\\s+|\\s+)in\\s+([^\\s\\(]+)',
        r'([^\\s]+?\\.py)\\s+in\\s+([^\\s\\(]+)', # Simpler pattern
        r'([^\\s"]+?\\.py)\\s+in\\s+([^\\s\\(]+)', # Simpler pattern
        r'File\\s+([^,]+?\\.py),\\s*line\\s+(\\d+)(?:,\\s+|\\s+)in\\s+([^\\s\\(]+)',
        r'([^:]+?\\.py):(\\d+)(?::\\s+|\\s+)in\\s+([^\\s\\(]+)'
    ]
    
    # More generic patterns (experimental - might need refinement)
    generic_stack_patterns = [
        # Catches "File path/to/file.ext, line 123, in some_function"
        r'File\\s+(?:"([^"]+)"|([\\w/\\\\.-]+?\\.[a-zA-Z]+))(?:",|,|:)?\\s*(?:line\\s+)?(\\d+)(?:,\\s+|\\s+)in\\s+([\\w.<>:]+)',
        # Catches "at package.class.method(SourceFile.ext:line)"
        r'at\\s+([\\w.]+)\\(([\\w.-]+\\.[a-zA-Z]+):(\\d+)\\)',
        # Catches "path/to/file.ext:line in function"
        r'([\\w/\\\\.-]+?\\.[a-zA-Z]+):(\\d+):\\s+in\\s+function\\s+([\\w.<>:]+)',
    ]
    
    processed_matches = set() # To avoid processing the same trace line multiple times

    for pattern_set_name, current_patterns in [("generic", generic_stack_patterns), ("python_specific", stack_patterns)]:
        for pattern in current_patterns:
            for match in re.finditer(pattern, content):
                match_tuple = match.groups()
                if match_tuple in processed_matches:
                    continue
                processed_matches.add(match_tuple)

                file_path_from_trace = None
                line_number_str = None
                method_name_from_trace = None

                if pattern_set_name == "generic":
                    if pattern == generic_stack_patterns[0]: 
                        file_path_from_trace = next((g for g in match_tuple[:2] if g is not None), None)
                        line_number_str = match_tuple[2]
                        method_name_from_trace = match_tuple[3]
                    elif pattern == generic_stack_patterns[1]: 
                        method_name_from_trace = match_tuple[0] 
                        file_path_from_trace = match_tuple[1]   
                        line_number_str = match_tuple[2]
                    elif pattern == generic_stack_patterns[2]: 
                        file_path_from_trace = match_tuple[0]
                        line_number_str = match_tuple[1]
                        method_name_from_trace = match_tuple[2]
                else: # python_specific patterns
                    if len(match_tuple) == 4: 
                        file_path_from_trace = next((g for g in match_tuple[:2] if g is not None), None) 
                        if file_path_from_trace is None and len(match_tuple) == 4 : 
                             file_path_from_trace = match_tuple[0]
                             line_number_str = match_tuple[1]
                             method_name_from_trace = match_tuple[2] if len(match_tuple) > 2 else None 
                        else: 
                            line_number_str = match_tuple[2] if len(match_tuple) > 2 else None
                            method_name_from_trace = match_tuple[3] if len(match_tuple) > 3 else None
                    elif len(match_tuple) == 2: 
                        file_path_from_trace = match_tuple[0]
                        method_name_from_trace = match_tuple[1]
                                      
                if not file_path_from_trace or not method_name_from_trace:
                    continue

                potential_file_path = os.path.join(working_tree_dir, file_path_from_trace) 
                if not os.path.exists(potential_file_path):
                    try:
                        search_term_for_kg = os.path.basename(file_path_from_trace)
                        file_entities_from_kg = kg.search_file_by_path(search_term_for_kg)
                        if file_entities_from_kg:
                            selected_entity = None
                            for entity_info in file_entities_from_kg:
                                path_in_kg = entity_info['file']['path']
                                if 'test' not in path_in_kg.lower():
                                    selected_entity = entity_info
                                    break
                            if not selected_entity: 
                                selected_entity = file_entities_from_kg[0]
                            actual_file_path = selected_entity['file']['path']
                        else:
                            continue
                    except Exception:
                        continue
                else:
                    actual_file_path = potential_file_path

                actual_parser = None
                if callable(parser): 
                    actual_parser = parser(actual_file_path) 
                elif parser is not None: 
                    actual_parser = parser
                
                if not actual_parser:
                    continue

                try:
                    classes = actual_parser.extract_classes(actual_file_path)
                    methods_in_file = actual_parser.get_global_methods(actual_file_path, repo_root) 
                    methods_in_file.extend(actual_parser.get_global_variables(actual_file_path, repo_root))
                except Exception as e:
                    twarn(f"Error parsing {actual_file_path}: {e}")
                    continue
                
                found_method_details = None
                target_method_simple_name = method_name_from_trace.split('.')[-1].split('(')[0].replace('<', '').replace('>', '')

                for class_info in classes:
                    for m_detail in class_info['methods']:
                        if m_detail['name'].split('.')[-1] == target_method_simple_name:
                            found_method_details = m_detail
                            break
                    if found_method_details: break
                
                if not found_method_details:
                    for m_detail in methods_in_file:
                        if m_detail['name'].split('.')[-1] == target_method_simple_name:
                            found_method_details = m_detail
                            break
                
                if found_method_details:
                    line_num = int(line_number_str) if line_number_str and line_number_str.isdigit() else found_method_details['start_line']
                    clean_actual_file_path = _clean_path(actual_file_path)
                    method_info_to_add = {
                        'name': found_method_details['name'], 
                        'file_path': clean_actual_file_path,
                        'signature': found_method_details['signature'],
                        'start_line': found_method_details['start_line'], 
                        'end_line': found_method_details['end_line'],   
                        'line_number': line_num, 
                        'source_code': found_method_details.get('source_code', ''),
                        'doc_string': found_method_details.get('doc_string', '')
                    }
                    methods.append(method_info_to_add)
                else:
                    pass

    return methods

def get_ref_ids(repo_name, content):
    refs = set()
    close_refs = re.findall(r'(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)', content, re.IGNORECASE)
    refs.update(close_refs)
    general_refs = re.findall(r'(?<![\w])#(\d+)(?![\w-])', content)
    refs.update(general_refs)
    url_pattern = f"https://github.com/{repo_name}/(?:issues|pull)/(\\d+)"
    url_refs = re.findall(url_pattern, content)
    refs.update(url_refs)
    if repo_name == 'django/django':
        url_pattern = f"https://code.djangoproject.com/ticket/(\\d+)"
        url_refs = re.findall(url_pattern, content)
        refs.update(url_refs)
    return refs

def relative_path(path):
    return normalize_repo_path(path)

def check_syntax(code):
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return False
    return True

def split_edit_multifile_commands(commands) -> dict[str, str]:
    """Split commands based on edited files."""
    file_to_commands = OrderedDict()

    def _repair_collapsed_replace_blocks(text: str) -> str:
        """Recover minor newline loss in replace-only patches from reasoning models."""
        repaired = text or ""
        repaired = re.sub(r"(?<!\n)###\s*", r"\n### ", repaired)
        repaired = re.sub(r"###\s*([^\n]+?)-\s*start_line\s*:", r"### \1\n- start_line:", repaired)
        repaired = re.sub(r"(?<!\n)-\s*start_line\s*:\s*(\d+)\s*-\s*end_line\s*:", r"- start_line: \1\n- end_line:", repaired)
        repaired = re.sub(r"(?<!\n)-\s*end_line\s*:\s*(\d+)\s*<<<<<<<\s*REPLACE", r"- end_line: \1\n<<<<<<< REPLACE\n", repaired)
        repaired = re.sub(r"(?<!\n)>>>>>>>\s*REPLACE", r"\n>>>>>>> REPLACE", repaired)
        return repaired.lstrip("\n")

    for command in commands:
        # normalize [REPLACE]/[/REPLACE] to internal <<<<<<< REPLACE/>>>>>>> REPLACE
        command = command.replace("[REPLACE]", "<<<<<<< REPLACE").replace("[/REPLACE]", ">>>>>>> REPLACE")
        command = _repair_collapsed_replace_blocks(command)
        file_name = None
        start_line = None
        end_line = None

        replace_only_chunks = []
        replace_only_pattern = re.compile(
            r"(###\s+[^\n]+\n(?:-\s*start_line\s*:\s*\d+\n)(?:-\s*end_line\s*:\s*\d+\n))"
            r"<<<<<<<\s*REPLACE\n",
        )
        raw_parts = command.split(">>>>>>> REPLACE")
        for i in range(len(raw_parts) - 1):
            part = raw_parts[i]
            header_match = replace_only_pattern.search(part)
            if not header_match:
                continue
            header = header_match.group(1)
            replacement_start = header_match.end()
            replacement = part[replacement_start:]
            replacement = replacement.strip('\r\n')
            file_match = re.search(r"^###\s+(.+?)\s*$", header, re.MULTILINE)
            start_match = re.search(r"^-\s*start_line\s*:\s*(\d+)\s*$", header, re.MULTILINE)
            end_match = re.search(r"^-\s*end_line\s*:\s*(\d+)\s*$", header, re.MULTILINE)
            if not (file_match and start_match and end_match):
                continue
            replace_file_name = "'" + file_match.group(1).strip() + "'"
            converted_command = {
                'command': "<<<<<<< REPLACE\n" + replacement + "\n>>>>>>> REPLACE",
                'start_line': int(start_match.group(1)),
                'end_line': int(end_match.group(1)),
                'replace_only': True,
            }
            if replace_file_name not in file_to_commands:
                file_to_commands[replace_file_name] = []
            dup_key = (converted_command['start_line'], converted_command['end_line'])
            if not any((cmd.get('start_line'), cmd.get('end_line')) == dup_key for cmd in file_to_commands[replace_file_name]):
                file_to_commands[replace_file_name].append(converted_command)
            replace_only_chunks.append(header_match.span())

        search_replace_text = replace_only_pattern.sub("", command)

        for subcommand in search_replace_text.split(">>>>>>> REPLACE")[:-1]:
            subcommand = subcommand.strip()
            if "### " in subcommand:
                lines = subcommand.split('\n')
                for line in lines:
                    if line.startswith('### '):
                        file_name = "'" + line.replace('### ', '').strip() + "'"
                    elif line.startswith('- start_line'):
                        start_line = int(line.split(':')[1].strip())
                    elif line.startswith('- end_line'):
                        end_line = int(line.split(':')[1].strip())

            if len(subcommand.split("<<<<<<< SEARCH")) != 2:
                continue
                
            converted_command = {
                'command': (
                    "<<<<<<< SEARCH"
                    + subcommand.split("<<<<<<< SEARCH")[1]
                    + "\n"
                    + ">>>>>>> REPLACE"
                ),
                'start_line': start_line,
                'end_line': end_line,
                'replace_only': False,
            }
            
            if file_name not in file_to_commands:
                file_to_commands[file_name] = []
            
            dup_key = (converted_command.get('start_line'), converted_command.get('end_line'))
            if not any((cmd.get('start_line'), cmd.get('end_line')) == dup_key
                      for cmd in file_to_commands[file_name]):
                file_to_commands[file_name].append(converted_command)
                
    return file_to_commands

def parse_diff_edit_commands_strict(commands, content, only_one_replace=False, require_line_range=False):
    def _normalized_lines(text):
        return [line.rstrip() for line in (text or "").splitlines()]

    def _normalized_block(text):
        return "\n".join(_normalized_lines(text)).strip("\n")

    def _similarity(left, right):
        if not left and not right:
            return 1.0
        return difflib.SequenceMatcher(None, left, right).ratio()

    def _find_fuzzy_span(current_content, original_text, start_line=None, end_line=None, window=10):
        original_block = _normalized_block(original_text)
        if not original_block:
            return None

        lines = current_content.splitlines()
        original_lines = original_block.splitlines()
        if not lines or not original_lines:
            return None

        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except (TypeError, ValueError):
            start_line = 0
            end_line = 0

        original_len = len(original_lines)
        candidate_lengths = {original_len}
        if original_len > 1:
            candidate_lengths.update({original_len - 1, original_len + 1})
        if original_len > 2:
            candidate_lengths.update({original_len - 2, original_len + 2})
        candidate_lengths = sorted(length for length in candidate_lengths if length > 0)

        if start_line > 0 and end_line >= start_line:
            anchor_start = max(0, start_line - 1)
            anchor_end = max(anchor_start, end_line - 1)
        else:
            anchor_start = 0
            anchor_end = len(lines) - 1

        search_start = max(0, anchor_start - window)
        search_end = min(len(lines), anchor_end + window + 1)

        best = None
        for span_len in candidate_lengths:
            if span_len > len(lines):
                continue
            max_start = min(search_end - span_len, len(lines) - span_len)
            if max_start < search_start:
                continue
            for start_index in range(search_start, max_start + 1):
                candidate_block = _normalized_block("\n".join(lines[start_index:start_index + span_len]))
                score = _similarity(candidate_block, original_block)
                if best is None or score > best[0]:
                    best = (score, start_index, span_len)

        if best is None or best[0] < 0.90:
            return None
        return best[1], best[2]

    def _replace_line_span(current_content, start_index, span_len, replace_text):
        lines = current_content.splitlines()
        replacement_lines = replace_text.splitlines()
        new_lines = lines[:start_index] + replacement_lines + lines[start_index + span_len:]
        new_content = "\n".join(new_lines)
        if current_content.endswith("\n"):
            new_content += "\n"
        return new_content

    def _find_normalized_span(current_content, original_text):
        original_block = _normalized_block(original_text)
        if not original_block:
            return None
        original_lines = original_block.splitlines()
        lines = current_content.splitlines()
        span_len = len(original_lines)
        if span_len == 0 or span_len > len(lines):
            return None
        for start_index in range(0, len(lines) - span_len + 1):
            candidate_block = _normalized_block("\n".join(lines[start_index:start_index + span_len]))
            if candidate_block == original_block:
                return start_index, span_len
        return None

    def _apply_by_line_range(current_content, start_line, end_line, replace_text, verify_original=True):
        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except (TypeError, ValueError):
            return current_content, False

        if start_line < 1 or end_line < start_line:
            return current_content, False

        lines = current_content.splitlines()
        if end_line > len(lines):
            # Clamp to file end instead of rejecting — LLM line numbers may overshoot
            end_line = len(lines)

        current_block = "\n".join(lines[start_line - 1 : end_line])
        if verify_original and _normalized_block(current_block) != _normalized_block(original):
            return current_content, False

        replacement_lines = replace_text.splitlines()
        new_content = _replace_line_span(current_content, start_line - 1, end_line - start_line + 1, replace_text)
        return new_content, True

    def parse_for_threedots(original, replace, content):
        if replace.startswith("...\n") and len(replace) > 4:
            replace = replace[4:]

        if original == "...":
            if not replace[0].isspace():
                for line in content.splitlines():
                    if len(line) > 0 and not line[0].isspace():
                        if content.count(line) == 1:
                            original = line
                            replace = replace + "\n\n" + line
                            break

                if original == "...":
                    print("Can't find a suitable replacement position")

        if original.startswith("...\n") and len(original) > 4:
            original = original[4:]

        return original, replace

    replaced = False
    original_content = content
    
    def _line_range_sort_key(command):
        try:
            return int(command.get('start_line') or 0)
        except (TypeError, ValueError):
            return 0

    ordered_commands = sorted(commands, key=_line_range_sort_key, reverse=True)

    for command in ordered_commands:
        if command.get('replace_only'):
            if '<<<<<<< REPLACE' not in command['command']:
                continue
            replace = command['command'].split('<<<<<<< REPLACE', 1)[1].split('>>>>>>> REPLACE', 1)[0].strip('\r\n')
            expected_start = command.get('start_line')
            expected_end = command.get('end_line')

            try:
                lines_before = original_content.splitlines()
                # Clamp end_line to file length — LLM may overshoot
                clamped_end = min(expected_end, len(lines_before))
                if 1 <= expected_start <= clamped_end:
                    original = "\n".join(lines_before[expected_start - 1 : clamped_end])
                else:
                    original = ""
            except Exception:
                original = ""

            if original.strip():
                original, replace = parse_for_threedots(original, replace, content)
                fallback_content, applied = _apply_by_line_range(
                    content, expected_start, expected_end, replace, verify_original=True,
                )
                if applied:
                    content = fallback_content
                    replaced = True
                    print(f"Successfully applied replace-only changes via verified line range L{expected_start}-{expected_end}")
                    continue

                if require_line_range:
                    print(f"Replace-only verified line range failed L{expected_start}-{expected_end}, attempting fuzzy fallback")

                normalized_span = _find_normalized_span(content, original)
                if normalized_span is not None:
                    content = _replace_line_span(content, normalized_span[0], normalized_span[1], replace)
                    replaced = True
                    print(f"Successfully applied replace-only via normalized span (requested L{expected_start}-{expected_end})")
                    continue

                fuzzy_span = _find_fuzzy_span(content, original, expected_start, expected_end)
                if fuzzy_span is not None:
                    content = _replace_line_span(content, fuzzy_span[0], fuzzy_span[1], replace)
                    replaced = True
                    print(f"Successfully applied replace-only via fuzzy span (requested L{expected_start}-{expected_end})")
                    continue

                print(f"Replace-only replacement failed L{expected_start}-{expected_end}")
                continue

            fallback_content, applied = _apply_by_line_range(
                content, expected_start, expected_end, replace, verify_original=False,
            )
            if applied:
                content = fallback_content
                replaced = True
                print(f"Successfully applied replace-only changes via line range L{expected_start}-{expected_end}")
                continue

            print(f"Replace-only line range replacement failed L{expected_start}-{expected_end}")
            continue

        search_replace = command['command'].split('<<<<<<< SEARCH')[1]
        search_replace_parts = search_replace.split('\n=======\n')
        if len(search_replace_parts) != 2:
            continue

        original = search_replace_parts[0].strip('\r\n')
        replace = search_replace_parts[1].split('>>>>>>> REPLACE')[0].strip('\r\n')

        original, replace = parse_for_threedots(original, replace, content)
        fallback_content, applied = _apply_by_line_range(
            content,
            command.get('start_line'),
            command.get('end_line'),
            replace,
        )
        if applied:
            content = fallback_content
            replaced = True
            print("Successfully applied changes via line range")
            continue

        if require_line_range:
            print("Line range replacement failed; refusing fallback because line range is required")
            continue

        normalized_span = _find_normalized_span(content, original)
        if normalized_span is not None:
            content = _replace_line_span(content, normalized_span[0], normalized_span[1], replace)
            replaced = True
            print("Successfully applied changes via normalized match")
            continue

        fuzzy_span = _find_fuzzy_span(
            content,
            original,
            command.get('start_line'),
            command.get('end_line'),
        )
        if fuzzy_span is not None:
            content = _replace_line_span(content, fuzzy_span[0], fuzzy_span[1], replace)
            replaced = True
            print("Successfully applied changes via fuzzy span")
            continue

        if original in content:
            content = content.replace(original, replace, 1)
            replaced = True
            print("Successfully applied changes via direct text match")
        else:
            print("Content doesn't match")
            print("Expected content:", original)

    if not replaced:
        print("No changes were made")

    return content

def get_commit_method_by_signature(repo, commit, file_path, method_signature):
    file_content = get_commit_file(repo, commit, file_path)
    _, methods = get_class_and_method_from_content(file_content, file_path, repo.name)
    for method in methods:
        if method_signature in method['signature']:
            return method
    return None

def extract_json_code(code):
    json_code = re.search(r'```json\n(.*?)\n```', code, re.DOTALL)
    if json_code:
        return json_code.group(1)
    return code
