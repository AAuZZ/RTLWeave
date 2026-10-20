from abc import ABC, abstractmethod
import os
import re
from term import warn as twarn

try:
    from verilog_ast import VerilogAstExtractor, VerilogAstUnavailable
    from patchable_projection import PatchableSpanProjector
except Exception:
    try:
        from .verilog_ast import VerilogAstExtractor, VerilogAstUnavailable
        from .patchable_projection import PatchableSpanProjector
    except Exception:
        VerilogAstExtractor = None
        PatchableSpanProjector = None
        VerilogAstUnavailable = RuntimeError

try:
    from utils import normalize_repo_path
except Exception:
    from .utils import normalize_repo_path

# === 新增：扩展名 → 语言 的映射 ==============
EXT_LANG_MAP = {
    '.v': 'verilog',
    '.sv': 'verilog',
    '.vh': 'verilog',
    '.svh': 'verilog',
}

def language_by_extension(file_path: str) -> str | None:
    """根据文件扩展名推断语言（不支持则返回 None）"""
    for ext, lang in EXT_LANG_MAP.items():
        if file_path.endswith(ext):
            return lang
    return None
# =========================================

class LanguageConfig(ABC):
    def __init__(self, language_name: str):
        self.language = language_name
        self.config = self._load_config()

    @abstractmethod
    def get_comment_prefix(self) -> str:
        pass

    @abstractmethod
    def get_search_patterns(self, entity_name: str) -> dict[str, str]:
        """
        Returns a dictionary of regex patterns for searching entities by name.
        Keys might be 'class', 'method', 'variable', 'import', 'string', etc.
        Values are regex pattern strings.
        """
        pass

    @abstractmethod
    def resolve_qualified_name_to_file_paths(self, base_path: str, qualified_name_parts: list[str]) -> list[tuple[str, str]]:
        """
        Resolves a qualified name (e.g., ['com', 'example', 'MyClass']) to potential file paths.
        Returns a list of (type, path) tuples, where type can be 'file', 'package', etc.
        """
        pass

    def _load_config(self):
        """Return minimal config dict for file extension handling etc."""
        default_configs = {
            'verilog': {
                'file_extensions': ['.v', '.sv', '.vh', '.svh'],
                'test_file_pattern': 'tb_'
            }
        }
        return default_configs.get(self.language, {'file_extensions': [], 'test_file_pattern': ''})

class VerilogLanguageConfig(LanguageConfig):
    def __init__(self):
        super().__init__('verilog')

    def get_comment_prefix(self) -> str:
        return "//"

    def get_search_patterns(self, entity_name: str) -> dict[str, str]:
        escaped_name = re.escape(entity_name)
        return {
            'module': rf'\b(?:module|interface|package|program)\s+{escaped_name}\b',
            'function': rf'\bfunction\b[\s\S]*?\b{escaped_name}\b',
            'task': rf'\btask\b[\s\S]*?\b{escaped_name}\b',
            'procedural_block': rf'\b(?:always|always_ff|always_comb|always_latch|initial)\b[\s\S]*?\b{escaped_name}\b',
            'continuous_assign': rf'\bassign\s+{escaped_name}\b',
            'instance': rf'\b{escaped_name}\s*(?:#\s*\([^;]*?\)\s*)?[A-Za-z_][\w$]*\s*\(',
            'macro': rf'`(?:define|ifdef|ifndef|elsif)\s+{escaped_name}\b',
            'signal_declaration': rf'\b(?:wire|reg|logic|input|output|inout|parameter|localparam)\b[^;]*\b{escaped_name}\b',
            'include': rf'`include\s+"[^"]*{escaped_name}[^"]*"',
            'comment': rf'(?://.*{escaped_name}|/\*[\s\S]*?{escaped_name}[\s\S]*?\*/)',
        }

    def resolve_qualified_name_to_file_paths(self, base_path: str, qualified_name_parts: list[str]) -> list[tuple[str, str]]:
        paths = []
        if not qualified_name_parts:
            return paths

        search_roots = ['', 'src', 'rtl', os.path.join('src', 'rtl'), 'include', os.path.join('src', 'include'), 'tb', 'test', 'tests']
        possible_file_ref = '/'.join(qualified_name_parts)
        has_file_extension = len(qualified_name_parts) > 1 and qualified_name_parts[-1] in {'v', 'sv', 'vh', 'svh'}
        if has_file_extension:
            possible_file_ref = '/'.join(qualified_name_parts[:-1]) + '.' + qualified_name_parts[-1]
            for root in search_roots:
                paths.append(('file', os.path.join(base_path, root, possible_file_ref)))

        stem = os.path.splitext(os.path.basename(possible_file_ref))[0] if has_file_extension else qualified_name_parts[-1]
        if '/' in stem or '\\' in stem:
            paths.append(('file', os.path.join(base_path, stem)))

        if not has_file_extension:
            for ext in self.config['file_extensions']:
                for root in search_roots:
                    paths.append(('file', os.path.join(base_path, root, possible_file_ref + ext)))
                    paths.append(('file', os.path.join(base_path, root, stem + ext)))

        paths.append(('package', os.path.join(base_path, possible_file_ref)))
        return paths

class LanguageConfigFactory:
    """Factory to create concrete LanguageConfig instances based on language name."""
    @staticmethod
    def get_config(language: str) -> LanguageConfig:
        if language == 'verilog':
            return VerilogLanguageConfig()
        raise ValueError(f"Unsupported language for LanguageConfigFactory: {language}")

class BaseParser(ABC):
    def __init__(self, language_name: str):
        self.language_config = LanguageConfigFactory.get_config(language_name)
        self._file_ast_cache = {} # Cache for parsed file ASTs

    @abstractmethod
    def get_compilation_unit(self, file_path: str):
        """Parses the file and returns the root AST node (e.g., CompilationUnit for Java).
           Results should be cached to avoid re-parsing.
        """
        pass

    @abstractmethod
    def parse_file(self, file_path):
        pass

    @abstractmethod
    def extract_classes(self, file_path):
        pass

    @abstractmethod
    def extract_methods(self, file_path):
        pass

    @abstractmethod
    def get_imports(self, file_path):
        pass

    @abstractmethod
    def get_global_methods(self, file_path, repo_name):
        pass

    @abstractmethod
    def get_global_variables(self, file_path, repo_name):
        pass

    @abstractmethod
    def analyze_method_calls_in_method(self, local_method_info, all_methods, kg, imports, repo_name):
        pass

    @abstractmethod
    def analyze_snippet_for_references(self, code_snippet_string):
        pass

class VerilogParser(BaseParser):
    REGION_END = {
        'module': 'endmodule',
        'interface': 'endinterface',
        'package': 'endpackage',
        'program': 'endprogram',
    }

    KEYWORDS = {
        'module', 'endmodule', 'interface', 'endinterface', 'package', 'endpackage',
        'program', 'endprogram', 'input', 'output', 'inout', 'wire', 'reg', 'logic',
        'assign', 'always', 'always_ff', 'always_comb', 'always_latch', 'initial',
        'begin', 'end', 'if', 'else', 'case', 'endcase', 'for', 'while', 'generate',
        'endgenerate', 'function', 'endfunction', 'task', 'endtask', 'parameter',
        'localparam', 'typedef', 'struct', 'enum', 'import', 'automatic',
    }

    def __init__(self):
        super().__init__('verilog')
        self.repo_root = ""
        self._verilog_ast_extractor = None
        if VerilogAstExtractor is not None:
            try:
                self._verilog_ast_extractor = VerilogAstExtractor(self.repo_root)
            except VerilogAstUnavailable as exc:
                twarn(f"Verilog AST unavailable; using regex: {exc}")
            except Exception as exc:
                twarn(f"Verilog AST init failed; using regex: {exc}")

    def set_repo_root(self, repo_root: str):
        self.repo_root = repo_root or ""
        if self._verilog_ast_extractor is not None:
            self._verilog_ast_extractor.repo_root = self.repo_root

    def _mark_parse_metadata(self, value, parse_source='regex', parse_confidence=0.55):
        if isinstance(value, list):
            return [
                self._mark_parse_metadata(item, parse_source, parse_confidence)
                for item in value
            ]
        if not isinstance(value, dict):
            return value
        value.setdefault('parse_source', parse_source)
        value.setdefault('parse_confidence', parse_confidence)
        for key in ('methods', 'rtl_entities', 'entities', 'patchable_spans'):
            if isinstance(value.get(key), list):
                value[key] = self._mark_parse_metadata(value[key], parse_source, parse_confidence)
        for key in ('method_signal_edges', 'method_span_edges', 'signal_edges', 'span_edges', 'instance_edges'):
            if isinstance(value.get(key), list):
                value[key] = self._mark_parse_metadata(value[key], parse_source, 0.5)
        return value

    def _clean_path(self, file_path: str) -> str:
        return normalize_repo_path(file_path, self.repo_root)

    def _read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _strip_comments_keep_lines(self, content: str) -> str:
        result = []
        i = 0
        in_block = False
        while i < len(content):
            ch = content[i]
            nxt = content[i:i + 2]
            if in_block:
                if nxt == '*/':
                    in_block = False
                    result.extend('  ')
                    i += 2
                else:
                    result.append('\n' if ch == '\n' else ' ')
                    i += 1
            elif nxt == '/*':
                in_block = True
                result.extend('  ')
                i += 2
            elif nxt == '//':
                while i < len(content) and content[i] != '\n':
                    result.append(' ')
                    i += 1
            else:
                result.append(ch)
                i += 1
        return ''.join(result)

    def parse_file(self, file_path):
        content = self._read_file(file_path)
        return self._strip_comments_keep_lines(content), content

    def get_compilation_unit(self, file_path: str):
        if file_path not in self._file_ast_cache:
            self._file_ast_cache[file_path] = self.parse_file(file_path)
        return self._file_ast_cache[file_path]

    def _find_regions(self, file_path):
        cleaned, content = self.parse_file(file_path)
        clean_file_path = self._clean_path(file_path)
        lines = cleaned.splitlines()
        raw_lines = content.splitlines()
        regions = []
        start_re = re.compile(r'^\s*(module|interface|package|program)\s+([A-Za-z_][\w$]*)\b')
        idx = 0
        while idx < len(lines):
            match = start_re.search(lines[idx])
            if not match:
                idx += 1
                continue
            kind, name = match.group(1), match.group(2)
            end_kw = self.REGION_END[kind]
            end_idx = idx
            for scan_idx in range(idx + 1, len(lines)):
                if re.search(rf'\b{end_kw}\b', lines[scan_idx]):
                    end_idx = scan_idx
                    break
            source = '\n'.join(raw_lines[idx:end_idx + 1])
            regions.append({
                'kind': kind,
                'name': name,
                'file_path': clean_file_path,
                'start_line': idx + 1,
                'end_line': end_idx + 1,
                'source_code': source,
                'doc_string': self._leading_comment(raw_lines, idx),
                'cleaned_source': '\n'.join(lines[idx:end_idx + 1]),
            })
            idx = end_idx + 1
        return regions

    def _leading_comment(self, lines, start_idx):
        comments = []
        idx = start_idx - 1
        while idx >= 0:
            stripped = lines[idx].strip()
            if stripped.startswith('//'):
                comments.insert(0, stripped[2:].strip())
                idx -= 1
                continue
            if stripped == '':
                idx -= 1
                continue
            break
        return ' '.join(comments)

    def _find_statement_end(self, lines, start_idx):
        balance = 0
        saw_begin = False
        for idx in range(start_idx, len(lines)):
            line = lines[idx]
            begin_count = len(re.findall(r'\bbegin\b', line))
            end_count = len(re.findall(r'\bend\b', line)) - len(re.findall(r'\b(?:endmodule|endcase|endgenerate|endfunction|endtask|endinterface|endpackage|endprogram)\b', line))
            if begin_count:
                saw_begin = True
            balance += begin_count
            balance -= max(end_count, 0)
            if saw_begin and balance <= 0 and idx > start_idx:
                return idx
            if not saw_begin and ';' in line:
                return idx
        return start_idx

    def _find_end_keyword(self, lines, start_idx, end_keyword):
        for idx in range(start_idx + 1, len(lines)):
            if re.search(rf'\b{end_keyword}\b', lines[idx]):
                return idx
        return start_idx

    def _identifier_from_declaration(self, line):
        before_semicolon = line.split(';', 1)[0]
        identifiers = re.findall(r'\b[A-Za-z_][\w$]*\b', before_semicolon)
        identifiers = [item for item in identifiers if item not in self.KEYWORDS]
        return identifiers[-1] if identifiers else None

    def _base_identifier(self, value):
        match = re.search(r'\b[A-Za-z_][\w$]*\b', value or '')
        return match.group(0) if match else ''

    def _extract_identifiers(self, text):
        identifiers = set(re.findall(r'\b[A-Za-z_][\w$]*\b', text or ''))
        return {
            item for item in identifiers
            if item not in self.KEYWORDS
            and item not in {'posedge', 'negedge', 'signed', 'unsigned'}
            and not item[0].isdigit()
        }

    def _extract_macro_references(self, text):
        macro_keywords = {'include', 'define', 'ifdef', 'ifndef', 'elsif', 'else', 'endif', 'undef'}
        return {
            item for item in re.findall(r'`([A-Za-z_][\w$]*)', text or '')
            if item not in macro_keywords
        }

    def _strip_decl_modifiers(self, text):
        text = re.sub(r'\b(?:wire|reg|logic|signed|unsigned|integer|bit)\b', ' ', text)
        text = re.sub(r'\[[^\]]+\]', ' ', text)
        return text

    def _split_decl_names(self, text):
        names = []
        for part in text.split(','):
            part = part.strip().rstrip(');')
            if not part:
                continue
            part = part.split('=', 1)[0].strip()
            part = re.sub(r'\[[^\]]+\]', ' ', part)
            identifiers = [
                item for item in re.findall(r'\b[A-Za-z_][\w$]*\b', part)
                if item not in self.KEYWORDS
                and item not in {'wire', 'reg', 'logic', 'signed', 'unsigned', 'integer', 'bit'}
            ]
            if identifiers:
                names.append(identifiers[-1])
        return names

    def _parse_declaration_line(self, line):
        stripped = line.strip()
        if not stripped:
            return []

        stripped = stripped.rstrip(',;')
        first = stripped.split()[0] if stripped.split() else ''
        if first in {'input', 'output', 'inout'}:
            label = 'Port'
            rtl_kind = 'port'
            direction = first
            rest = stripped[len(first):].strip()
        elif first in {'wire', 'reg', 'logic'}:
            label = 'Signal'
            rtl_kind = first
            direction = ''
            rest = stripped[len(first):].strip()
        elif first in {'parameter', 'localparam'}:
            label = 'Parameter'
            rtl_kind = first
            direction = ''
            rest = stripped[len(first):].strip()
        else:
            return []

        width_match = re.search(r'\[[^\]]+\]', rest)
        width = width_match.group(0) if width_match else ''
        rest = self._strip_decl_modifiers(rest)
        names = self._split_decl_names(rest)
        return [
            {
                'label': label,
                'rtl_kind': rtl_kind,
                'direction': direction,
                'width': width,
                'signal_name': name,
                'declaration': line.strip(),
            }
            for name in names
        ]

    def _extract_region_entities(self, region):
        source_lines = region['source_code'].splitlines()
        cleaned_lines = region['cleaned_source'].splitlines()
        module_name = region['name']
        file_path = region['file_path']
        start_offset = region['start_line'] - 1
        entities = []
        signal_index = {}
        seen = set()

        for idx, line in enumerate(cleaned_lines):
            for decl in self._parse_declaration_line(line):
                signal_name = decl['signal_name']
                entity_name = f"{module_name}.{signal_name}"
                key = (decl['label'], entity_name)
                if key in seen:
                    continue
                seen.add(key)
                entity = {
                    'label': decl['label'],
                    'name': entity_name,
                    'file_path': file_path,
                    'module_name': module_name,
                    'rtl_kind': decl['rtl_kind'],
                    'verilog_kind': decl['rtl_kind'],
                    'signal_name': signal_name,
                    'direction': decl['direction'],
                    'width': decl['width'],
                    'declaration': decl['declaration'],
                    'start_line': start_offset + idx + 1,
                    'end_line': start_offset + idx + 1,
                    'source_code': source_lines[idx] if idx < len(source_lines) else decl['declaration'],
                }
                entity['semantic_summary'] = (
                    f"{decl['label']} {entity_name}; module={module_name}; "
                    f"kind={decl['rtl_kind']}; direction={decl['direction'] or 'internal'}; "
                    f"width={decl['width'] or 'scalar'}; declaration={decl['declaration']}"
                )
                entities.append(entity)
                signal_index[signal_name] = entity

        return entities, signal_index

    def _extract_macros(self, file_path):
        content = self._read_file(file_path)
        clean_file_path = self._clean_path(file_path)
        entities = []
        for idx, line in enumerate(content.splitlines(), 1):
            define_match = re.match(r'\s*`define\s+([A-Za-z_][\w$]*)\b(.*)', line)
            if not define_match:
                continue
            macro = define_match.group(1)
            value = define_match.group(2).strip()
            name = f"`{macro}"
            entities.append({
                'label': 'Macro',
                'name': name,
                'file_path': clean_file_path,
                'module_name': '',
                'rtl_kind': 'macro',
                'verilog_kind': 'macro',
                'signal_name': macro,
                'direction': '',
                'width': '',
                'declaration': line.strip(),
                'start_line': idx,
                'end_line': idx,
                'source_code': line,
                'semantic_summary': f"Macro {name}`; value={value}; declaration={line.strip()}",
            })
        return entities

    def _analyze_signal_usage(self, source, signal_index, verilog_kind, assign_target=None):
        signal_names = set(signal_index.keys())
        if verilog_kind == 'module_body':
            return {'reads': set(), 'writes': set(), 'drives': set(), 'connects': set(), 'feeds': set()}
        if not signal_names:
            return {'reads': set(), 'writes': set(), 'drives': set(), 'connects': set(), 'feeds': set()}

        cleaned = self._strip_comments_keep_lines(source)
        identifiers = self._extract_identifiers(cleaned)
        writes = set()
        drives = set()
        connects = set()

        if verilog_kind == 'assign':
            target_name = self._base_identifier(assign_target or cleaned.split('=', 1)[0])
            if target_name in signal_names:
                drives.add(target_name)
                writes.add(target_name)
        elif verilog_kind == 'instance':
            for match in re.finditer(r'\.[A-Za-z_][\w$]*\s*\(([^)]*)\)', cleaned):
                connects.update(self._extract_identifiers(match.group(1)) & signal_names)
        elif verilog_kind not in {'module_body'}:
            for match in re.finditer(r'\b([A-Za-z_][\w$]*)\b\s*(?:\[[^\]]+\])?\s*(?:<=|=)', cleaned):
                candidate = match.group(1)
                if candidate in signal_names:
                    writes.add(candidate)

        read_candidates = (identifiers & signal_names) - writes - drives
        if verilog_kind == 'instance':
            read_candidates -= connects

        feeds = set()
        for source_signal in read_candidates | connects:
            for target_signal in writes | drives:
                if source_signal != target_signal:
                    feeds.add((source_signal, target_signal))

        return {
            'reads': read_candidates,
            'writes': writes,
            'drives': drives,
            'connects': connects,
            'feeds': feeds,
        }

    def _method_semantic_summary(self, module_name, verilog_kind, signature, source, signal_index, assign_target=None, module_type=None, instance_name=None):
        usage = self._analyze_signal_usage(source, signal_index, verilog_kind, assign_target)
        pieces = [
            f"module={module_name}",
            f"kind={verilog_kind}",
            f"signature={signature}",
        ]
        if assign_target:
            pieces.append(f"drives={self._base_identifier(assign_target)}")
        if module_type:
            pieces.append(f"instantiates={module_type}")
        if instance_name:
            pieces.append(f"instance={instance_name}")
        for key in ('reads', 'writes', 'drives', 'connects'):
            values = sorted(usage[key])
            if values:
                pieces.append(f"{key}={', '.join(values)}")
        macro_refs = sorted(self._extract_macro_references(source))
        if macro_refs:
            pieces.append(f"macros={', '.join('`' + item for item in macro_refs)}")
        compact_source = ' '.join(line.strip() for line in source.splitlines()[:6] if line.strip())
        if compact_source:
            pieces.append(f"source_excerpt={compact_source[:500]}")
        return '; '.join(pieces)

    def _class_semantic_summary(self, region, entities):
        ports = [e for e in entities if e['label'] == 'Port']
        signals = [e for e in entities if e['label'] == 'Signal']
        params = [e for e in entities if e['label'] == 'Parameter']
        header = ' '.join(line.strip() for line in region['source_code'].splitlines()[:12] if line.strip())
        return (
            f"{region['kind']} {region['name']}; "
            f"ports={', '.join(e['signal_name'] for e in ports[:20])}; "
            f"signals={', '.join(e['signal_name'] for e in signals[:20])}; "
            f"parameters={', '.join(e['signal_name'] for e in params[:20])}; "
            f"header={header[:700]}"
        )

    def _extract_module_members(self, region, signal_index=None):
        signal_index = signal_index or {}
        source_lines = region['source_code'].splitlines()
        cleaned_lines = region['cleaned_source'].splitlines()
        module_name = region['name']
        file_path = region['file_path']
        start_offset = region['start_line'] - 1
        members = [{
            'name': f"{module_name}.module",
            'signature': f"{region['kind']} {module_name}",
            'file_path': file_path,
            'start_line': region['start_line'],
            'end_line': region['end_line'],
            'source_code': region['source_code'],
            'doc_string': region.get('doc_string', ''),
            'verilog_kind': 'module_body',
            'semantic_summary': self._method_semantic_summary(
                module_name,
                'module_body',
                f"{region['kind']} {module_name}",
                region['source_code'],
                signal_index,
            ),
        }]

        for idx, line in enumerate(cleaned_lines):
            stripped = line.strip()
            if not stripped:
                continue

            if re.match(r'^(function|task)\b', stripped):
                kind = stripped.split()[0]
                name = self._identifier_from_declaration(stripped)
                if not name:
                    continue
                end_idx = self._find_end_keyword(cleaned_lines, idx, f"end{kind}")
                members.append({
                    'name': f"{module_name}.{name}",
                    'signature': f"{kind} {module_name}.{name}",
                    'file_path': file_path,
                    'start_line': start_offset + idx + 1,
                    'end_line': start_offset + end_idx + 1,
                    'source_code': '\n'.join(source_lines[idx:end_idx + 1]),
                    'doc_string': self._leading_comment(source_lines, idx),
                    'verilog_kind': kind,
                    'semantic_summary': self._method_semantic_summary(
                        module_name,
                        kind,
                        f"{kind} {module_name}.{name}",
                        '\n'.join(source_lines[idx:end_idx + 1]),
                        signal_index,
                    ),
                })
                continue

            if re.match(r'^(always|always_ff|always_comb|always_latch|initial)\b', stripped):
                kind = stripped.split()[0]
                end_idx = self._find_statement_end(cleaned_lines, idx)
                members.append({
                    'name': f"{module_name}.{kind}@{start_offset + idx + 1}",
                    'signature': f"{module_name}.{kind}@{start_offset + idx + 1}",
                    'file_path': file_path,
                    'start_line': start_offset + idx + 1,
                    'end_line': start_offset + end_idx + 1,
                    'source_code': '\n'.join(source_lines[idx:end_idx + 1]),
                    'doc_string': self._leading_comment(source_lines, idx),
                    'verilog_kind': kind,
                    'semantic_summary': self._method_semantic_summary(
                        module_name,
                        kind,
                        f"{module_name}.{kind}@{start_offset + idx + 1}",
                        '\n'.join(source_lines[idx:end_idx + 1]),
                        signal_index,
                    ),
                })
                continue

            assign_match = re.match(r'^assign\s+([A-Za-z_][\w$]*(?:\[[^\]]+\])?)\s*=', stripped)
            if assign_match:
                target = assign_match.group(1)
                target_name = self._base_identifier(target) or target
                end_idx = self._find_statement_end(cleaned_lines, idx)
                members.append({
                    'name': f"{module_name}.assign.{target_name}@{start_offset + idx + 1}",
                    'signature': f"assign {target}",
                    'file_path': file_path,
                    'start_line': start_offset + idx + 1,
                    'end_line': start_offset + end_idx + 1,
                    'source_code': '\n'.join(source_lines[idx:end_idx + 1]),
                    'doc_string': self._leading_comment(source_lines, idx),
                    'verilog_kind': 'assign',
                    'target_signal': target_name,
                    'semantic_summary': self._method_semantic_summary(
                        module_name,
                        'assign',
                        f"assign {target}",
                        '\n'.join(source_lines[idx:end_idx + 1]),
                        signal_index,
                        assign_target=target,
                    ),
                })
                continue

            inst_match = re.match(r'^([A-Za-z_][\w$]*)\s+(?:#\s*\([^;]*?\)\s*)?([A-Za-z_][\w$]*)\s*\(', stripped)
            if inst_match:
                module_type = inst_match.group(1)
                if module_type in self.KEYWORDS:
                    continue
                instance_name = self._extract_instance_name(stripped, module_type)
                if not instance_name:
                    continue
                end_idx = self._find_statement_end(cleaned_lines, idx)
                members.append({
                    'name': f"{module_name}.inst.{module_type}.{instance_name}",
                    'signature': f"{module_name} instantiates {module_type} as {instance_name}",
                    'file_path': file_path,
                    'start_line': start_offset + idx + 1,
                    'end_line': start_offset + end_idx + 1,
                    'source_code': '\n'.join(source_lines[idx:end_idx + 1]),
                    'doc_string': '',
                    'verilog_kind': 'instance',
                    'instance_module': module_type,
                    'instance_name': instance_name,
                    'semantic_summary': self._method_semantic_summary(
                        module_name,
                        'instance',
                        f"{module_name} instantiates {module_type} as {instance_name}",
                        '\n'.join(source_lines[idx:end_idx + 1]),
                        signal_index,
                        module_type=module_type,
                        instance_name=instance_name,
                    ),
                })

        return members

    def _extract_instance_name(self, line, module_type):
        without_type = line[len(module_type):].strip()
        if without_type.startswith('#'):
            paren = 0
            for idx, ch in enumerate(without_type):
                if ch == '(':
                    paren += 1
                elif ch == ')':
                    paren -= 1
                    if paren == 0:
                        without_type = without_type[idx + 1:].strip()
                        break
        match = re.match(r'([A-Za-z_][\w$]*)\s*\(', without_type)
        return match.group(1) if match else None

    def _extract_classes_regex(self, file_path):
        classes = []
        for region in self._find_regions(file_path):
            entities, signal_index = self._extract_region_entities(region)
            classes.append({
                'name': region['name'],
                'file_path': region['file_path'],
                'start_line': region['start_line'],
                'end_line': region['end_line'],
                'source_code': region['source_code'],
                'doc_string': region.get('doc_string', ''),
                'verilog_kind': region['kind'],
                'semantic_summary': self._class_semantic_summary(region, entities),
                'rtl_entities': entities,
                'methods': self._extract_module_members(region, signal_index),
            })
        return classes

    def extract_classes(self, file_path):
        if self._verilog_ast_extractor is not None:
            try:
                return self._verilog_ast_extractor.extract_classes(file_path)
            except Exception as exc:
                twarn(f"Verilog AST extraction failed for {file_path}; using regex: {exc}")
        return self._mark_parse_metadata(self._extract_classes_regex(file_path), 'regex', 0.55)

    def extract_methods(self, file_path):
        methods = []
        for class_info in self.extract_classes(file_path):
            methods.extend(class_info.get('methods', []))
        return methods

    def get_imports(self, file_path):
        content = self._read_file(file_path)
        imports = {}
        for match in re.finditer(r'`include\s+"([^"]+)"', content):
            include_path = match.group(1)
            imports[os.path.basename(include_path)] = include_path
        for match in re.finditer(r'\bimport\s+([A-Za-z_][\w$]*(?:::[A-Za-z_*][\w$*]*)?)\s*;', content):
            imported = match.group(1)
            imports[imported.split('::')[0]] = imported
        return imports

    def get_global_methods(self, file_path, repo_name):
        return []

    def get_global_variables(self, file_path, repo_name):
        content = self._read_file(file_path)
        clean_file_path = self._clean_path(file_path)
        variables = []
        for idx, line in enumerate(content.splitlines(), 1):
            define_match = re.match(r'\s*`define\s+([A-Za-z_][\w$]*)\b(.*)', line)
            if define_match:
                macro = define_match.group(1)
                variables.append({
                    'name': f"`{macro}",
                    'signature': line.strip(),
                    'file_path': clean_file_path,
                    'start_line': idx,
                    'end_line': idx,
                    'source_code': line,
                    'doc_string': '',
                    'verilog_kind': 'macro',
                    'parse_source': 'regex',
                    'parse_confidence': 0.55,
                    'semantic_summary': f"Macro `{macro}`; declaration={line.strip()}",
                })
        return variables

    def _extract_rtl_graph_regex(self, file_path):
        entities = []
        method_signal_edges = []
        signal_edges = []
        instance_edges = []
        legacy_classes = []
        entity_keys = set()

        for macro in self._extract_macros(file_path):
            key = (macro['label'], macro['name'], macro['file_path'], macro.get('module_name', ''))
            if key not in entity_keys:
                entity_keys.add(key)
                entities.append(macro)

        for region in self._find_regions(file_path):
            region_entities, signal_index = self._extract_region_entities(region)
            for entity in region_entities:
                key = (entity['label'], entity['name'], entity['file_path'], entity.get('module_name', ''))
                if key not in entity_keys:
                    entity_keys.add(key)
                    entities.append(entity)
            methods = self._extract_module_members(region, signal_index)
            legacy_classes.append({
                'name': region['name'],
                'file_path': region['file_path'],
                'methods': methods,
            })

            for method in methods:
                verilog_kind = method.get('verilog_kind', '')
                if verilog_kind == 'module_body':
                    continue
                usage = self._analyze_signal_usage(
                    method.get('source_code', ''),
                    signal_index,
                    verilog_kind,
                    assign_target=method.get('target_signal'),
                )

                for macro_name in sorted(self._extract_macro_references(method.get('source_code', ''))):
                    macro_entity = {
                        'label': 'Macro',
                        'name': f"`{macro_name}",
                        'file_path': region['file_path'],
                        'module_name': region['name'],
                        'rtl_kind': 'macro_ref',
                        'verilog_kind': 'macro_ref',
                        'signal_name': macro_name,
                        'direction': '',
                        'width': '',
                        'declaration': f"`{macro_name}",
                        'start_line': method.get('start_line'),
                        'end_line': method.get('start_line'),
                        'source_code': f"`{macro_name}",
                        'semantic_summary': f"Macro reference `{macro_name}`; module={region['name']}; used_by={method['name']}",
                    }
                    key = (
                        macro_entity['label'],
                        macro_entity['name'],
                        macro_entity['file_path'],
                        macro_entity.get('module_name', ''),
                    )
                    if key not in entity_keys:
                        entity_keys.add(key)
                        entities.append(macro_entity)
                    method_signal_edges.append({
                        'method_name': method['name'],
                        'method_signature': method['signature'],
                        'method_file_path': method['file_path'],
                        'target_label': 'Macro',
                        'target_name': macro_entity['name'],
                        'target_file_path': macro_entity['file_path'],
                        'target_module_name': macro_entity['module_name'],
                        'description': 'uses macro',
                        'reverse_description': 'used by method',
                        'relation_kind': 'mentions',
                    })

                for signal_name in sorted(usage['reads']):
                    entity = signal_index[signal_name]
                    method_signal_edges.append({
                        'method_name': method['name'],
                        'method_signature': method['signature'],
                        'method_file_path': method['file_path'],
                        'target_label': entity['label'],
                        'target_name': entity['name'],
                        'target_file_path': entity['file_path'],
                        'target_module_name': entity['module_name'],
                        'description': 'reads signal',
                        'reverse_description': 'read by method',
                        'relation_kind': 'reads',
                    })

                for signal_name in sorted(usage['writes']):
                    entity = signal_index[signal_name]
                    method_signal_edges.append({
                        'method_name': method['name'],
                        'method_signature': method['signature'],
                        'method_file_path': method['file_path'],
                        'target_label': entity['label'],
                        'target_name': entity['name'],
                        'target_file_path': entity['file_path'],
                        'target_module_name': entity['module_name'],
                        'description': 'writes signal',
                        'reverse_description': 'written by method',
                        'relation_kind': 'writes',
                    })

                for signal_name in sorted(usage['drives']):
                    entity = signal_index[signal_name]
                    method_signal_edges.append({
                        'method_name': method['name'],
                        'method_signature': method['signature'],
                        'method_file_path': method['file_path'],
                        'target_label': entity['label'],
                        'target_name': entity['name'],
                        'target_file_path': entity['file_path'],
                        'target_module_name': entity['module_name'],
                        'description': 'drives signal',
                        'reverse_description': 'driven by method',
                        'relation_kind': 'drives',
                    })

                for signal_name in sorted(usage['connects']):
                    entity = signal_index[signal_name]
                    method_signal_edges.append({
                        'method_name': method['name'],
                        'method_signature': method['signature'],
                        'method_file_path': method['file_path'],
                        'target_label': entity['label'],
                        'target_name': entity['name'],
                        'target_file_path': entity['file_path'],
                        'target_module_name': entity['module_name'],
                        'description': 'connects signal',
                        'reverse_description': 'connected by instance',
                        'relation_kind': 'connects',
                    })

                for source_signal, target_signal in sorted(usage['feeds']):
                    source_entity = signal_index[source_signal]
                    target_entity = signal_index[target_signal]
                    signal_edges.append({
                        'source_label': source_entity['label'],
                        'source_name': source_entity['name'],
                        'source_file_path': source_entity['file_path'],
                        'source_module_name': source_entity['module_name'],
                        'target_label': target_entity['label'],
                        'target_name': target_entity['name'],
                        'target_file_path': target_entity['file_path'],
                        'target_module_name': target_entity['module_name'],
                        'description': 'feeds signal',
                        'reverse_description': 'fed by signal',
                        'relation_kind': 'feeds',
                    })

                if verilog_kind == 'instance' and method.get('instance_module'):
                    instance_edges.append({
                        'method_name': method['name'],
                        'method_signature': method['signature'],
                        'method_file_path': method['file_path'],
                        'class_name': method['instance_module'],
                        'description': 'instantiates module',
                        'reverse_description': 'instantiated by method',
                        'relation_kind': 'instantiates',
                    })

        projection = (
            PatchableSpanProjector().project_from_legacy(legacy_classes)
            if PatchableSpanProjector is not None
            else {'patchable_spans': [], 'method_span_edges': [], 'span_edges': []}
        )
        for span in projection.get('patchable_spans') or []:
            key = (span['label'], span['name'], span['file_path'], span.get('module_name', ''))
            if key not in entity_keys:
                entity_keys.add(key)
                entities.append(span)

        return {
            'entities': entities,
            'patchable_spans': projection.get('patchable_spans') or [],
            'method_signal_edges': method_signal_edges,
            'method_span_edges': projection.get('method_span_edges') or [],
            'signal_edges': signal_edges,
            'span_edges': projection.get('span_edges') or [],
            'instance_edges': instance_edges,
        }

    def extract_rtl_graph(self, file_path):
        if self._verilog_ast_extractor is not None:
            try:
                return self._verilog_ast_extractor.extract_rtl_graph(file_path)
            except Exception as exc:
                twarn(f"Verilog RTL extraction failed for {file_path}; using regex: {exc}")
        return self._mark_parse_metadata(self._extract_rtl_graph_regex(file_path), 'regex', 0.55)

    def _extract_instantiations_from_source(self, source):
        instantiations = []
        for line in self._strip_comments_keep_lines(source).splitlines():
            stripped = line.strip()
            match = re.match(r'^([A-Za-z_][\w$]*)\s+(?:#\s*\([^;]*?\)\s*)?([A-Za-z_][\w$]*)\s*\(', stripped)
            if match and match.group(1) not in self.KEYWORDS:
                instantiations.append((match.group(1), match.group(2)))
        return instantiations

    def analyze_method_calls_in_method(self, local_method_info, all_methods, kg, imports, repo_name):
        source = local_method_info.get('source_code', '')
        if not source:
            return
        caller_name = local_method_info.get('name')
        caller_signature = local_method_info.get('signature', caller_name)
        if not caller_name or not caller_signature:
            return

        linked = set()
        for module_type, _instance_name in self._extract_instantiations_from_source(source):
            target_suffix = f"{module_type}.module"
            for method in all_methods:
                if method['name'] == target_suffix or method['name'].endswith('.' + target_suffix):
                    key = (caller_name, method['name'])
                    if key in linked or caller_name == method['name']:
                        continue
                    linked.add(key)
                    kg.link_method_calls(caller_name, caller_signature, method['name'], method.get('signature', method['name']))
                    break

        for method in all_methods:
            callee_name = method.get('name', '')
            simple_name = callee_name.split('.')[-1].split('@')[0]
            if not simple_name or simple_name in {'module'}:
                continue
            if re.search(rf'\b{re.escape(simple_name)}\s*\(', source) and caller_name != callee_name:
                key = (caller_name, callee_name)
                if key in linked:
                    continue
                linked.add(key)
                kg.link_method_calls(caller_name, caller_signature, callee_name, method.get('signature', callee_name))

    def analyze_snippet_for_references(self, code_snippet_string):
        references = []
        for match in re.finditer(r'([A-Za-z0-9_./\\-]+\.(?:v|sv|vh|svh))\b', code_snippet_string):
            references.append(('file', match.group(1).replace('\\', '/')))
        for match in re.finditer(r'\b(?:module|interface|package|program)\s+([A-Za-z_][\w$]*)\b', code_snippet_string):
            references.append(('module', match.group(1)))
        for module_type, _instance_name in self._extract_instantiations_from_source(code_snippet_string):
            references.append(('module_instantiation', module_type))
        for match in re.finditer(r'\b(?:always_ff|always_comb|always_latch|always|initial|assign|function|task)\b', code_snippet_string):
            references.append(('hdl_construct', match.group(0)))

        ordered = []
        seen = set()
        for ref in references:
            if ref not in seen:
                ordered.append(ref)
                seen.add(ref)
        return ordered

class ParserFactory:
    @staticmethod
    def create_parser(language):
        if language == 'verilog':
            return VerilogParser()
        raise ValueError(f"Unsupported language: {language}")
