import os
import re
from dataclasses import dataclass
from typing import Iterable

try:
    from patchable_projection import PatchableSpanProjector
except Exception:  # pragma: no cover
    from .patchable_projection import PatchableSpanProjector

try:
    from tree_sitter import Language, Parser
    import tree_sitter_systemverilog
except Exception as import_error:  # pragma: no cover - exercised by fallback path.
    Language = None
    Parser = None
    tree_sitter_systemverilog = None
    TREE_SITTER_IMPORT_ERROR = import_error
else:
    TREE_SITTER_IMPORT_ERROR = None

try:
    from verilog_timing import classify_verilog_entity_timing
except Exception:  # pragma: no cover - package-relative fallback.
    from .verilog_timing import classify_verilog_entity_timing


class VerilogAstUnavailable(RuntimeError):
    pass


@dataclass
class VerilogAstFile:
    file_path: str
    clean_file_path: str
    content: str
    source_bytes: bytes
    lines: list[str]
    cleaned_lines: list[str]
    tree: object


class VerilogAstExtractor:
    REGION_NODE_TYPES = {
        "module_declaration": "module",
        "interface_declaration": "interface",
        "package_declaration": "package",
        "program_declaration": "program",
    }

    KEYWORDS = {
        "module", "endmodule", "interface", "endinterface", "package", "endpackage",
        "program", "endprogram", "input", "output", "inout", "wire", "reg", "logic",
        "assign", "always", "always_ff", "always_comb", "always_latch", "initial",
        "begin", "end", "if", "else", "case", "endcase", "for", "while", "generate",
        "endgenerate", "function", "endfunction", "task", "endtask", "parameter",
        "localparam", "typedef", "struct", "enum", "import", "automatic", "signed",
        "unsigned", "bit", "integer", "default",
    }

    HDL_DECL_TYPES = {
        "net_declaration",
        "data_declaration",
        "parameter_declaration",
        "local_parameter_declaration",
        "ansi_port_declaration",
        "port_declaration",
    }

    EDIT_TARGET_TYPES = {
        "always_construct",
        "initial_construct",
        "continuous_assign",
        "function_declaration",
        "function_body_declaration",
        "task_declaration",
        "task_body_declaration",
        "module_instantiation",
    }

    TIMING_TAGS = {
        "registered_output",
        "combinational_idle",
        "edge_sensitive",
        "immediate_observable",
    }

    def __init__(self, repo_root: str = ""):
        if TREE_SITTER_IMPORT_ERROR is not None:
            raise VerilogAstUnavailable(str(TREE_SITTER_IMPORT_ERROR))
        self.parser = Parser(Language(tree_sitter_systemverilog.language()))
        self._cache: dict[str, VerilogAstFile] = {}
        self.repo_root = repo_root
        self.patchable_projector = PatchableSpanProjector()

    def parse_file(self, file_path: str) -> VerilogAstFile:
        if file_path in self._cache:
            return self._cache[file_path]
        with open(file_path, "rb") as f:
            source_bytes = f.read()
        content = source_bytes.decode("utf-8", errors="ignore")
        tree = self.parser.parse(source_bytes)
        parsed = VerilogAstFile(
            file_path=file_path,
            clean_file_path=self._clean_path(file_path),
            content=content,
            source_bytes=source_bytes,
            lines=content.splitlines(),
            cleaned_lines=self._strip_comments_keep_lines(content).splitlines(),
            tree=tree,
        )
        self._cache[file_path] = parsed
        return parsed

    def extract_classes(self, file_path: str) -> list[dict]:
        parsed = self.parse_file(file_path)
        classes = []
        for region in self._find_regions(parsed):
            entities, signal_index = self._extract_region_entities(parsed, region)
            methods = self._extract_module_members(parsed, region, signal_index)
            methods = [
                self._annotate_with_timing_hints(
                    method,
                    method.get("verilog_kind", ""),
                    method.get("source_code", ""),
                    assign_target=method.get("target_signal"),
                    module_type=method.get("instance_module"),
                )
                for method in methods
            ]
            entities = [
                self._annotate_with_timing_hints(
                    entity,
                    entity.get("verilog_kind", ""),
                    entity.get("source_code", ""),
                    assign_target=entity.get("declaration"),
                    module_type=entity.get("module_name"),
                )
                for entity in entities
            ]
            classes.append(self._annotate_timing_metadata({
                "name": region["name"],
                "file_path": region["file_path"],
                "start_line": region["start_line"],
                "end_line": region["end_line"],
                "source_code": region["source_code"],
                "doc_string": region.get("doc_string", ""),
                "verilog_kind": region["kind"],
                "semantic_summary": self._class_semantic_summary(region, entities),
                "parse_source": "ast",
                "parse_confidence": 0.95,
                "rtl_entities": entities,
                "methods": methods,
            }))
        return classes

    def extract_methods(self, file_path: str) -> list[dict]:
        methods = []
        for class_info in self.extract_classes(file_path):
            methods.extend(class_info.get("methods", []))
        return methods

    def extract_rtl_graph(self, file_path: str) -> dict:
        parsed = self.parse_file(file_path)
        entities = []
        method_signal_edges = []
        signal_edges = []
        instance_edges = []
        method_span_edges = []
        span_edges = []
        patchable_spans = []
        entity_keys = set()

        def add_entity(entity):
            entity = self._annotate_timing_metadata(entity)
            key = (
                entity["label"],
                entity["name"],
                entity["file_path"],
                entity.get("module_name", ""),
            )
            if key not in entity_keys:
                entity_keys.add(key)
                entities.append(entity)
            return entity

        macros_by_name = {}
        for macro in self._extract_macros(parsed):
            add_entity(macro)
            macros_by_name[macro["signal_name"]] = macro
        for entity in self._extract_includes(parsed):
            add_entity(entity)
        for entity in self._extract_imports(parsed):
            add_entity(entity)
        for entity in self._extract_typedefs(parsed):
            add_entity(entity)

        for scope in self._extract_conditional_scopes(parsed):
            add_entity(scope)
            macro_name = scope.get("signal_name", "")
            if macro_name:
                macro = macros_by_name.get(macro_name)
                if macro is None:
                    macro = self._macro_ref_entity(
                        parsed.clean_file_path,
                        macro_name,
                        scope.get("start_line"),
                        module_name="",
                        parse_source="ast",
                        parse_confidence=0.85,
                    )
                    add_entity(macro)
                    macros_by_name[macro_name] = macro
                signal_edges.append({
                    "source_label": "Macro",
                    "source_name": macro["name"],
                    "source_file_path": macro["file_path"],
                    "source_module_name": macro.get("module_name", ""),
                    "target_label": "ConditionalCompilationScope",
                    "target_name": scope["name"],
                    "target_file_path": scope["file_path"],
                    "target_module_name": scope.get("module_name", ""),
                    "description": "guards conditional compilation",
                    "reverse_description": "guarded by macro",
                    "relation_kind": "guards",
                    "parse_source": "ast",
                    "parse_confidence": 0.85,
                })

        for region in self._find_regions(parsed):
            region_entities, signal_index = self._extract_region_entities(parsed, region)
            region_entities = [
                self._annotate_with_timing_hints(
                    entity,
                    entity.get("verilog_kind", ""),
                    entity.get("source_code", ""),
                    assign_target=entity.get("declaration"),
                    module_type=entity.get("module_name"),
                )
                for entity in region_entities
            ]
            for entity in region_entities:
                add_entity(entity)
            projection_semantic_entities = list(region_entities)

            methods = self._extract_module_members(parsed, region, signal_index)
            control_flow_entities = []
            control_flow_method_edges = []
            control_flow_signal_edges = []
            methods = [
                self._annotate_with_timing_hints(
                    method,
                    method.get("verilog_kind", ""),
                    method.get("source_code", ""),
                    assign_target=method.get("target_signal"),
                    module_type=method.get("instance_module"),
                )
                for method in methods
            ]

            for method in methods:
                verilog_kind = method.get("verilog_kind", "")
                if verilog_kind == "module_body":
                    continue
                instance_entity = self._instance_entity_for_method(region, method)
                if instance_entity:
                    add_entity(instance_entity)
                    projection_semantic_entities.append(instance_entity)
                    method_signal_edges.append({
                        "method_name": method["name"],
                        "method_signature": method["signature"],
                        "method_file_path": method["file_path"],
                        "target_label": "Instance",
                        "target_name": instance_entity["name"],
                        "target_file_path": instance_entity["file_path"],
                        "target_module_name": instance_entity.get("module_name", ""),
                        "description": "mentions instance",
                        "reverse_description": "mentioned by instantiation statement",
                        "relation_kind": "mentions",
                        "parse_source": "ast",
                        "parse_confidence": 0.9,
                    })
                domain_entities, domain_method_edges, domain_signal_edges = self._clock_reset_domain_artifacts(
                    region,
                    method,
                    signal_index,
                )
                for entity in domain_entities:
                    add_entity(entity)
                    projection_semantic_entities.append(entity)
                method_signal_edges.extend(domain_method_edges)
                signal_edges.extend(domain_signal_edges)
                cf_entities, cf_method_edges, cf_signal_edges = self._extract_control_flow_artifacts(
                    parsed,
                    region,
                    method,
                    signal_index,
                )
                for entity in cf_entities:
                    add_entity(entity)
                    projection_semantic_entities.append(entity)
                control_flow_entities.extend(cf_entities)
                control_flow_method_edges.extend(cf_method_edges)
                control_flow_signal_edges.extend(cf_signal_edges)
                usage = self._analyze_signal_usage(
                    method.get("source_code", ""),
                    signal_index,
                    verilog_kind,
                    assign_target=method.get("target_signal"),
                )

                for macro_name in sorted(self._extract_macro_references(method.get("source_code", ""))):
                    macro_entity = self._macro_ref_entity(
                        region["file_path"],
                        macro_name,
                        method.get("start_line"),
                        module_name=region["name"],
                        parse_source="ast",
                        parse_confidence=0.75,
                    )
                    add_entity(macro_entity)
                    projection_semantic_entities.append(macro_entity)
                    method_signal_edges.append({
                        "method_name": method["name"],
                        "method_signature": method["signature"],
                        "method_file_path": method["file_path"],
                        "target_label": "Macro",
                        "target_name": macro_entity["name"],
                        "target_file_path": macro_entity["file_path"],
                        "target_module_name": macro_entity["module_name"],
                        "description": "mentions macro",
                        "reverse_description": "mentioned by method",
                        "relation_kind": "mentions",
                        "parse_source": "ast",
                        "parse_confidence": 0.75,
                    })

                for signal_name in sorted(usage["reads"]):
                    entity = signal_index[signal_name]
                    method_signal_edges.append(self._method_entity_edge(method, entity, "reads", "reads signal", "read by method", 0.82))
                for signal_name in sorted(usage["writes"]):
                    entity = signal_index[signal_name]
                    method_signal_edges.append(self._method_entity_edge(method, entity, "writes", "writes signal", "written by method", 0.85))
                for signal_name in sorted(usage["drives"]):
                    entity = signal_index[signal_name]
                    method_signal_edges.append(self._method_entity_edge(method, entity, "drives", "drives signal", "driven by method", 0.9))
                for signal_name in sorted(usage["connects"]):
                    entity = signal_index[signal_name]
                    method_signal_edges.append(self._method_entity_edge(method, entity, "connects", "connects signal", "connected by instance", 0.78))

                for source_signal, target_signal in sorted(usage["feeds"]):
                    source_entity = signal_index[source_signal]
                    target_entity = signal_index[target_signal]
                    signal_edges.append({
                        "source_label": source_entity["label"],
                        "source_name": source_entity["name"],
                        "source_file_path": source_entity["file_path"],
                        "source_module_name": source_entity.get("module_name", ""),
                        "target_label": target_entity["label"],
                        "target_name": target_entity["name"],
                        "target_file_path": target_entity["file_path"],
                        "target_module_name": target_entity.get("module_name", ""),
                        "description": "feeds signal",
                        "reverse_description": "fed by signal",
                        "relation_kind": "feeds",
                        "parse_source": "heuristic",
                        "parse_confidence": 0.62,
                    })

                if verilog_kind == "instance" and method.get("instance_module"):
                    instance_edges.append({
                        "method_name": method["name"],
                        "method_signature": method["signature"],
                        "method_file_path": method["file_path"],
                        "class_name": method["instance_module"],
                        "description": "instantiates module",
                        "reverse_description": "instantiated by method",
                        "relation_kind": "instantiates",
                        "parse_source": "ast",
                        "parse_confidence": 0.92,
                    })
                    if self._is_testbench(region):
                        instance_edges.append({
                            "method_name": method["name"],
                            "method_signature": method["signature"],
                            "method_file_path": method["file_path"],
                            "class_name": method["instance_module"],
                            "description": "exercises module",
                            "reverse_description": "exercised by testbench",
                            "relation_kind": "exercises",
                            "parse_source": "heuristic",
                            "parse_confidence": 0.7,
                            })

            projection = self.patchable_projector.project(
                parsed,
                region,
                signal_index,
                methods,
                projection_semantic_entities,
            )
            for span in projection.get("patchable_spans") or []:
                add_entity(span)
                patchable_spans.append(span)
            method_span_edges.extend(projection.get("method_span_edges") or [])
            span_edges.extend(projection.get("span_edges") or [])

            method_signal_edges.extend(control_flow_method_edges)
            signal_edges.extend(control_flow_signal_edges)
            for edge in region.get("state_edges", []):
                signal_edges.append(edge)

        return {
            "entities": entities,
            "patchable_spans": patchable_spans,
            "method_signal_edges": method_signal_edges,
            "method_span_edges": method_span_edges,
            "signal_edges": signal_edges,
            "span_edges": span_edges,
            "instance_edges": instance_edges,
        }

    def _find_regions(self, parsed: VerilogAstFile) -> list[dict]:
        regions = []
        for node in self._walk(parsed.tree.root_node):
            kind = self.REGION_NODE_TYPES.get(node.type)
            if not kind:
                continue
            name = self._region_name(parsed, node)
            if not name:
                continue
            start_line, end_line = self._line_span(node)
            regions.append({
                "kind": kind,
                "name": name,
                "file_path": parsed.clean_file_path,
                "start_line": start_line,
                "end_line": end_line,
                "source_code": self._node_source(parsed, node),
                "cleaned_source": "\n".join(parsed.cleaned_lines[start_line - 1:end_line]),
                "doc_string": self._leading_comment(parsed.lines, start_line - 1),
                "_node": node,
            })
        return regions

    def _extract_region_entities(self, parsed: VerilogAstFile, region: dict) -> tuple[list[dict], dict]:
        module_node = region["_node"]
        module_name = region["name"]
        entities = []
        signal_index = {}
        seen = set()

        def add_decl(decl: dict, node):
            signal_name = decl["signal_name"]
            entity_name = f"{module_name}.{signal_name}"
            key = (decl["label"], entity_name, signal_name)
            if key in seen:
                return None
            seen.add(key)
            start_line, end_line = self._line_span(node)
            source_code = self._node_source(parsed, node)
            entity = {
                "label": decl["label"],
                "name": entity_name,
                "file_path": region["file_path"],
                "module_name": module_name,
                "rtl_kind": decl["rtl_kind"],
                "verilog_kind": decl["rtl_kind"],
                "signal_name": signal_name,
                "direction": decl.get("direction", ""),
                "width": decl.get("width", ""),
                "declaration": decl.get("declaration", "").strip(),
                "start_line": start_line,
                "end_line": end_line,
                "start_column": int(node.start_point.column),
                "end_column": int(node.end_point.column),
                "source_code": source_code,
                "parse_source": "ast",
                "parse_confidence": 0.9,
            }
            entity["semantic_summary"] = (
                f"{decl['label']} {entity_name}; module={module_name}; "
                f"kind={decl['rtl_kind']}; direction={decl.get('direction') or 'internal'}; "
                f"width={decl.get('width') or 'scalar'}; declaration={decl.get('declaration', '').strip()}"
            )
            entities.append(entity)
            signal_index[signal_name] = entity
            return entity

        for node in self._walk(module_node):
            if node.type not in self.HDL_DECL_TYPES:
                continue
            if not self._same_node(self._nearest_region(node), module_node):
                continue
            for decl in self._parse_declaration_node(parsed, node):
                add_decl(decl, node)

        if self._is_testbench(region):
            entities.append({
                "label": "Testbench",
                "name": f"{module_name}.testbench",
                "file_path": region["file_path"],
                "module_name": module_name,
                "rtl_kind": "testbench",
                "verilog_kind": "testbench",
                "signal_name": module_name,
                "direction": "",
                "width": "",
                "declaration": f"{region['kind']} {module_name}",
                "start_line": region["start_line"],
                "end_line": region["end_line"],
                "source_code": self._module_header_excerpt(region["source_code"]),
                "semantic_summary": f"Testbench module {module_name}; file={region['file_path']}",
                "parse_source": "heuristic",
                "parse_confidence": 0.78,
            })

        entities.extend(self._extract_generate_blocks(parsed, region))
        entities.extend(self._extract_assertions(parsed, region))
        state_entities, state_edges = self._extract_states(parsed, region, signal_index)
        entities.extend(state_entities)
        region["state_edges"] = state_edges
        return entities, signal_index

    def _extract_module_members(self, parsed: VerilogAstFile, region: dict, signal_index=None) -> list[dict]:
        signal_index = signal_index or {}
        module_node = region["_node"]
        module_name = region["name"]
        members = [{
            "name": f"{module_name}.module",
            "signature": f"{region['kind']} {module_name}",
            "file_path": region["file_path"],
            "start_line": region["start_line"],
            "end_line": region["end_line"],
            "source_code": region["source_code"],
            "doc_string": region.get("doc_string", ""),
            "verilog_kind": "module_body",
            "parse_source": "ast",
            "parse_confidence": 0.95,
            "semantic_summary": self._method_semantic_summary(
                module_name,
                "module_body",
                f"{region['kind']} {module_name}",
                region["source_code"],
                signal_index,
            ),
        }]

        for node in module_node.children:
            if node.type not in self.EDIT_TARGET_TYPES:
                continue
            source = self._node_source(parsed, node)
            start_line, end_line = self._line_span(node)
            doc_string = self._leading_comment(parsed.lines, start_line - 1)

            if node.type in {"function_declaration", "function_body_declaration", "task_declaration", "task_body_declaration"}:
                kind = "function" if "function" in node.type else "task"
                name = self._function_or_task_name(source, kind)
                if not name:
                    continue
                signature = f"{kind} {module_name}.{name}"
                members.append({
                    "name": f"{module_name}.{name}",
                    "signature": signature,
                    "file_path": region["file_path"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "source_code": source,
                    "doc_string": doc_string,
                    "verilog_kind": kind,
                    "parse_source": "ast",
                    "parse_confidence": 0.9,
                    "semantic_summary": self._method_semantic_summary(module_name, kind, signature, source, signal_index),
                })
                continue

            if node.type in {"always_construct", "initial_construct"}:
                kind = self._process_kind(source)
                signature = f"{module_name}.{kind}@{start_line}"
                members.append({
                    "name": signature,
                    "signature": signature,
                    "file_path": region["file_path"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "source_code": source,
                    "doc_string": doc_string,
                    "verilog_kind": kind,
                    "parse_source": "ast",
                    "parse_confidence": 0.92,
                    "semantic_summary": self._method_semantic_summary(module_name, kind, signature, source, signal_index),
                })
                continue

            if node.type == "continuous_assign":
                target = self._assignment_target(source)
                target_name = self._base_identifier(target) or f"assign@{start_line}"
                signature = f"assign {target}"
                members.append({
                    "name": f"{module_name}.assign.{target_name}@{start_line}",
                    "signature": signature,
                    "file_path": region["file_path"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "source_code": source,
                    "doc_string": doc_string,
                    "verilog_kind": "assign",
                    "target_signal": target_name,
                    "parse_source": "ast",
                    "parse_confidence": 0.95,
                    "semantic_summary": self._method_semantic_summary(
                        module_name,
                        "assign",
                        signature,
                        source,
                        signal_index,
                        assign_target=target,
                    ),
                })
                continue

            if node.type == "module_instantiation":
                module_type, instance_name = self._instantiation_names(source)
                if not module_type or not instance_name:
                    continue
                signature = f"{module_name} instantiates {module_type} as {instance_name}"
                members.append({
                    "name": f"{module_name}.inst.{module_type}.{instance_name}",
                    "signature": signature,
                    "file_path": region["file_path"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "source_code": source,
                    "doc_string": "",
                    "verilog_kind": "instance",
                    "instance_module": module_type,
                    "instance_name": instance_name,
                    "parse_source": "ast",
                    "parse_confidence": 0.93,
                    "semantic_summary": self._method_semantic_summary(
                        module_name,
                        "instance",
                        signature,
                        source,
                        signal_index,
                        module_type=module_type,
                        instance_name=instance_name,
                    ),
                })

        return members

    def _extract_macros(self, parsed: VerilogAstFile) -> list[dict]:
        entities = []
        for node in self._walk(parsed.tree.root_node):
            if node.type != "text_macro_definition":
                continue
            source = self._node_source(parsed, node).strip()
            match = re.match(r"`define\s+([A-Za-z_][\w$]*)\b(.*)", source)
            if not match:
                continue
            macro, value = match.group(1), match.group(2).strip()
            start_line, end_line = self._line_span(node)
            name = f"`{macro}"
            entities.append({
                "label": "Macro",
                "name": name,
                "file_path": parsed.clean_file_path,
                "module_name": "",
                "rtl_kind": "macro",
                "verilog_kind": "macro",
                "signal_name": macro,
                "direction": "",
                "width": "",
                "declaration": source.splitlines()[0],
                "start_line": start_line,
                "end_line": end_line,
                "source_code": source,
                "semantic_summary": f"Macro {name}; value={value}; declaration={source.splitlines()[0]}",
                "parse_source": "ast",
                "parse_confidence": 0.95,
            })
        return entities

    def _extract_includes(self, parsed: VerilogAstFile) -> list[dict]:
        entities = []
        for match in re.finditer(r'`include\s+"([^"]+)"', parsed.content):
            include_path = match.group(1).strip()
            start_line, end_line = self._span_from_offsets(parsed.content, match.start(), match.end())
            source = match.group(0)
            name = f"{parsed.clean_file_path}.include.{include_path}@{start_line}"
            entities.append({
                "label": "Include",
                "name": name,
                "file_path": parsed.clean_file_path,
                "module_name": self._module_at_line(parsed, start_line),
                "rtl_kind": "include",
                "verilog_kind": "include",
                "signal_name": include_path,
                "direction": "",
                "width": "",
                "declaration": source,
                "start_line": start_line,
                "end_line": end_line,
                "source_code": source,
                "semantic_summary": f"Include directive {source}; include_path={include_path}",
                "parse_source": "regex",
                "parse_confidence": 0.86,
            })
        return entities

    def _extract_imports(self, parsed: VerilogAstFile) -> list[dict]:
        entities = []
        pattern = re.compile(r"\bimport\s+([^;]+);", re.MULTILINE)
        for match in pattern.finditer(parsed.content):
            import_expr = re.sub(r"\s+", " ", match.group(1)).strip()
            if not import_expr:
                continue
            start_line, end_line = self._span_from_offsets(parsed.content, match.start(), match.end())
            module_name = self._module_at_line(parsed, start_line)
            package_name = import_expr.split("::", 1)[0].strip()
            name = f"{module_name or parsed.clean_file_path}.import.{import_expr}@{start_line}"
            source = match.group(0)
            entities.append({
                "label": "Import",
                "name": name,
                "file_path": parsed.clean_file_path,
                "module_name": module_name,
                "rtl_kind": "import",
                "verilog_kind": "import",
                "signal_name": package_name,
                "direction": "",
                "width": "",
                "declaration": source,
                "start_line": start_line,
                "end_line": end_line,
                "source_code": source,
                "semantic_summary": f"Import declaration {source}; package={package_name}; scope={module_name or 'file'}",
                "parse_source": "regex",
                "parse_confidence": 0.8,
            })
        return entities

    def _extract_typedefs(self, parsed: VerilogAstFile) -> list[dict]:
        entities = []
        pattern = re.compile(r"\btypedef\b[\s\S]*?;", re.MULTILINE)
        for match in pattern.finditer(parsed.content):
            source = match.group(0)
            compact = re.sub(r"\s+", " ", source).strip()
            type_name = self._typedef_name(compact)
            if not type_name:
                continue
            start_line, end_line = self._span_from_offsets(parsed.content, match.start(), match.end())
            module_name = self._module_at_line(parsed, start_line)
            name = f"{module_name or parsed.clean_file_path}.typedef.{type_name}"
            entities.append({
                "label": "Typedef",
                "name": name,
                "file_path": parsed.clean_file_path,
                "module_name": module_name,
                "rtl_kind": "typedef",
                "verilog_kind": "typedef",
                "signal_name": type_name,
                "direction": "",
                "width": "",
                "declaration": compact,
                "start_line": start_line,
                "end_line": end_line,
                "source_code": source,
                "semantic_summary": f"Typedef {type_name}; scope={module_name or 'file'}; declaration={compact}",
                "parse_source": "regex",
                "parse_confidence": 0.78,
            })
        return self._dedupe_entities(entities)

    def _extract_conditional_scopes(self, parsed: VerilogAstFile) -> list[dict]:
        scopes = []
        for node in self._walk(parsed.tree.root_node):
            if node.type != "conditional_compilation_directive":
                continue
            source = self._node_source(parsed, node).strip()
            match = re.match(r"`(ifdef|ifndef|elsif|else|endif)\b\s*([A-Za-z_][\w$]*)?", source)
            if not match:
                continue
            kind, macro = match.group(1), match.group(2) or ""
            start_line, end_line = self._line_span(node)
            name = f"{parsed.clean_file_path}.ifdef.{kind}@{start_line}"
            scopes.append({
                "label": "ConditionalCompilationScope",
                "name": name,
                "file_path": parsed.clean_file_path,
                "module_name": "",
                "rtl_kind": kind,
                "verilog_kind": "conditional_compilation",
                "signal_name": macro,
                "direction": "",
                "width": "",
                "declaration": source,
                "start_line": start_line,
                "end_line": end_line,
                "source_code": source,
                "semantic_summary": f"Conditional compilation {kind}; macro={macro or 'none'}; declaration={source}",
                "parse_source": "ast",
                "parse_confidence": 0.9,
            })
        return scopes

    def _extract_generate_blocks(self, parsed: VerilogAstFile, region: dict) -> list[dict]:
        entities = []
        for node in self._walk(region["_node"]):
            if "generate" not in node.type or node.type in {"generate", "endgenerate"}:
                continue
            if not self._same_node(self._nearest_region(node), region["_node"]):
                continue
            start_line, end_line = self._line_span(node)
            source = self._node_source(parsed, node)
            name = f"{region['name']}.generate@{start_line}"
            entities.append({
                "label": "GenerateBlock",
                "name": name,
                "file_path": region["file_path"],
                "module_name": region["name"],
                "rtl_kind": node.type,
                "verilog_kind": "generate",
                "signal_name": "",
                "direction": "",
                "width": "",
                "declaration": self._first_nonempty_line(source),
                "start_line": start_line,
                "end_line": end_line,
                "source_code": source,
                "semantic_summary": f"Generate block {name}; kind={node.type}; excerpt={self._compact_source(source)}",
                "parse_source": "ast",
                "parse_confidence": 0.88,
            })
        return self._dedupe_entities(entities)

    def _extract_assertions(self, parsed: VerilogAstFile, region: dict) -> list[dict]:
        entities = []
        for node in self._walk(region["_node"]):
            if "assert" not in node.type:
                continue
            if not self._same_node(self._nearest_region(node), region["_node"]):
                continue
            start_line, end_line = self._line_span(node)
            source = self._node_source(parsed, node)
            name = f"{region['name']}.assert@{start_line}"
            entities.append({
                "label": "Assertion",
                "name": name,
                "file_path": region["file_path"],
                "module_name": region["name"],
                "rtl_kind": node.type,
                "verilog_kind": "assertion",
                "signal_name": "",
                "direction": "",
                "width": "",
                "declaration": self._first_nonempty_line(source),
                "start_line": start_line,
                "end_line": end_line,
                "source_code": source,
                "semantic_summary": f"Assertion {name}; kind={node.type}; excerpt={self._compact_source(source)}",
                "parse_source": "ast",
                "parse_confidence": 0.88,
            })
        return self._dedupe_entities(entities)

    def _extract_control_flow_artifacts(self, parsed: VerilogAstFile, region: dict, method: dict, signal_index: dict) -> tuple[list[dict], list[dict], list[dict]]:
        source_lines = (method.get("source_code") or "").splitlines()
        if not source_lines:
            return [], [], []

        module_name = region["name"]
        base_line = int(method.get("start_line") or region.get("start_line") or 1)
        branch_entities = []
        method_edges = []
        hdl_edges = []
        seen_branch = set()
        seen_condition = set()

        def add_branch(
            branch_kind: str,
            expr: str,
            line_no: int,
            span_end: int = None,
            parse_source: str = "heuristic",
            parse_confidence: float = 0.72,
        ):
            expr = (expr or "").strip()
            key = (branch_kind, line_no, expr)
            if key in seen_branch:
                return None
            seen_branch.add(key)
            branch_name = f"{module_name}.{method.get('verilog_kind', 'rtl')}.branch@{line_no}"
            branch_entity = {
                "label": "Branch",
                "name": branch_name,
                "file_path": region["file_path"],
                "module_name": module_name,
                "rtl_kind": branch_kind,
                "verilog_kind": "branch",
                "signal_name": branch_kind,
                "direction": "",
                "width": "",
                "declaration": expr or branch_kind,
                "start_line": line_no,
                "end_line": span_end or line_no,
                "source_code": expr or branch_kind,
                "semantic_summary": (
                    f"Branch {branch_name}; module={module_name}; "
                    f"method={method.get('signature', '')}; branch_kind={branch_kind}; "
                    f"condition={expr or 'none'}"
                ),
                "parse_source": parse_source,
                "parse_confidence": parse_confidence,
            }
            branch_entities.append(branch_entity)
            method_edges.append({
                "method_name": method["name"],
                "method_signature": method["signature"],
                "method_file_path": method["file_path"],
                "target_label": "Branch",
                "target_name": branch_name,
                "target_file_path": region["file_path"],
                "target_module_name": module_name,
                "description": f"{branch_kind} branch",
                "reverse_description": f"branched from {method.get('verilog_kind', 'rtl')}",
                "relation_kind": "branches",
                "parse_source": parse_source,
                "parse_confidence": max(parse_confidence, 0.78),
            })
            return branch_entity

        def add_condition(
            branch_entity: dict,
            expr: str,
            line_no: int,
            branch_kind: str,
            parse_source: str = "heuristic",
            parse_confidence: float = 0.7,
        ):
            expr = (expr or "").strip()
            normalized_expr = re.sub(r"\s+", " ", expr)
            if not normalized_expr:
                return None
            key = (branch_kind, line_no, normalized_expr)
            if key in seen_condition:
                return None
            seen_condition.add(key)
            cond_name = f"{module_name}.{method.get('verilog_kind', 'rtl')}.cond@{line_no}"
            signal_names = sorted(self._extract_identifiers(normalized_expr) & set(signal_index.keys()))
            condition_entity = {
                "label": "Condition",
                "name": cond_name,
                "file_path": region["file_path"],
                "module_name": module_name,
                "rtl_kind": branch_kind,
                "verilog_kind": "condition",
                "signal_name": signal_names[0] if signal_names else normalized_expr,
                "direction": "",
                "width": "",
                "declaration": normalized_expr,
                "start_line": line_no,
                "end_line": line_no,
                "source_code": normalized_expr,
                "semantic_summary": (
                    f"Condition {cond_name}; module={module_name}; "
                    f"branch_kind={branch_kind}; expr={normalized_expr}; "
                    f"signals={', '.join(signal_names) if signal_names else 'none'}"
                ),
                "parse_source": parse_source,
                "parse_confidence": parse_confidence,
            }
            branch_entities.append(condition_entity)
            hdl_edges.append({
                "source_label": "Branch",
                "source_name": branch_entity["name"],
                "source_file_path": branch_entity["file_path"],
                "source_module_name": branch_entity.get("module_name", ""),
                "target_label": "Condition",
                "target_name": cond_name,
                "target_file_path": condition_entity["file_path"],
                "target_module_name": module_name,
                "description": "guards condition",
                "reverse_description": "condition guarded by branch",
                "relation_kind": "guards",
                "parse_source": parse_source,
                "parse_confidence": max(parse_confidence, 0.75),
            })
            for signal_name in signal_names:
                signal_entity = signal_index.get(signal_name)
                if not signal_entity:
                    continue
                hdl_edges.append({
                    "source_label": "Condition",
                    "source_name": cond_name,
                    "source_file_path": condition_entity["file_path"],
                    "source_module_name": module_name,
                    "target_label": signal_entity["label"],
                    "target_name": signal_entity["name"],
                    "target_file_path": signal_entity["file_path"],
                    "target_module_name": signal_entity.get("module_name", ""),
                    "description": "reads signal",
                    "reverse_description": "read by condition",
                    "relation_kind": "reads",
                    "parse_source": parse_source,
                    "parse_confidence": max(parse_confidence - 0.05, 0.68),
                })
            return condition_entity

        method_start = int(method.get("start_line") or 0)
        method_end = int(method.get("end_line") or method_start)
        for node in self._walk(region["_node"]):
            if node.type not in {"conditional_statement", "case_statement", "case_item"}:
                continue
            line_no, span_end = self._line_span(node)
            if line_no < method_start or span_end > method_end:
                continue
            if node.type == "conditional_statement":
                branch_kind = "if"
                expr_node = next((child for child in node.children if child.type == "cond_predicate"), None)
                expr = self._node_source(parsed, expr_node) if expr_node is not None else ""
            elif node.type == "case_statement":
                branch_kind = "case"
                expr_node = next((child for child in node.children if child.type == "case_expression"), None)
                expr = self._node_source(parsed, expr_node) if expr_node is not None else ""
            else:
                branch_kind = "case_item"
                expr_node = next((child for child in node.children if child.type == "case_item_expression"), None)
                expr = self._node_source(parsed, expr_node) if expr_node is not None else "default"
            branch_entity = add_branch(
                branch_kind,
                expr,
                line_no,
                span_end,
                parse_source="ast",
                parse_confidence=0.9,
            )
            if branch_entity is not None:
                add_condition(
                    branch_entity,
                    expr,
                    line_no,
                    branch_kind,
                    parse_source="ast",
                    parse_confidence=0.88,
                )

        for local_idx, raw_line in enumerate(source_lines, start=1):
            abs_line = base_line + local_idx - 1
            stripped = raw_line.strip()
            if not stripped:
                continue

            if stripped.startswith("if ") or stripped.startswith("if(") or stripped.startswith("else if") or stripped.startswith("else if("):
                branch_kind = "if"
                expr_match = re.search(r"\b(?:else\s+)?if\s*\((.*?)\)", stripped)
                expr = expr_match.group(1) if expr_match else ""
                branch_entity = add_branch(branch_kind, expr, abs_line, abs_line)
                if branch_entity is not None:
                    add_condition(branch_entity, expr, abs_line, branch_kind)
                continue

            if stripped.startswith("case ") or stripped.startswith("case(") or re.match(r"^\b(?:unique|priority)\s+case\b", stripped):
                branch_kind = "case"
                expr_match = re.search(r"\bcase\s*\((.*?)\)", stripped)
                expr = expr_match.group(1) if expr_match else ""
                branch_entity = add_branch(branch_kind, expr, abs_line, abs_line)
                if branch_entity is not None:
                    add_condition(branch_entity, expr, abs_line, branch_kind)
                continue

            if re.match(r"^(?:default\s*:|[A-Za-z0-9_'`\[\]\s\-\+\*\/\&\|\^\~]+:)\s*$", stripped):
                branch_kind = "case_item"
                expr = stripped.rstrip(":").strip()
                branch_entity = add_branch(branch_kind, expr, abs_line, abs_line)
                if branch_entity is not None:
                    add_condition(branch_entity, expr, abs_line, branch_kind)

        return self._dedupe_entities(branch_entities), method_edges, hdl_edges

    def _extract_states(self, parsed: VerilogAstFile, region: dict, signal_index: dict) -> tuple[list[dict], list[dict]]:
        state_signals = {name for name in signal_index if "state" in name.lower()}
        if not state_signals:
            return [], []
        entities = []
        edges = []
        by_name = {}
        for node in self._walk(region["_node"]):
            if node.type != "case_statement":
                continue
            source = self._node_source(parsed, node)
            selector_match = re.search(r"\bcase\s*\(([^)]*)\)", source)
            if not selector_match:
                continue
            selector = self._base_identifier(selector_match.group(1))
            if selector not in state_signals:
                continue
            case_items = [child for child in node.children if child.type == "case_item"]
            for item in case_items:
                item_source = self._node_source(parsed, item)
                label_match = re.match(r"\s*([^:]+)\s*:", item_source, re.DOTALL)
                if not label_match:
                    continue
                state_name = self._state_name_from_expr(label_match.group(1))
                if not state_name:
                    continue
                state_entity = self._state_entity(region, selector, state_name, item)
                entities.append(state_entity)
                by_name[state_name] = state_entity
                for target_name in self._transition_targets(item_source, state_signals):
                    target_entity = by_name.get(target_name) or self._state_entity(region, selector, target_name, item)
                    by_name[target_name] = target_entity
                    entities.append(target_entity)
                    edges.append({
                        "source_label": "State",
                        "source_name": state_entity["name"],
                        "source_file_path": state_entity["file_path"],
                        "source_module_name": state_entity.get("module_name", ""),
                        "target_label": "State",
                        "target_name": target_entity["name"],
                        "target_file_path": target_entity["file_path"],
                        "target_module_name": target_entity.get("module_name", ""),
                        "description": "transitions to state",
                        "reverse_description": "transitioned from state",
                        "relation_kind": "transitions_to",
                        "parse_source": "heuristic",
                        "parse_confidence": 0.62,
                    })
        return self._dedupe_entities(entities), edges

    def _instance_entity_for_method(self, region: dict, method: dict) -> dict | None:
        if method.get("verilog_kind") != "instance":
            return None
        instance_name = method.get("instance_name") or ""
        instance_module = method.get("instance_module") or ""
        if not instance_name or not instance_module:
            return None
        name = f"{region['name']}.{instance_name}"
        return {
            "label": "Instance",
            "name": name,
            "file_path": region["file_path"],
            "module_name": region["name"],
            "rtl_kind": "module_instantiation",
            "verilog_kind": "instance",
            "signal_name": instance_name,
            "direction": "",
            "width": "",
            "declaration": method.get("signature", ""),
            "start_line": method.get("start_line"),
            "end_line": method.get("end_line"),
            "source_code": method.get("source_code", ""),
            "semantic_summary": (
                f"Instance {name}; parent_module={region['name']}; "
                f"instantiates={instance_module}; instance_name={instance_name}"
            ),
            "parse_source": "ast",
            "parse_confidence": 0.93,
        }

    def _clock_reset_domain_artifacts(self, region: dict, method: dict, signal_index: dict) -> tuple[list[dict], list[dict], list[dict]]:
        source = method.get("source_code", "")
        verilog_kind = method.get("verilog_kind", "")
        if verilog_kind not in {"always", "always_ff", "always_latch"}:
            return [], [], []

        header = source.split("begin", 1)[0]
        events = [
            (edge, signal.strip())
            for edge, signal in re.findall(r"\b(posedge|negedge)\s+([A-Za-z_][\w$]*)", header)
            if signal.strip()
        ]
        if not events:
            return [], [], []

        entities = []
        method_edges = []
        signal_edges = []
        seen = set()
        for edge, signal in events:
            is_reset = bool(re.search(r"(?:^|_)(rst|reset)(?:_|$)", signal, re.IGNORECASE))
            label = "ResetDomain" if is_reset else "ClockDomain"
            relation_kind = "reset_by" if is_reset else "clocked_by"
            description = "reset by signal" if is_reset else "clocked by signal"
            reverse_description = "resets logic" if is_reset else "clocks logic"
            key = (label, signal)
            if key in seen:
                continue
            seen.add(key)
            name = f"{region['name']}.{label}.{signal}"
            entity = {
                "label": label,
                "name": name,
                "file_path": region["file_path"],
                "module_name": region["name"],
                "rtl_kind": edge,
                "verilog_kind": label.lower(),
                "signal_name": signal,
                "direction": "",
                "width": "",
                "declaration": f"{edge} {signal}",
                "start_line": method.get("start_line"),
                "end_line": method.get("start_line"),
                "source_code": f"{edge} {signal}",
                "semantic_summary": (
                    f"{label} {name}; module={region['name']}; "
                    f"edge={edge}; signal={signal}; method={method.get('signature', '')}"
                ),
                "parse_source": "ast",
                "parse_confidence": 0.86,
            }
            entities.append(entity)
            method_edges.append({
                "method_name": method["name"],
                "method_signature": method["signature"],
                "method_file_path": method["file_path"],
                "target_label": label,
                "target_name": name,
                "target_file_path": region["file_path"],
                "target_module_name": region["name"],
                "description": description,
                "reverse_description": reverse_description,
                "relation_kind": relation_kind,
                "parse_source": "ast",
                "parse_confidence": 0.86,
            })
            signal_entity = signal_index.get(signal)
            if signal_entity:
                signal_edges.append({
                    "source_label": label,
                    "source_name": name,
                    "source_file_path": region["file_path"],
                    "source_module_name": region["name"],
                    "target_label": signal_entity["label"],
                    "target_name": signal_entity["name"],
                    "target_file_path": signal_entity["file_path"],
                    "target_module_name": signal_entity.get("module_name", ""),
                    "description": "reads signal",
                    "reverse_description": f"read by {label.lower()}",
                    "relation_kind": "reads",
                    "parse_source": "ast",
                    "parse_confidence": 0.82,
                })
        return entities, method_edges, signal_edges

    def _state_entity(self, region: dict, state_signal: str, state_name: str, node) -> dict:
        start_line, end_line = self._line_span(node)
        name = f"{region['name']}.{state_signal}.{state_name}"
        return {
            "label": "State",
            "name": name,
            "file_path": region["file_path"],
            "module_name": region["name"],
            "rtl_kind": "state",
            "verilog_kind": "state",
            "signal_name": state_name,
            "direction": "",
            "width": "",
            "declaration": state_name,
            "start_line": start_line,
            "end_line": end_line,
            "source_code": state_name,
            "semantic_summary": f"FSM state {state_name}; module={region['name']}; state_signal={state_signal}",
            "parse_source": "heuristic",
            "parse_confidence": 0.65,
        }

    def _parse_declaration_node(self, parsed: VerilogAstFile, node) -> list[dict]:
        text = self._node_source(parsed, node)
        decls = []
        if node.type in {"ansi_port_declaration", "port_declaration"}:
            decls.extend(self._parse_declaration_text(text, force_port=True))
        else:
            decls.extend(self._parse_declaration_text(text, force_port=False))
        return decls

    def _parse_declaration_text(self, text: str, force_port=False) -> list[dict]:
        first_line = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if not first_line:
            return []
        first_line = first_line.rstrip(",;")
        direction_match = re.search(r"\b(input|output|inout|ref)\b", first_line)
        decl_kind_match = re.search(r"\b(wire|reg|logic|bit|integer)\b", first_line)
        param_match = re.search(r"\b(localparam|parameter)\b", first_line)

        if force_port or direction_match:
            label = "Port"
            rtl_kind = "port"
            direction = direction_match.group(1) if direction_match else ""
            rest = first_line
        elif param_match:
            label = "Parameter"
            rtl_kind = param_match.group(1)
            direction = ""
            rest = first_line[param_match.end():]
        elif decl_kind_match:
            label = "Signal"
            rtl_kind = decl_kind_match.group(1)
            direction = ""
            rest = first_line[decl_kind_match.end():]
        else:
            return []

        width_match = re.search(r"\[[^\]]+\]", rest)
        width = width_match.group(0) if width_match else ""
        rest = re.sub(r"\b(input|output|inout|ref|wire|reg|logic|bit|integer|signed|unsigned|localparam|parameter|var)\b", " ", rest)
        rest = re.sub(r"\[[^\]]+\]", " ", rest)
        names = self._split_decl_names(rest)
        return [{
            "label": label,
            "rtl_kind": rtl_kind,
            "direction": direction,
            "width": width,
            "signal_name": name,
            "declaration": first_line,
        } for name in names]

    def _split_decl_names(self, text: str) -> list[str]:
        names = []
        for part in text.split(","):
            part = part.strip().rstrip(");")
            if not part:
                continue
            part = part.split("=", 1)[0].strip()
            identifiers = [
                item for item in re.findall(r"\b[A-Za-z_][\w$]*\b", part)
                if item not in self.KEYWORDS
            ]
            if identifiers:
                names.append(identifiers[-1])
        return names

    def _method_entity_edge(self, method, entity, relation_kind, description, reverse_description, confidence):
        return {
            "method_name": method["name"],
            "method_signature": method["signature"],
            "method_file_path": method["file_path"],
            "target_label": entity["label"],
            "target_name": entity["name"],
            "target_file_path": entity["file_path"],
            "target_module_name": entity.get("module_name", ""),
            "description": description,
            "reverse_description": reverse_description,
            "relation_kind": relation_kind,
            "parse_source": "heuristic" if relation_kind in {"reads", "writes", "connects"} else "ast",
            "parse_confidence": confidence,
        }

    def _analyze_signal_usage(self, source, signal_index, verilog_kind, assign_target=None):
        signal_names = set(signal_index.keys())
        empty = {"reads": set(), "writes": set(), "drives": set(), "connects": set(), "feeds": set()}
        if verilog_kind == "module_body" or not signal_names:
            return empty
        cleaned = self._strip_comments_keep_lines(source)
        identifiers = self._extract_identifiers(cleaned)
        writes = set()
        drives = set()
        connects = set()

        if verilog_kind == "assign":
            target_name = self._base_identifier(assign_target or cleaned.split("=", 1)[0])
            if target_name in signal_names:
                drives.add(target_name)
                writes.add(target_name)
        elif verilog_kind == "instance":
            for match in re.finditer(r"\.[A-Za-z_][\w$]*\s*\(([^)]*)\)", cleaned):
                connects.update(self._extract_identifiers(match.group(1)) & signal_names)
        else:
            for match in re.finditer(r"\b([A-Za-z_][\w$]*)\b\s*(?:\[[^\]]+\])?\s*(?:<=|=)", cleaned):
                candidate = match.group(1)
                if candidate in signal_names:
                    writes.add(candidate)

        read_candidates = (identifiers & signal_names) - writes - drives
        if verilog_kind == "instance":
            read_candidates -= connects
        feeds = {
            (source_signal, target_signal)
            for source_signal in (read_candidates | connects)
            for target_signal in (writes | drives)
            if source_signal != target_signal
        }
        return {
            "reads": read_candidates,
            "writes": writes,
            "drives": drives,
            "connects": connects,
            "feeds": feeds,
        }

    def _method_semantic_summary(self, module_name, verilog_kind, signature, source, signal_index, assign_target=None, module_type=None, instance_name=None):
        usage = self._analyze_signal_usage(source, signal_index, verilog_kind, assign_target)
        pieces = [f"module={module_name}", f"kind={verilog_kind}", f"signature={signature}"]
        if assign_target:
            pieces.append(f"drives={self._base_identifier(assign_target)}")
        if module_type:
            pieces.append(f"instantiates={module_type}")
        if instance_name:
            pieces.append(f"instance={instance_name}")
        for key in ("reads", "writes", "drives", "connects"):
            values = sorted(usage[key])
            if values:
                pieces.append(f"{key}={', '.join(values)}")
        macro_refs = sorted(self._extract_macro_references(source))
        if macro_refs:
            pieces.append(f"macros={', '.join('`' + item for item in macro_refs)}")
        compact_source = self._compact_source(source)
        if compact_source:
            pieces.append(f"source_excerpt={compact_source[:500]}")
        return "; ".join(pieces)

    def _timing_hints_for_entity(self, verilog_kind, source, assign_target=None, module_type=None):
        cleaned = self._strip_comments_keep_lines(source or "")
        source_lower = cleaned.lower()
        hints = []

        if verilog_kind in {"always_ff", "always_latch", "always"}:
            hints.append("edge_sensitive")
        if verilog_kind == "assign":
            hints.append("combinational_idle")
        if verilog_kind in {"instance", "module_body"}:
            hints.append("registered_output")
        if verilog_kind in {"assign", "always_comb"} or "assign" in source_lower:
            hints.append("immediate_observable")

        if assign_target:
            target_name = self._base_identifier(assign_target)
            if target_name and target_name.startswith(("spi_", "sck", "clk")):
                hints.append("registered_output")

        if module_type and "shifter" in module_type.lower():
            hints.append("immediate_observable")

        return [hint for hint in dict.fromkeys(hints) if hint in self.TIMING_TAGS]

    def _annotate_timing_metadata(self, entity: dict) -> dict:
        annotated = dict(entity)
        if annotated.get("timing_tags") is None:
            annotated["timing_tags"] = []
        if isinstance(annotated.get("timing_tags"), str):
            annotated["timing_tags"] = [tag.strip() for tag in annotated["timing_tags"].split(",") if tag.strip()]
        derived = classify_verilog_entity_timing(annotated)
        merged_tags = list(dict.fromkeys(list(annotated.get("timing_tags") or []) + list(derived.get("timing_tags") or [])))
        annotated.update(derived)
        annotated["timing_tags"] = [tag for tag in merged_tags if tag in self.TIMING_TAGS]
        return annotated

    def _annotate_with_timing_hints(self, entity: dict, verilog_kind: str, source: str, assign_target=None, module_type=None) -> dict:
        annotated = dict(entity)
        existing_tags = annotated.get("timing_tags") or []
        if isinstance(existing_tags, str):
            existing_tags = [tag.strip() for tag in existing_tags.split(",") if tag.strip()]
        hint_tags = self._timing_hints_for_entity(
            verilog_kind,
            source,
            assign_target=assign_target,
            module_type=module_type,
        )
        annotated["timing_tags"] = [
            tag
            for tag in dict.fromkeys(list(existing_tags) + list(hint_tags))
            if tag in self.TIMING_TAGS
        ]
        return self._annotate_timing_metadata(annotated)

    def _class_semantic_summary(self, region, entities):
        ports = [e for e in entities if e["label"] == "Port"]
        signals = [e for e in entities if e["label"] == "Signal"]
        params = [e for e in entities if e["label"] == "Parameter"]
        header = self._module_header_excerpt(region["source_code"])
        return (
            f"{region['kind']} {region['name']}; "
            f"ports={', '.join(e['signal_name'] for e in ports[:20])}; "
            f"signals={', '.join(e['signal_name'] for e in signals[:20])}; "
            f"parameters={', '.join(e['signal_name'] for e in params[:20])}; "
            f"header={header[:700]}"
        )

    def _macro_ref_entity(self, file_path, macro_name, line, module_name="", parse_source="heuristic", parse_confidence=0.65):
        name = f"`{macro_name}"
        return {
            "label": "Macro",
            "name": name,
            "file_path": file_path,
            "module_name": module_name or "",
            "rtl_kind": "macro_ref",
            "verilog_kind": "macro_ref",
            "signal_name": macro_name,
            "direction": "",
            "width": "",
            "declaration": name,
            "start_line": line,
            "end_line": line,
            "source_code": name,
            "semantic_summary": f"Macro reference {name}; module={module_name or 'file_scope'}",
            "parse_source": parse_source,
            "parse_confidence": parse_confidence,
        }

    def _region_name(self, parsed: VerilogAstFile, node) -> str:
        for child in node.children:
            if child.type.endswith("_header"):
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    return self._node_source(parsed, name_node).strip()
                identifiers = self._identifier_texts(parsed, child)
                if identifiers:
                    return identifiers[0]
        identifiers = self._identifier_texts(parsed, node)
        return identifiers[0] if identifiers else ""

    def _function_or_task_name(self, source: str, kind: str) -> str:
        header = source.split(";", 1)[0]
        identifiers = [
            item for item in re.findall(r"\b[A-Za-z_][\w$]*\b", header)
            if item not in self.KEYWORDS
        ]
        return identifiers[-1] if identifiers else ""

    def _span_from_offsets(self, content: str, start_offset: int, end_offset: int) -> tuple[int, int]:
        start_line = content.count("\n", 0, start_offset) + 1
        end_line = content.count("\n", 0, max(start_offset, end_offset - 1)) + 1
        return start_line, end_line

    def _module_at_line(self, parsed: VerilogAstFile, line_no: int) -> str:
        if not getattr(parsed, "tree", None) or not getattr(parsed.tree, "root_node", None):
            return ""
        for region in self._find_regions(parsed):
            if int(region.get("start_line") or 0) <= line_no <= int(region.get("end_line") or 0):
                return region.get("name", "")
        return ""

    def _typedef_name(self, compact_source: str) -> str:
        text = compact_source.rstrip(";").strip()
        match = re.search(r"\b([A-Za-z_][\w$]*)\s*$", text)
        if not match:
            return ""
        candidate = match.group(1)
        if candidate in self.KEYWORDS:
            return ""
        return candidate

    def _process_kind(self, source: str) -> str:
        match = re.match(r"\s*(always_ff|always_comb|always_latch|always|initial)\b", source)
        return match.group(1) if match else "always"

    def _assignment_target(self, source: str) -> str:
        match = re.search(r"\bassign\s+(.+?)\s*=", source, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _instantiation_names(self, source: str) -> tuple[str, str]:
        match = re.match(r"\s*([A-Za-z_][\w$]*)\s*(?:#\s*\([\s\S]*?\)\s*)?([A-Za-z_][\w$]*)\s*\(", source)
        if not match:
            return "", ""
        module_type, instance_name = match.group(1), match.group(2)
        if module_type in self.KEYWORDS:
            return "", ""
        return module_type, instance_name

    def _state_name_from_expr(self, expr: str) -> str:
        expr = expr.strip()
        if expr == "default" or re.match(r"\d", expr):
            return ""
        identifiers = self._extract_identifiers(expr.replace("`", " "))
        if not identifiers:
            return ""
        return sorted(identifiers)[0]

    def _transition_targets(self, source: str, state_signals: set[str]) -> list[str]:
        targets = []
        for signal in state_signals:
            pattern = rf"\b{re.escape(signal)}\b\s*(?:<=|=)\s*([`A-Za-z_][\w$`]*)"
            for match in re.finditer(pattern, source):
                target = match.group(1).lstrip("`")
                if target and target not in targets:
                    targets.append(target)
        return targets

    def _identifier_texts(self, parsed: VerilogAstFile, node) -> list[str]:
        return [
            self._node_source(parsed, item).strip().lstrip("\\")
            for item in self._walk(node)
            if item.type in {"simple_identifier", "escaped_identifier"}
        ]

    def _extract_identifiers(self, text):
        identifiers = set(re.findall(r"\b[A-Za-z_][\w$]*\b", text or ""))
        return {
            item for item in identifiers
            if item not in self.KEYWORDS
            and item not in {"posedge", "negedge", "signed", "unsigned"}
            and not item[0].isdigit()
        }

    def _extract_macro_references(self, text):
        macro_keywords = {"include", "define", "ifdef", "ifndef", "elsif", "else", "endif", "undef"}
        return {
            item for item in re.findall(r"`([A-Za-z_][\w$]*)", text or "")
            if item not in macro_keywords
        }

    def _base_identifier(self, value):
        match = re.search(r"\b[A-Za-z_][\w$]*\b", value or "")
        return match.group(0) if match else ""

    def _nearest_region(self, node):
        current = node
        while current is not None:
            if current.type in self.REGION_NODE_TYPES:
                return current
            current = current.parent
        return None

    def _same_node(self, left, right) -> bool:
        if left is None or right is None:
            return False
        return (
            left.type == right.type
            and left.start_byte == right.start_byte
            and left.end_byte == right.end_byte
        )

    def _walk(self, node) -> Iterable:
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    def _node_source(self, parsed: VerilogAstFile, node) -> str:
        return parsed.source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    def _line_span(self, node) -> tuple[int, int]:
        start_line = node.start_point.row + 1
        end_line = node.end_point.row + (1 if node.end_point.column > 0 else 0)
        return start_line, max(start_line, end_line)

    def _leading_comment(self, lines, start_idx):
        comments = []
        idx = start_idx - 1
        while idx >= 0:
            stripped = lines[idx].strip()
            if stripped.startswith("//"):
                comments.insert(0, stripped[2:].strip())
                idx -= 1
                continue
            if stripped == "":
                idx -= 1
                continue
            break
        return " ".join(comments)

    def _strip_comments_keep_lines(self, content: str) -> str:
        result = []
        i = 0
        in_block = False
        while i < len(content):
            ch = content[i]
            nxt = content[i:i + 2]
            if in_block:
                if nxt == "*/":
                    in_block = False
                    result.extend("  ")
                    i += 2
                else:
                    result.append("\n" if ch == "\n" else " ")
                    i += 1
            elif nxt == "/*":
                in_block = True
                result.extend("  ")
                i += 2
            elif nxt == "//":
                while i < len(content) and content[i] != "\n":
                    result.append(" ")
                    i += 1
            else:
                result.append(ch)
                i += 1
        return "".join(result)

    def _clean_path(self, file_path: str) -> str:
        path = os.path.normpath(file_path).replace("\\", "/")
        if self.repo_root:
            try:
                repo_root_abs = os.path.abspath(self.repo_root).replace("\\", "/").rstrip("/")
                path_abs = os.path.abspath(path).replace("\\", "/")
                if path_abs.startswith(repo_root_abs + "/"):
                    return os.path.relpath(path_abs, repo_root_abs).replace("\\", "/")
            except ValueError:
                pass
        if os.path.isabs(path):
            try:
                path = os.path.relpath(path, os.getcwd()).replace("\\", "/")
            except ValueError:
                pass
        if path.startswith("workdirs/"):
            parts = path.split("/")
            if len(parts) > 4 and parts[2] == "repos":
                return "/".join(parts[4:])
        for marker in ("playground", "verilog_repair_cases"):
            prefix = marker + "/"
            if path.startswith(prefix):
                parts = path.split("/")
                if len(parts) > 2:
                    return "/".join(parts[2:])
            marker_idx = path.find("/" + prefix)
            if marker_idx >= 0:
                parts = path[marker_idx + 1:].split("/")
                if len(parts) > 2:
                    return "/".join(parts[2:])
        return path

    def _is_testbench(self, region: dict) -> bool:
        file_path = region["file_path"].replace("\\", "/").lower()
        name = region["name"].lower()
        return "/tb/" in f"/{file_path}" or "/test/" in f"/{file_path}" or name.endswith("_tb") or "testbench" in name

    def _module_header_excerpt(self, source: str) -> str:
        lines = []
        for line in source.splitlines():
            lines.append(line.strip())
            if ");" in line or len(lines) >= 16:
                break
        return " ".join(line for line in lines if line)

    def _compact_source(self, source: str) -> str:
        return " ".join(line.strip() for line in source.splitlines()[:8] if line.strip())

    def _first_nonempty_line(self, source: str) -> str:
        for line in source.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    def _dedupe_entities(self, entities: list[dict]) -> list[dict]:
        deduped = []
        seen = set()
        for entity in entities:
            key = (entity["label"], entity["name"], entity["file_path"], entity.get("module_name", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped
