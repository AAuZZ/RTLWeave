from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


class PatchableSpanProjector:
    """Project a SystemVerilog AST into source regions suitable for repair.

    Tree-sitter supplies syntax boundaries.  The lightweight analysis in this
    class adds the local read/write/drive and guard information needed to link
    semantic RTL entities to those boundaries.  It intentionally does not try
    to elaborate parameters or resolve full cross-module types.
    """

    NODE_KINDS = {
        "always_construct": "procedural_block",
        "initial_construct": "procedural_block",
        "continuous_assign": "continuous_assign",
        "function_declaration": "function",
        "function_body_declaration": "function",
        "task_declaration": "task",
        "task_body_declaration": "task",
        "module_instantiation": "instance_binding",
        "case_statement": "case_statement",
        "case_item": "case_item",
        "conditional_statement": "conditional_branch",
        "loop_generate_construct": "generate_block",
        "conditional_generate_construct": "generate_block",
        "generate_region": "generate_block",
        "parameter_declaration": "parameter_definition",
        "local_parameter_declaration": "parameter_definition",
        "ansi_port_declaration": "port_declaration",
        "port_declaration": "port_declaration",
        "net_declaration": "signal_declaration",
        "data_declaration": "signal_declaration",
    }

    FINE_KINDS = {
        "continuous_assign",
        "case_item",
        "parameter_definition",
        "port_declaration",
        "signal_declaration",
        "instance_binding",
    }

    KEYWORDS = {
        "always", "always_comb", "always_ff", "always_latch", "assign", "automatic",
        "begin", "bit", "case", "casex", "casez", "default", "else", "end",
        "endcase", "endfunction", "endgenerate", "endmodule", "endtask", "for",
        "function", "generate", "if", "inout", "input", "integer", "localparam",
        "logic", "module", "negedge", "output", "parameter", "posedge", "priority",
        "reg", "signed", "task", "unique", "unsigned", "wire",
    }

    def project(
        self,
        parsed: Any,
        region: Dict[str, Any],
        signal_index: Dict[str, Dict[str, Any]],
        methods: Sequence[Dict[str, Any]],
        semantic_entities: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        region_node = region["_node"]
        candidates: List[Tuple[Any, str]] = []
        seen = set()
        for node in self._walk(region_node):
            kind = self.NODE_KINDS.get(node.type)
            if not kind or not self._same_node(self._nearest_region(node), region_node):
                continue
            key = self._node_key(node)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((node, kind))

        # A malformed or macro-heavy module can still be edited at module scope.
        if not candidates:
            candidates.append((region_node, "module_body"))

        spans: List[Dict[str, Any]] = []
        span_by_node: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        node_by_span_name: Dict[str, Any] = {}
        for node, kind in candidates:
            start_line, end_line = self._line_span(node)
            source = self._node_source(parsed, node)
            start_column = int(node.start_point.column)
            end_column = int(node.end_point.column)
            name = (
                f"{region['name']}.patch.{kind}"
                f"@{start_line}:{start_column}-{end_line}:{end_column}"
            )
            parent_name = self._parent_span_name(node, span_by_node)
            usage = self._analyze_local_semantics(node, source, signal_index, parsed)
            role = self._repair_role(kind, region.get("file_path", ""))
            granularity = "fine" if kind in self.FINE_KINDS else (
                "module" if kind == "module_body" else "block"
            )
            span = {
                "label": "PatchableSpan",
                "name": name,
                "file_path": region["file_path"],
                "module_name": region["name"],
                "rtl_kind": kind,
                "verilog_kind": kind,
                "span_kind": kind,
                "patch_granularity": granularity,
                "parent_span": parent_name,
                "signal_name": self._primary_signal(usage),
                "direction": "",
                "width": "",
                "declaration": self._first_nonempty_line(source),
                "start_line": start_line,
                "end_line": end_line,
                "start_column": start_column,
                "end_column": end_column,
                "source_code": source,
                "semantic_summary": self._semantic_summary(
                    region["name"], kind, source, usage, parent_name
                ),
                "parse_source": "ast+static",
                "parse_confidence": 0.95 if kind != "module_body" else 0.75,
                "repair_role": role,
                "reads_signals": sorted(usage["reads"]),
                "writes_signals": sorted(usage["writes"]),
                "drives_signals": sorted(usage["drives"]),
                "connects_signals": sorted(usage["connects"]),
                "guards_signals": sorted(usage["guards"]),
                "port_bindings": usage["port_bindings"],
                "projection_source": "tree_sitter_ast+lightweight_static_analysis",
            }
            spans.append(span)
            span_by_node[self._node_key(node)] = span
            node_by_span_name[name] = node

        # Parent spans may be discovered after children during unusual grammars.
        for span in spans:
            if span.get("parent_span"):
                continue
            node = node_by_span_name[span["name"]]
            span["parent_span"] = self._parent_span_name(node, span_by_node)

        span_edges: List[Dict[str, Any]] = []
        method_span_edges: List[Dict[str, Any]] = []

        for span in spans:
            if span.get("parent_span"):
                span_edges.append(self._span_edge(
                    "PatchableSpan", span["parent_span"], region,
                    "PatchableSpan", span["name"], region,
                    "contains", "contains nested patch span", "nested in patch span", 0.95,
                ))

            containing_method = self._smallest_containing_method(span, methods)
            if containing_method:
                method_span_edges.append({
                    "method_name": containing_method["name"],
                    "method_signature": containing_method["signature"],
                    "method_file_path": containing_method["file_path"],
                    "target_label": "PatchableSpan",
                    "target_name": span["name"],
                    "target_file_path": span["file_path"],
                    "target_module_name": span["module_name"],
                    "description": "projects method to patchable span",
                    "reverse_description": "patchable projection of method",
                    "relation_kind": "projects_to",
                    "parse_source": "ast",
                    "parse_confidence": 0.95,
                })

            for relation, key, description, reverse in (
                ("reads", "reads_signals", "reads signal", "read by patch span"),
                ("writes", "writes_signals", "writes signal", "written by patch span"),
                ("drives", "drives_signals", "drives signal", "driven by patch span"),
                ("connects", "connects_signals", "connects signal", "connected by patch span"),
            ):
                for signal_name in span.get(key) or []:
                    target = signal_index.get(signal_name)
                    if not target:
                        continue
                    span_edges.append(self._span_edge(
                        "PatchableSpan", span["name"], region,
                        target["label"], target["name"], target,
                        relation, description, reverse, 0.88 if relation != "connects" else 0.76,
                    ))

        # Project semantic evidence to the smallest source span that contains it.
        for entity in semantic_entities:
            if entity.get("label") == "PatchableSpan":
                continue
            target_span = self._smallest_containing_span(entity, spans)
            if not target_span:
                continue
            span_edges.append(self._span_edge(
                entity["label"], entity["name"], entity,
                "PatchableSpan", target_span["name"], target_span,
                "projects_to", "projects semantic entity to patch span",
                "patch span represents semantic entity", 0.9,
            ))

        return {
            "patchable_spans": spans,
            "method_span_edges": self._dedupe_edges(method_span_edges),
            "span_edges": self._dedupe_edges(span_edges),
        }

    def project_from_legacy(
        self,
        classes: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build coarse spans when Tree-sitter is unavailable."""
        spans: List[Dict[str, Any]] = []
        method_edges: List[Dict[str, Any]] = []
        for class_info in classes:
            for method in class_info.get("methods") or []:
                if method.get("verilog_kind") == "module_body":
                    continue
                kind = self._legacy_kind(method.get("verilog_kind") or "logic_block")
                start = int(method.get("start_line") or 1)
                end = int(method.get("end_line") or start)
                name = f"{class_info.get('name', 'rtl')}.patch.{kind}@{start}:{end}"
                source = method.get("source_code") or ""
                span = {
                    "label": "PatchableSpan",
                    "name": name,
                    "file_path": method.get("file_path") or class_info.get("file_path") or "",
                    "module_name": class_info.get("name") or "",
                    "rtl_kind": kind,
                    "verilog_kind": kind,
                    "span_kind": kind,
                    "patch_granularity": "fine" if kind in self.FINE_KINDS else "block",
                    "parent_span": "",
                    "signal_name": method.get("target_signal") or "",
                    "direction": "",
                    "width": "",
                    "declaration": self._first_nonempty_line(source),
                    "start_line": start,
                    "end_line": end,
                    "source_code": source,
                    "semantic_summary": method.get("semantic_summary") or source[:500],
                    "parse_source": "regex+static",
                    "parse_confidence": 0.55,
                    "repair_role": self._repair_role(kind, method.get("file_path") or ""),
                    "reads_signals": [],
                    "writes_signals": [],
                    "drives_signals": [],
                    "connects_signals": [],
                    "guards_signals": [],
                    "port_bindings": [],
                    "projection_source": "legacy_parser_projection",
                }
                spans.append(span)
                method_edges.append({
                    "method_name": method["name"],
                    "method_signature": method["signature"],
                    "method_file_path": method["file_path"],
                    "target_label": "PatchableSpan",
                    "target_name": name,
                    "target_file_path": span["file_path"],
                    "target_module_name": span["module_name"],
                    "description": "projects method to patchable span",
                    "reverse_description": "patchable projection of method",
                    "relation_kind": "projects_to",
                    "parse_source": "regex",
                    "parse_confidence": 0.55,
                })
        return {"patchable_spans": spans, "method_span_edges": method_edges, "span_edges": []}

    def _analyze_local_semantics(self, node, source, signal_index, parsed) -> Dict[str, Any]:
        signal_names = set(signal_index)
        cleaned = self._strip_comments(source)
        identifiers = self._identifiers(cleaned) & signal_names
        writes = set()
        for match in re.finditer(
            r"\b([A-Za-z_][\w$]*)\b\s*(?:\[[^\]]+\])?\s*(?:<=|=(?!=))",
            cleaned,
        ):
            if match.group(1) in signal_names:
                writes.add(match.group(1))

        kind = self.NODE_KINDS.get(node.type, "module_body")
        if kind in {"port_declaration", "signal_declaration"}:
            declared_here = {
                name for name, entity in signal_index.items()
                if str(entity.get("source_code") or "").strip() == source.strip()
            }
            identifiers -= declared_here
        drives = set(writes) if kind == "continuous_assign" else set()
        connects = set()
        bindings: List[str] = []
        if kind == "instance_binding":
            for match in re.finditer(r"\.([A-Za-z_][\w$]*)\s*\(([^()]*)\)", cleaned):
                port, expression = match.group(1), re.sub(r"\s+", " ", match.group(2)).strip()
                connected = self._identifiers(expression) & signal_names
                connects.update(connected)
                bindings.append(f"{port}={expression}"[:240])

        guards = self._guard_signals(node, parsed, signal_names)
        reads = (identifiers - writes) | guards
        if kind == "instance_binding":
            reads -= connects
        return {
            "reads": reads,
            "writes": writes,
            "drives": drives,
            "connects": connects,
            "guards": guards,
            "port_bindings": bindings[:40],
        }

    def _guard_signals(self, node, parsed, signal_names) -> set[str]:
        guards: set[str] = set()
        current = node
        while current is not None:
            if current.type == "conditional_statement":
                child = next((c for c in current.children if c.type == "cond_predicate"), None)
                if child is not None:
                    guards.update(self._identifiers(self._node_source(parsed, child)) & signal_names)
                break
            if current.type in {"case_statement", "case_item"}:
                case_node = current if current.type == "case_statement" else self._ancestor(current, "case_statement")
                if case_node is not None:
                    expr = next((c for c in case_node.children if c.type == "case_expression"), None)
                    if expr is not None:
                        guards.update(self._identifiers(self._node_source(parsed, expr)) & signal_names)
                if current.type == "case_item":
                    expr = next((c for c in current.children if c.type == "case_item_expression"), None)
                    if expr is not None:
                        guards.update(self._identifiers(self._node_source(parsed, expr)) & signal_names)
                break
            current = current.parent
        return guards

    def _smallest_containing_method(self, span, methods):
        candidates = []
        for method in methods:
            if method.get("verilog_kind") == "module_body":
                continue
            if self._contains(method, span):
                size = int(method.get("end_line") or 0) - int(method.get("start_line") or 0)
                candidates.append((size, method))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _smallest_containing_span(self, entity, spans):
        candidates = []
        for span in spans:
            if self._contains(span, entity):
                size = int(span.get("end_line") or 0) - int(span.get("start_line") or 0)
                candidates.append((size, span))
        return min(candidates, key=lambda item: (item[0], item[1].get("start_line") or 0))[1] if candidates else None

    @staticmethod
    def _contains(container, item) -> bool:
        if container.get("file_path") != item.get("file_path"):
            return False
        start = int(container.get("start_line") or 0)
        end = int(container.get("end_line") or 0)
        item_start = int(item.get("start_line") or 0)
        item_end = int(item.get("end_line") or item_start)
        if not (start and item_start and start <= item_start and end >= item_end):
            return False
        item_start_column = item.get("start_column")
        item_end_column = item.get("end_column")
        container_start_column = container.get("start_column")
        container_end_column = container.get("end_column")
        if (
            start == end == item_start == item_end
            and item_start_column is not None and item_end_column is not None
            and container_start_column is not None and container_end_column is not None
        ):
            return (
                int(container_start_column) <= int(item_start_column)
                and int(container_end_column) >= int(item_end_column)
            )
        return True

    def _parent_span_name(self, node, span_by_node) -> str:
        current = node.parent
        while current is not None:
            span = span_by_node.get(self._node_key(current))
            if span:
                return span["name"]
            current = current.parent
        return ""

    @staticmethod
    def _repair_role(kind: str, file_path: str) -> str:
        basename = file_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if kind == "instance_binding":
            return "top_level_wiring"
        if kind == "parameter_definition" or "csr" in basename or "regs" in basename:
            return "config_register"
        return "direct_driver"

    @staticmethod
    def _primary_signal(usage: Dict[str, Any]) -> str:
        for key in ("drives", "writes", "guards", "connects", "reads"):
            values = sorted(usage.get(key) or [])
            if values:
                return values[0]
        return ""

    @staticmethod
    def _semantic_summary(module_name, kind, source, usage, parent_name) -> str:
        parts = [f"patchable_span={kind}", f"module={module_name}"]
        if parent_name:
            parts.append(f"parent={parent_name}")
        for key in ("guards", "reads", "writes", "drives", "connects"):
            values = sorted(usage.get(key) or [])
            if values:
                parts.append(f"{key}={', '.join(values)}")
        excerpt = re.sub(r"\s+", " ", source or "").strip()
        if excerpt:
            parts.append(f"source_excerpt={excerpt[:700]}")
        return "; ".join(parts)

    @staticmethod
    def _span_edge(source_label, source_name, source, target_label, target_name, target,
                   relation_kind, description, reverse_description, confidence):
        return {
            "source_label": source_label,
            "source_name": source_name,
            "source_file_path": source.get("file_path", ""),
            "source_module_name": source.get("module_name", ""),
            "target_label": target_label,
            "target_name": target_name,
            "target_file_path": target.get("file_path", ""),
            "target_module_name": target.get("module_name", ""),
            "description": description,
            "reverse_description": reverse_description,
            "relation_kind": relation_kind,
            "parse_source": "ast+static",
            "parse_confidence": confidence,
        }

    @staticmethod
    def _dedupe_edges(edges):
        result = []
        seen = set()
        for edge in edges:
            key = (
                edge.get("source_label") or edge.get("method_name"),
                edge.get("source_name") or edge.get("method_signature"),
                edge.get("target_label"), edge.get("target_name"), edge.get("relation_kind"),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(edge)
        return result

    @staticmethod
    def _legacy_kind(kind: str) -> str:
        return {
            "assign": "continuous_assign",
            "instance": "instance_binding",
            "always": "procedural_block",
            "always_ff": "procedural_block",
            "always_comb": "procedural_block",
            "always_latch": "procedural_block",
            "initial": "procedural_block",
        }.get(kind, kind)

    @staticmethod
    def _identifiers(text: str) -> set[str]:
        return {
            item for item in re.findall(r"\b[A-Za-z_][\w$]*\b", text or "")
            if item not in PatchableSpanProjector.KEYWORDS
        }

    @staticmethod
    def _strip_comments(text: str) -> str:
        text = re.sub(r"/\*[\s\S]*?\*/", " ", text or "")
        return re.sub(r"//[^\n]*", " ", text)

    @staticmethod
    def _first_nonempty_line(source: str) -> str:
        return next((line.strip() for line in (source or "").splitlines() if line.strip()), "")[:500]

    @staticmethod
    def _ancestor(node, node_type):
        current = node.parent
        while current is not None:
            if current.type == node_type:
                return current
            current = current.parent
        return None

    @staticmethod
    def _nearest_region(node):
        current = node
        region_types = {"module_declaration", "interface_declaration", "package_declaration", "program_declaration"}
        while current is not None:
            if current.type in region_types:
                return current
            current = current.parent
        return None

    @staticmethod
    def _same_node(left, right) -> bool:
        return bool(
            left is not None and right is not None
            and left.type == right.type
            and left.start_byte == right.start_byte
            and left.end_byte == right.end_byte
        )

    @staticmethod
    def _node_key(node) -> Tuple[Any, ...]:
        return node.type, node.start_byte, node.end_byte

    @staticmethod
    def _walk(node) -> Iterable[Any]:
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    @staticmethod
    def _node_source(parsed, node) -> str:
        return parsed.source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    @staticmethod
    def _line_span(node) -> Tuple[int, int]:
        start = node.start_point.row + 1
        end = node.end_point.row + (1 if node.end_point.column > 0 else 0)
        return start, max(start, end)
