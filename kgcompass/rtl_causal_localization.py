from __future__ import annotations

import math
import heapq
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rtl_anchors import RTLAnchor, classify_mapped_anchor, extract_rtl_anchors, normalize_identifier
except Exception:  # pragma: no cover
    from .rtl_anchors import RTLAnchor, classify_mapped_anchor, extract_rtl_anchors, normalize_identifier


CAUSAL_LABELS = {
    "Signal", "Port", "Parameter", "State", "ClockDomain", "ResetDomain", "Instance",
    "Branch", "Condition", "Macro",
}
ANCHOR_LABELS = sorted(CAUSAL_LABELS | {"Class", "File"})
CAUSAL_RELATION_TYPES = (
    "WRITES", "DRIVES", "READS", "GUARDS", "FEEDS", "CONNECTS", "BINDS_PORT",
    "CLOCKED_BY", "RESET_BY", "PARAMETERIZES", "PROJECTS_TO",
)
CAUSAL_COSTS = {
    "WRITES": 0.25,
    "DRIVES": 0.25,
    "PROJECTS_TO": 0.25,
    "READS": 0.5,
    "GUARDS": 0.5,
    "FEEDS": 0.5,
    "CLOCKED_BY": 0.5,
    "RESET_BY": 0.5,
    "BINDS_PORT": 0.5,
    "CONNECTS": 0.5,
    "PARAMETERIZES": 1.0,
}
MODE_PRIORITY = {
    "direct_location": 0,
    "causal": 1,
    "structural_fallback": 2,
    "text_only": 3,
}
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_$]*\b")


class RTLCausalLocalizer:
    """Issue-grounded localization over PatchableSpan nodes.

    B1 uses the original embedding/Levenshtein text score with typed paths.
    B2 replaces Levenshtein with canonical-identifier IDF coverage and emits
    Issue-to-PatchableSpan inference paths.  Issue anchors are ranking
    features; they are not inserted into the initial inference path.  The RTL
    support path instead starts from a query-local semantic projection of the
    Issue node and contains typed RTL relationships only.
    """

    def __init__(
        self,
        driver,
        vector_weight: float = 0.6,
        decay_factor: float = 0.6,
        mode: str = "rtl_causal_b2",
    ):
        self.driver = driver
        self.vector_weight = float(vector_weight)
        self.decay_factor = float(decay_factor)
        self.mode = mode
        self.use_identifier_similarity = mode not in {"rtl_causal_b1", "b1"}

    @property
    def ranking_version(self) -> str:
        return (
            "rtl-causal-b2-issue-span-idf-support-v2"
            if self.use_identifier_similarity
            else "rtl-causal-b1-issue-span-v1"
        )

    def localize(
        self,
        issue_text: str,
        issue_mentioned_files: Optional[Sequence[str]] = None,
        limit: int = 50,
        max_cost: float = 3.0,
        evidence_text: str = "",
        query_embedding: Optional[Sequence[float]] = None,
        capture_ablation: bool = False,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        query_text = "\n".join(item for item in (issue_text, evidence_text) if item)
        anchors = extract_rtl_anchors(issue_text, source="issue")
        if evidence_text:
            anchors.extend(extract_rtl_anchors(evidence_text, source="validation"))
        anchors = self._merge_file_hints(anchors, issue_mentioned_files or [])
        anchors = self._dedupe_anchors(anchors)

        with self.driver.session() as session:
            self._map_anchors(session, anchors)
            spans = self._load_spans(
                session,
                query_text,
                include_levenshtein=not self.use_identifier_similarity,
                query_embedding=query_embedding,
            )
            grounded = self._effective_anchors(anchors, spans)
            identifier_anchors = self._scoring_anchors(grounded)
            identifier_scores = (
                self._identifier_scores(identifier_anchors, spans)
                if self.use_identifier_similarity
                else {}
            )
            self._attach_text_scores(spans, identifier_anchors, identifier_scores)

            direct_paths = self._direct_candidates(anchors, spans)
            semantic_seeds = self._semantic_issue_seeds(
                session,
                query_embedding=query_embedding,
            )
            causal_paths = self._causal_candidates(
                session,
                semantic_seeds,
                max_cost=max_cost,
            )
            structural_paths = self._structural_candidates(anchors, spans)

        ranked = self._relevance_candidates(
            spans,
            direct_paths,
            causal_paths,
            structural_paths,
            limit=limit,
        )
        evidence_entities = self._mapped_evidence_entities(anchors, ranked)
        root_issue = {
            "type": "issue",
            "name": "root",
            "issue_id": "root",
            "title": issue_text[:240],
            "content": issue_text,
            "similarity": 2.0,
            "distance": 0,
            "path": [],
        }
        results = {
            "patchable_spans": list(ranked),
            "edit_targets": list(ranked),
            "methods": [],
            "classes": [],
            "rtl_entities": list(evidence_entities),
            "evidence_entities": list(evidence_entities),
            "issues": [root_issue],
        }
        extracted_count = len(anchors)
        effective_count = len(identifier_anchors)
        eligible_count = len(self._scoring_anchors(anchors))
        metadata = {
            "ranking_version": self.ranking_version,
            "inference_path_schema": "issue-patchable-span-v1",
            "localization_mode": self.mode,
            "extracted_anchor_count": extracted_count,
            "grounded_anchor_count": len(grounded),
            "effective_anchor_count": effective_count,
            "failed_anchor_count": eligible_count - effective_count,
            "structural_scope_anchor_count": len([
                anchor for anchor in grounded if anchor not in identifier_anchors
            ]),
            "fusion_mode": "semantic_identifier_idf_coverage"
            if self.use_identifier_similarity and effective_count
            else "semantic_only"
            if self.use_identifier_similarity
            else "semantic_levenshtein",
            "anchors": [anchor.to_dict() for anchor in anchors],
            "semantic_issue_seeds": semantic_seeds,
            "max_causal_cost": float(max_cost),
            "mode_counts": dict(self._mode_counts(ranked)),
        }
        if capture_ablation:
            # Export pre-truncation rankings for offline evaluation. Gold
            # patches are deliberately unavailable to this computation.
            counts = dict(self._identifier_scores(identifier_anchors, spans))
            canonical = {}
            for anchor in identifier_anchors:
                canonical.setdefault(anchor.normalized.canonical_identifier, anchor)
            neighborhoods = {
                str(span["element_id"]): self._identifier_neighborhood(span)
                for span in spans
            }
            hits = []
            for anchor in canonical.values():
                exact = anchor.normalized.exact_identifier
                base = anchor.normalized.base_identifier
                exact_hits = {key for key, (exacts, _) in neighborhoods.items() if exact and exact in exacts}
                hits.append(exact_hits or {key for key, (_, bases) in neighborhoods.items() if base and base in bases})
            unweighted = {
                key: sum(key in matching for matching in hits) / len(hits) if hits else 0.0
                for key in neighborhoods
            }
            def compact(span):
                return {key: span.get(key) for key in (
                    "name", "file_path", "start_line", "end_line", "span_kind",
                    "parse_source", "semantic_similarity", "identifier_similarity",
                    "distance", "distance_mode", "score",
                )}
            def score(span, identifier):
                semantic = float(span.get("semantic_similarity") or 0.0)
                return (self.vector_weight * semantic + (1 - self.vector_weight) * identifier[str(span["element_id"])]) if identifier_anchors else semantic
            by_semantic = sorted(spans, key=lambda span: (-float(span.get("semantic_similarity") or 0.0), self._span_identity(span)))
            by_identifier = sorted(spans, key=lambda span: (-score(span, unweighted), self._span_identity(span)))
            by_idf = sorted(spans, key=lambda span: (-score(span, counts), self._span_identity(span)))
            full = self._layer_candidates(spans, direct_paths, causal_paths, structural_paths, limit=max(1, len(spans)))
            metadata["ablation"] = {
                "candidate_count": len(spans),
                "vector_weight": self.vector_weight,
                "decay_factor": self.decay_factor,
                "rankings": {
                    key: [compact(span) for span in values]
                    for key, values in {
                        "semantic": by_semantic,
                        "semantic_identifier": by_identifier,
                        "semantic_identifier_idf": by_idf,
                        "full_rtl_path": full,
                    }.items()
                },
            }
        return results, metadata

    def _relevance_candidates(
        self,
        spans: Sequence[Dict[str, Any]],
        direct_paths: Dict[str, Dict[str, Any]],
        causal_paths: Dict[str, Dict[str, Any]],
        structural_paths: Dict[str, Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Rank by semantic--identifier relevance and attach paths as evidence.

        RTL paths describe why a selected span is structurally supported.  They
        do not override the relevance order: doing so promotes weakly related
        spans merely because they happen to be graph-reachable.
        """
        ranked = []
        for span in spans:
            span_id = str(span.get("element_id"))
            if span_id in direct_paths:
                mode, path_info = "direct_location", direct_paths[span_id]
            elif span_id in causal_paths:
                mode, path_info = "causal", causal_paths[span_id]
            elif span_id in structural_paths:
                mode, path_info = "structural_fallback", structural_paths[span_id]
            else:
                mode, path_info = "text_only", {}
            distance = path_info.get("distance")
            ranked.append(self._candidate_payload(
                span,
                mode,
                float(span.get("text_score") or 0.0),
                float(distance) if distance is not None else None,
                path_info,
            ))
        ranked.sort(key=lambda candidate: (
            -float(candidate.get("score") or 0.0),
            self._span_identity(candidate),
        ))
        ranked = ranked[:max(0, int(limit))]
        for rank, candidate in enumerate(ranked, 1):
            candidate["global_rank"] = rank
            candidate["mode_rank"] = rank
        return ranked

    @staticmethod
    def _merge_file_hints(anchors: List[RTLAnchor], file_hints: Sequence[str]) -> List[RTLAnchor]:
        for hint in file_hints:
            normalized_path = str(hint or "").replace("\\", "/").strip()
            if not normalized_path:
                continue
            anchors.append(RTLAnchor(
                text=normalized_path,
                anchor_type="file",
                normalized=normalize_identifier(normalized_path.rsplit("/", 1)[-1]),
                file_path=normalized_path,
                source="issue_file_hint",
            ))
        return anchors

    @staticmethod
    def _dedupe_anchors(anchors: Sequence[RTLAnchor]) -> List[RTLAnchor]:
        result = []
        seen = set()
        for anchor in anchors:
            if anchor.key in seen:
                continue
            seen.add(anchor.key)
            result.append(anchor)
        return result

    def _map_anchors(self, session, anchors: Sequence[RTLAnchor]) -> None:
        scope_files = self._scope_file_ranks(anchors)
        for anchor in anchors:
            if anchor.anchor_type in {"file", "direct_location"}:
                records = session.run(
                    """
                    MATCH (n)
                    WHERE (n:File OR n:Class)
                      AND (
                        toLower(coalesce(n.path, n.file_path, '')) = toLower($path)
                        OR toLower(coalesce(n.path, n.file_path, '')) ENDS WITH '/' + toLower($path)
                        OR toLower(coalesce(n.path, n.file_path, '')) ENDS WITH '/' + toLower($basename)
                      )
                    RETURN id(n) AS element_id, labels(n) AS labels,
                           coalesce(n.name, n.path, '') AS name,
                           coalesce(n.path, n.file_path, '') AS file_path,
                           coalesce(n.module_name, n.name, '') AS module_name,
                           n.start_line AS start_line, n.end_line AS end_line
                    LIMIT 20
                    """,
                    path=anchor.file_path,
                    basename=anchor.file_path.rsplit("/", 1)[-1],
                )
                anchor.matched_nodes = [record.data() for record in records]
                if anchor.matched_nodes:
                    anchor.match_mode = "exact_file"
                    anchor.mapping_confidence = 1.0
                continue

            exact = anchor.normalized.exact_identifier
            base = anchor.normalized.base_identifier
            records = session.run(
                """
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN $labels)
                  AND (
                    toLower(coalesce(n.signal_name, '')) = $exact
                    OR toLower(coalesce(n.name, '')) = $exact
                    OR toLower(coalesce(n.name, '')) ENDS WITH '.' + $exact
                    OR toLower(coalesce(n.module_name, '')) = $exact
                    OR ($allow_declaration AND toLower(coalesce(n.declaration, '')) =~ $declaration_pattern)
                  )
                RETURN id(n) AS element_id, labels(n) AS labels,
                       coalesce(n.name, '') AS name,
                       coalesce(n.file_path, n.path, '') AS file_path,
                       coalesce(n.module_name, '') AS module_name,
                       coalesce(n.signal_name, '') AS signal_name,
                       n.start_line AS start_line, n.end_line AS end_line,
                       n.direction AS direction
                LIMIT 80
                """,
                labels=ANCHOR_LABELS,
                exact=exact,
                allow_declaration=bool(anchor.explicit),
                declaration_pattern=(
                    ".*(^|[^a-z0-9_$])" + re.escape(exact) + "([^a-z0-9_$]|$).*"
                ),
            )
            nodes = [record.data() for record in records]
            match_mode = "exact"
            confidence = 1.0
            if not nodes and base:
                records = session.run(
                    """
                    MATCH (n)
                    WHERE any(label IN labels(n) WHERE label IN $labels)
                      AND (
                        toLower(coalesce(n.signal_name, n.name, '')) STARTS WITH $base
                        OR toLower(coalesce(n.name, '')) ENDS WITH '.' + $base
                      )
                    RETURN id(n) AS element_id, labels(n) AS labels,
                           coalesce(n.name, '') AS name,
                           coalesce(n.file_path, n.path, '') AS file_path,
                           coalesce(n.module_name, '') AS module_name,
                           coalesce(n.signal_name, '') AS signal_name,
                           n.start_line AS start_line, n.end_line AS end_line,
                           n.direction AS direction
                    LIMIT 120
                    """,
                    labels=ANCHOR_LABELS,
                    base=base,
                )
                nodes = [
                    record.data()
                    for record in records
                    if normalize_identifier(
                        record.get("signal_name") or str(record.get("name") or "").rsplit(".", 1)[-1]
                    ).base_identifier == base
                ]
                match_mode = "base"
                confidence = 0.75

            nodes.sort(key=lambda node: self._mapping_sort_key(node, scope_files))
            if nodes and scope_files:
                best_scope_rank = self._mapping_sort_key(nodes[0], scope_files)[0]
                if best_scope_rank == 0:
                    nodes = [
                        node for node in nodes
                        if self._mapping_sort_key(node, scope_files)[0] == 0
                    ]
                else:
                    nodes = []
            anchor.matched_nodes = nodes[:20]
            if anchor.matched_nodes:
                anchor.match_mode = match_mode
                anchor.mapping_confidence = confidence

    @staticmethod
    def _scope_file_ranks(anchors: Sequence[RTLAnchor]) -> Dict[str, int]:
        explicit_files = [
            anchor.file_path.lower()
            for anchor in anchors
            if anchor.anchor_type in {"file", "direct_location"}
            and anchor.file_path
            and anchor.source in {"issue", "validation"}
        ]
        inferred_files = [
            anchor.file_path.lower()
            for anchor in anchors
            if anchor.anchor_type in {"file", "direct_location"}
            and anchor.file_path
            and anchor.file_path.lower() not in explicit_files
        ]
        primary = explicit_files or inferred_files[:1]
        ranks = {path: 0 for path in primary}
        next_rank = 1
        for path in inferred_files:
            if path in ranks:
                continue
            ranks[path] = next_rank
            next_rank += 1
        return ranks

    @staticmethod
    def _mapping_sort_key(node: Dict[str, Any], scope_files: Dict[str, int]) -> Tuple[int, str, str]:
        file_path = str(node.get("file_path") or "").lower()
        scope_rank = min(
            (
                rank
                for scope, rank in scope_files.items()
                if file_path == scope or file_path.endswith("/" + scope)
            ),
            default=10**6,
        )
        return (scope_rank, file_path, str(node.get("name") or ""))

    @staticmethod
    def _load_spans(
        session,
        query_text: str,
        include_levenshtein: bool,
        query_embedding: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        levenshtein = (
            "apoc.text.levenshteinSimilarity(root_text, "
            "toLower(coalesce(p.semantic_summary, p.source_code, p.name, '')))"
            if include_levenshtein
            else "0.0"
        )
        query_embedding_clause = (
            "WITH $query_embedding AS query_embedding"
            if query_embedding is not None
            else "MATCH (root:Issue {id: 'root'}) WHERE root.embedding IS NOT NULL WITH root.embedding AS query_embedding"
        )
        query = f"""
        {query_embedding_clause}
        MATCH (p:PatchableSpan)
        WHERE p.embedding IS NOT NULL
        WITH query_embedding, p, toLower($query_text) AS root_text
        RETURN id(p) AS element_id,
               p.name AS name, p.file_path AS file_path, p.module_name AS module_name,
               p.start_line AS start_line, p.end_line AS end_line,
               p.start_column AS start_column, p.end_column AS end_column,
               p.span_kind AS span_kind, p.patch_granularity AS patch_granularity,
               p.parent_span AS parent_span, p.source_code AS source_code,
               p.semantic_summary AS semantic_summary, p.repair_role AS repair_role,
               p.parse_source AS parse_source, p.parse_confidence AS parse_confidence,
               p.verilog_kind AS verilog_kind, p.rtl_kind AS rtl_kind,
               p.reads_signals AS reads_signals, p.writes_signals AS writes_signals,
               p.drives_signals AS drives_signals, p.connects_signals AS connects_signals,
               p.guards_signals AS guards_signals, p.port_bindings AS port_bindings,
               p.projection_source AS projection_source,
               gds.similarity.cosine(query_embedding, p.embedding) AS semantic_similarity,
               {levenshtein} AS levenshtein_similarity
        """
        return [record.data() for record in session.run(
            query,
            query_text=query_text,
            query_embedding=list(query_embedding) if query_embedding is not None else None,
        )]

    def _effective_anchors(
        self,
        anchors: Sequence[RTLAnchor],
        spans: Sequence[Dict[str, Any]],
    ) -> List[RTLAnchor]:
        # A repository-level identifier-neighborhood hit is itself a successful
        # grounding even when no standalone semantic node was extracted.
        neighborhoods = [self._identifier_neighborhood(span) for span in spans]
        effective = []
        for anchor in anchors:
            if anchor.matched_nodes:
                effective.append(anchor)
                continue
            exact = anchor.normalized.exact_identifier
            base = anchor.normalized.base_identifier
            if any(exact in exacts or (base and base in bases) for exacts, bases in neighborhoods):
                anchor.match_mode = "span_neighborhood"
                anchor.mapping_confidence = 0.6
                effective.append(anchor)
        return effective

    @staticmethod
    def _scoring_anchors(anchors: Sequence[RTLAnchor]) -> List[RTLAnchor]:
        # File and direct-location anchors constrain structural scope.  Only
        # grounded RTL identifiers contribute to identifier coverage.
        return [anchor for anchor in anchors if anchor.anchor_type == "identifier"]

    @staticmethod
    def _identifier_neighborhood(span: Dict[str, Any]) -> Tuple[set, set]:
        values = [
            span.get("file_path"),
            str(span.get("file_path") or "").rsplit("/", 1)[-1],
            span.get("module_name"),
            span.get("name"),
            span.get("span_kind"),
            *(span.get("reads_signals") or []),
            *(span.get("writes_signals") or []),
            *(span.get("drives_signals") or []),
            *(span.get("connects_signals") or []),
            *(span.get("guards_signals") or []),
        ]
        values.extend(IDENTIFIER_RE.findall(str(span.get("source_code") or "")))
        exacts = set()
        bases = set()
        for value in values:
            if not value:
                continue
            normalized = normalize_identifier(str(value).rsplit(".", 1)[-1])
            if normalized.exact_identifier:
                exacts.add(normalized.exact_identifier)
            if normalized.base_identifier:
                bases.add(normalized.base_identifier)
        return exacts, bases

    def _identifier_scores(
        self,
        anchors: Sequence[RTLAnchor],
        spans: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not anchors:
            return {str(span.get("element_id")): 0.0 for span in spans}

        # Grounding remains exact-first with scoped base-form fallback, but
        # ranking uses one canonical form per identifier.  This implements a
        # simple IDF-weighted coverage ratio rather than separate exact/base
        # similarities and avoids counting RTL role variants more than once.
        canonical_identifiers: List[str] = []
        seen_identifiers = set()
        for anchor in anchors:
            canonical = anchor.normalized.canonical_identifier
            if not canonical or canonical in seen_identifiers:
                continue
            seen_identifiers.add(canonical)
            canonical_identifiers.append(canonical)

        neighborhoods: Dict[str, Tuple[set, set]] = {}
        for span in spans:
            neighborhoods[str(span.get("element_id"))] = self._identifier_neighborhood(span)

        # Exact identifiers are authoritative.  A role-normalized base form is
        # used for an anchor only when no PatchableSpan contains its exact
        # spelling, preventing _i/_o/_q variants from creating duplicate weak
        # matches when an exact repository object is available.
        matched_spans: Dict[str, set[str]] = {}
        for anchor in anchors:
            canonical = anchor.normalized.canonical_identifier
            if canonical not in canonical_identifiers or canonical in matched_spans:
                continue
            exact = anchor.normalized.exact_identifier
            base = anchor.normalized.base_identifier
            exact_hits = {
                span_id
                for span_id, (exact_forms, _) in neighborhoods.items()
                if exact and exact in exact_forms
            }
            matched_spans[canonical] = exact_hits or {
                span_id
                for span_id, (_, base_forms) in neighborhoods.items()
                if base and base in base_forms
            }
        document_count = max(1, len(spans))
        document_frequency = {
            identifier: len(matched_spans.get(identifier, set()))
            for identifier in canonical_identifiers
        }
        idf = {
            identifier: math.log(
                (document_count + 1) / (document_frequency[identifier] + 1)
            ) + 1.0
            for identifier in canonical_identifiers
        }
        denominator = sum(idf.values())
        scores: Dict[str, float] = {}
        for span_id in neighborhoods:
            numerator = sum(
                idf[identifier]
                for identifier in canonical_identifiers
                if span_id in matched_spans.get(identifier, set())
            )
            scores[span_id] = numerator / denominator if denominator else 0.0
        return scores

    def _attach_text_scores(
        self,
        spans: Sequence[Dict[str, Any]],
        effective: Sequence[RTLAnchor],
        identifier_scores: Dict[str, float],
    ) -> None:
        for span in spans:
            semantic = float(span.get("semantic_similarity") or 0.0)
            if self.use_identifier_similarity:
                identifier = float(identifier_scores.get(str(span.get("element_id")), 0.0))
                text_score = (
                    self.vector_weight * semantic + (1.0 - self.vector_weight) * identifier
                    if effective
                    else semantic
                )
                span["identifier_similarity"] = identifier
                span["levenshtein_similarity"] = None
                span["fusion_mode"] = (
                    "semantic_identifier_idf_coverage" if effective else "semantic_only"
                )
            else:
                lexical = float(span.get("levenshtein_similarity") or 0.0)
                text_score = self.vector_weight * semantic + (1.0 - self.vector_weight) * lexical
                span["identifier_similarity"] = None
                span["fusion_mode"] = "semantic_levenshtein"
            span["embedding_similarity"] = semantic
            span["effective_anchor_count"] = len(effective)
            span["text_score"] = text_score

    @staticmethod
    def _direct_candidates(
        anchors: Sequence[RTLAnchor],
        spans: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for anchor in anchors:
            if anchor.anchor_type != "direct_location" or not anchor.file_path or not anchor.line:
                continue
            matches = [
                span for span in spans
                if RTLCausalLocalizer._same_file(span.get("file_path"), anchor.file_path)
                and int(span.get("start_line") or 0) <= anchor.line <= int(span.get("end_line") or 0)
            ]
            if not matches:
                continue
            matches.sort(key=lambda span: (
                int(span.get("end_line") or 0) - int(span.get("start_line") or 0),
                int(span.get("start_line") or 0),
            ))
            span = matches[0]
            result[str(span["element_id"])] = {
                "distance": 0.0,
                "path": [],
                "matched_anchors": [anchor.to_dict(include_nodes=False)],
                "location_confidence": anchor.mapping_confidence or 0.9,
            }
        return result

    @staticmethod
    def _semantic_issue_seeds(
        session,
        *,
        query_embedding: Optional[Sequence[float]] = None,
        limit: int = 24,
    ) -> List[Dict[str, Any]]:
        """Project the Issue query onto a small set of RTL semantic entities.

        This projection is deliberately independent of identifier anchors.
        It supplies query-local entry points for the bounded RTL shortest-path
        search while anchors remain an IDF-weighted relevance feature.
        """

        embedding_clause = (
            "WITH $query_embedding AS query_embedding"
            if query_embedding is not None
            else (
                "MATCH (root:Issue {id: 'root'}) "
                "WHERE root.embedding IS NOT NULL "
                "WITH root.embedding AS query_embedding"
            )
        )
        query = f"""
        {embedding_clause}
        MATCH (seed)
        WHERE any(label IN labels(seed) WHERE label IN $labels)
          AND seed.embedding IS NOT NULL
        WITH seed, gds.similarity.cosine(query_embedding, seed.embedding) AS similarity
        ORDER BY similarity DESC, id(seed)
        LIMIT $limit
        RETURN id(seed) AS element_id, labels(seed) AS labels,
               coalesce(seed.name, seed.signal_name, seed.path, '') AS name,
               coalesce(seed.file_path, seed.path, '') AS file_path,
               coalesce(seed.module_name, '') AS module_name,
               similarity
        """
        return [
            record.data()
            for record in session.run(
                query,
                labels=sorted(CAUSAL_LABELS),
                limit=max(1, int(limit)),
                query_embedding=(
                    list(query_embedding) if query_embedding is not None else None
                ),
            )
        ]

    def _causal_candidates(
        self,
        session,
        semantic_seeds: Sequence[Dict[str, Any]],
        max_cost: float,
    ) -> Dict[str, Dict[str, Any]]:
        seed_by_id = {
            int(seed["element_id"]): dict(seed)
            for seed in semantic_seeds
            if seed.get("element_id") is not None
            and set(seed.get("labels") or []) & CAUSAL_LABELS
        }
        if not seed_by_id:
            return {}

        relation_pattern = "|".join(CAUSAL_RELATION_TYPES)
        query = f"""
        MATCH (seed)-[rel:{relation_pattern}]-(reachable)
        WHERE id(seed) IN $seed_ids
        WITH DISTINCT rel
        WITH startNode(rel) AS source, endNode(rel) AS target, rel
        RETURN id(rel) AS relationship_id,
               id(source) AS source_id, labels(source) AS source_labels,
               coalesce(source.name, source.signal_name, source.path, '') AS source_name,
               id(target) AS target_id, labels(target) AS target_labels,
               coalesce(target.name, target.signal_name, target.path, '') AS target_name,
               type(rel) AS type, coalesce(rel.description, '') AS description,
               coalesce(rel.relation_kind, toLower(type(rel))) AS relation_kind
        """
        adjacency: Dict[int, List[Tuple[int, Dict[str, Any], bool]]] = defaultdict(list)
        nodes: Dict[int, Dict[str, Any]] = {}
        # Expand each node once, rather than enumerating every cyclic walk.
        # Four frontier expansions contain every edge needed by the same
        # four-hop Dijkstra search, without an arbitrary 50,000-edge cutoff.
        frontier = set(seed_by_id)
        expanded = set()
        edge_records = {}
        for _ in range(4):
            if not frontier:
                break
            next_frontier = set()
            frontier_ids = sorted(frontier)
            for offset in range(0, len(frontier_ids), 500):
                for record in session.run(query, seed_ids=frontier_ids[offset:offset + 500]):
                    row = dict(record)
                    key = row.get("relationship_id")
                    if key is None:  # Minimal test doubles need no DB IDs.
                        key = (row["source_id"], row["target_id"], row["type"])
                    edge_records[key] = row
                    next_frontier.update((int(row["source_id"]), int(row["target_id"])))
            expanded.update(frontier)
            frontier = next_frontier - expanded
        for record in sorted(edge_records.values(), key=lambda row: (int(row["source_id"]), int(row["target_id"]), row["type"], row.get("description") or "")):
            source_id = int(record.get("source_id"))
            target_id = int(record.get("target_id"))
            nodes[source_id] = {
                "element_id": source_id,
                "labels": list(record.get("source_labels") or []),
                "name": str(record.get("source_name") or ""),
            }
            nodes[target_id] = {
                "element_id": target_id,
                "labels": list(record.get("target_labels") or []),
                "name": str(record.get("target_name") or ""),
            }
            relationship = {
                "type": str(record.get("type") or ""),
                "description": str(record.get("description") or ""),
                "relation_kind": str(record.get("relation_kind") or ""),
            }
            adjacency[source_id].append((target_id, relationship, True))
            adjacency[target_id].append((source_id, relationship, False))

        # Multi-source bounded Dijkstra.  The semantic projection into each
        # seed has zero causal cost; only typed RTL relationships contribute.
        heap: List[Tuple[float, int, int, int, int, List[Dict[str, Any]]]] = []
        sequence = 0
        for seed_id in sorted(seed_by_id):
            heapq.heappush(heap, (0.0, 0, sequence, seed_id, seed_id, []))
            sequence += 1
        best_state: Dict[Tuple[int, int], float] = {}
        result: Dict[str, Dict[str, Any]] = {}
        while heap:
            cost, hops, _, node_id, origin_id, path = heapq.heappop(heap)
            state = (node_id, hops)
            if cost > best_state.get(state, math.inf) or cost > float(max_cost):
                continue
            best_state[state] = cost
            node = nodes.get(node_id) or seed_by_id.get(node_id) or {}
            if hops and "PatchableSpan" in set(node.get("labels") or []):
                span_id = str(node_id)
                current = result.get(span_id)
                if current is None or cost < float(current["distance"]):
                    result[span_id] = {
                        "distance": cost,
                        "path": path,
                        "matched_anchors": [],
                        "semantic_seed": seed_by_id.get(origin_id, {}),
                    }
            if hops >= 4:
                continue
            for neighbor_id, relationship, forward in adjacency.get(node_id, []):
                edge_cost = CAUSAL_COSTS.get(relationship["type"], 1.0)
                next_cost = cost + edge_cost
                next_state = (neighbor_id, hops + 1)
                if next_cost >= best_state.get(next_state, math.inf) or next_cost > float(max_cost):
                    continue
                start = nodes.get(node_id) or seed_by_id.get(node_id) or {}
                end = nodes.get(neighbor_id) or seed_by_id.get(neighbor_id) or {}
                step = {
                    "start_node": start.get("name"),
                    "start_labels": start.get("labels") or [],
                    "end_node": end.get("name"),
                    "end_labels": end.get("labels") or [],
                    **relationship,
                    "traversal_direction": "forward" if forward else "reverse",
                    "contributes_to_causal_distance": True,
                }
                best_state[next_state] = next_cost
                heapq.heappush(
                    heap,
                    (
                        next_cost,
                        hops + 1,
                        sequence,
                        neighbor_id,
                        origin_id,
                        [*path, step],
                    ),
                )
                sequence += 1
        return result

    @staticmethod
    def _structural_candidates(
        anchors: Sequence[RTLAnchor],
        spans: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        scope_ranks = RTLCausalLocalizer._scope_file_ranks(anchors)
        file_anchors = [
            anchor for anchor in anchors
            if anchor.anchor_type in {"file", "direct_location"} and anchor.file_path
        ]
        module_anchors = []
        for anchor in anchors:
            if classify_mapped_anchor(anchor) != "structural":
                continue
            for node in anchor.matched_nodes:
                if "Class" in (node.get("labels") or []):
                    module_anchors.append((anchor, str(node.get("name") or node.get("module_name") or "")))

        for span in spans:
            span_id = str(span.get("element_id"))
            matched = []
            distance = None
            scope_rank = 10**6
            for anchor in file_anchors:
                if RTLCausalLocalizer._same_file(span.get("file_path"), anchor.file_path):
                    matched.append(anchor)
                    distance = min(distance if distance is not None else 1.0, 1.0)
                    scope_rank = min(scope_rank, scope_ranks.get(anchor.file_path.lower(), 10**6))
            for anchor, module_name in module_anchors:
                if module_name and str(span.get("module_name") or "").lower() == module_name.lower():
                    matched.append(anchor)
                    distance = min(distance if distance is not None else 0.5, 0.5)
                    node_files = [
                        str(node.get("file_path") or "").lower()
                        for node in anchor.matched_nodes
                        if "Class" in (node.get("labels") or [])
                    ]
                    if node_files:
                        scope_rank = min(
                            scope_rank,
                            *(scope_ranks.get(path, 10**6) for path in node_files),
                        )
            if distance is None:
                continue
            result[span_id] = {
                "distance": distance,
                "scope_rank": scope_rank,
                "path": [],
                "matched_anchors": [anchor.to_dict(include_nodes=False) for anchor in matched],
            }
        return result

    def _layer_candidates(
        self,
        spans: Sequence[Dict[str, Any]],
        direct_paths: Dict[str, Dict[str, Any]],
        causal_paths: Dict[str, Dict[str, Any]],
        structural_paths: Dict[str, Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        span_by_id = {str(span.get("element_id")): span for span in spans}

        def build_list(mode: str, paths: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
            candidates = []
            for span_id, path_info in paths.items():
                span = span_by_id.get(span_id)
                if not span:
                    continue
                distance = float(path_info.get("distance") or 0.0)
                if mode == "direct_location":
                    score = float(path_info.get("location_confidence") or 1.0)
                else:
                    score = float(span.get("text_score") or 0.0) * (self.decay_factor ** distance)
                candidates.append(self._candidate_payload(span, mode, score, distance, path_info))
            if mode == "structural_fallback":
                candidates.sort(key=lambda candidate: (
                    int(candidate.get("scope_rank") if candidate.get("scope_rank") is not None else 10**6),
                    *self._within_mode_sort_key(candidate),
                ))
            else:
                candidates.sort(key=self._within_mode_sort_key)
            for rank, candidate in enumerate(candidates, 1):
                candidate["mode_rank"] = rank
            return candidates

        direct = build_list("direct_location", direct_paths)
        causal = build_list("causal", causal_paths)
        structural = build_list("structural_fallback", structural_paths)
        excluded = set(direct_paths) | set(causal_paths) | set(structural_paths)
        text = [
            self._candidate_payload(span, "text_only", float(span.get("text_score") or 0.0), None, {})
            for span_id, span in span_by_id.items()
            if span_id not in excluded
        ]
        text.sort(key=self._within_mode_sort_key)
        for rank, candidate in enumerate(text, 1):
            candidate["mode_rank"] = rank

        ranked = []
        seen = set()
        for candidate in [*direct, *causal, *structural, *text]:
            identity = self._span_identity(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            ranked.append(candidate)
            if len(ranked) >= int(limit):
                break
        for rank, candidate in enumerate(ranked, 1):
            candidate["global_rank"] = rank
        return ranked

    def _candidate_payload(
        self,
        span: Dict[str, Any],
        mode: str,
        score: float,
        distance: Optional[float],
        path_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        excluded = {"element_id", "text_score"}
        payload = {key: value for key, value in span.items() if key not in excluded}
        matched_anchors = self._dedupe_anchor_payloads(
            path_info.get("matched_anchors") or []
        )
        rtl_path = list(path_info.get("path") or [])
        inference_paths = self._issue_inference_paths(
            span=span,
            mode=mode,
            semantic_seed=path_info.get("semantic_seed") or {},
            rtl_path=rtl_path,
        )
        primary_inference_path = inference_paths[0] if inference_paths else []
        payload.update({
            "type": "patchable_span",
            "signature": span.get("name"),
            "entity_role": "edit_target",
            "score": score,
            "similarity": score,
            "distance": distance,
            "causal_distance": distance if mode == "causal" else None,
            "structural_distance": distance if mode == "structural_fallback" else None,
            "distance_mode": mode,
            "matched_anchors": matched_anchors,
            "scope_rank": path_info.get("scope_rank"),
            # The causal/support path contains RTL relationships only.  The
            # compatibility `path` field now carries the complete inference
            # chain beginning at Issue:root.
            "causal_path": rtl_path if mode == "causal" else [],
            "rtl_support_path": rtl_path if mode == "causal" else [],
            "grounding_path": [
                path[0]
                for path in inference_paths
                if path and path[0].get("relation_kind") == "semantic_projection"
            ],
            "inference_paths": inference_paths,
            "path": primary_inference_path,
            "path_schema": "issue-patchable-span-v1",
            "ranking_version": self.ranking_version,
        })
        return payload

    @staticmethod
    def _dedupe_anchor_payloads(
        anchors: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen = set()
        for anchor in anchors:
            normalized = anchor.get("normalized") or {}
            key = (
                str(anchor.get("anchor_type") or ""),
                str(normalized.get("exact_identifier") or anchor.get("text") or ""),
                str(anchor.get("file_path") or ""),
                anchor.get("line"),
                str(anchor.get("source") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(anchor))
        return result

    @classmethod
    def _issue_inference_paths(
        cls,
        *,
        span: Dict[str, Any],
        mode: str,
        semantic_seed: Dict[str, Any],
        rtl_path: Sequence[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Build an Issue-to-PatchableSpan explanation without anchor nodes.

        For a causal result the first query-local step records semantic
        projection from the Issue to the RTL seed.  It is not part of the RTL
        support path or its distance.  Every following step is a real typed RTL
        relationship returned from the KG search.
        """

        target = str(span.get("name") or span.get("signature") or "PatchableSpan")
        if mode == "causal" and rtl_path:
            seed_name = str(
                semantic_seed.get("name")
                or rtl_path[0].get("start_node")
                or "RTL semantic seed"
            )
            seed_labels = list(
                semantic_seed.get("labels")
                or rtl_path[0].get("start_labels")
                or []
            )
            projection = {
                "start_node": "Issue:root",
                "start_labels": ["Issue"],
                "end_node": seed_name,
                "end_labels": seed_labels,
                "type": "SEMANTIC_PROJECTS_TO",
                "relation_kind": "semantic_projection",
                "description": "Issue semantics project to an RTL entity used by the shortest-path query",
                "semantic_similarity": semantic_seed.get("similarity"),
                "contributes_to_causal_distance": False,
            }
            return [[projection, *(dict(step) for step in rtl_path)]]

        relation_type, relation_kind, description = {
            "direct_location": (
                "LOCATES",
                "direct_location",
                "an explicit source location resolves to this PatchableSpan",
            ),
            "structural_fallback": (
                "SCOPES",
                "structural_fallback",
                "an issue file or module scope contains this PatchableSpan",
            ),
        }.get(
            mode,
            (
                "SEMANTIC_RETRIEVAL",
                "text_only",
                "semantic fallback without an RTL support path",
            ),
        )
        return [[{
            "start_node": "Issue:root",
            "start_labels": ["Issue"],
            "end_node": target,
            "end_labels": ["PatchableSpan"],
            "type": relation_type,
            "relation_kind": relation_kind,
            "description": description,
            "contributes_to_causal_distance": False,
        }]]

    @staticmethod
    def _anchor_display_name(anchor: Dict[str, Any]) -> str:
        anchor_type = str(anchor.get("anchor_type") or "anchor")
        text = str(anchor.get("text") or anchor.get("file_path") or "unknown")
        prefix = {
            "direct_location": "SourceLocation",
            "file": "FileAnchor",
            "identifier": "RTLAnchor",
        }.get(anchor_type, "Anchor")
        return f"{prefix}:{text}"

    @staticmethod
    def _anchor_labels(anchor: Dict[str, Any]) -> List[str]:
        return {
            "direct_location": ["SourceLocation"],
            "file": ["File"],
            "identifier": ["RTLAnchor"],
        }.get(str(anchor.get("anchor_type") or ""), ["Anchor"])

    @staticmethod
    def _within_mode_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, str, int, str]:
        distance = candidate.get("distance")
        return (
            -float(candidate.get("score") or 0.0),
            float(distance) if distance is not None else math.inf,
            str(candidate.get("file_path") or ""),
            int(candidate.get("start_line") or 0),
            str(candidate.get("name") or ""),
        )

    @staticmethod
    def _span_identity(candidate: Dict[str, Any]) -> Tuple[str, int, int, str]:
        return (
            str(candidate.get("file_path") or ""),
            int(candidate.get("start_line") or 0),
            int(candidate.get("end_line") or 0),
            str(candidate.get("name") or ""),
        )

    @staticmethod
    def _same_file(left: Any, right: Any) -> bool:
        left_path = str(left or "").replace("\\", "/").lower()
        right_path = str(right or "").replace("\\", "/").lower()
        return bool(left_path and right_path) and (
            left_path == right_path
            or left_path.endswith("/" + right_path)
            or right_path.endswith("/" + left_path)
        )

    @staticmethod
    def _format_path(nodes: Sequence[Dict[str, Any]], relationships: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for index, relationship in enumerate(relationships):
            start = nodes[index] if index < len(nodes) else {}
            end = nodes[index + 1] if index + 1 < len(nodes) else {}
            source_id = relationship.get("source_id")
            target_id = relationship.get("target_id")
            traversed_forward = (
                source_id is None
                or target_id is None
                or (
                    source_id == start.get("element_id")
                    and target_id == end.get("element_id")
                )
            )
            result.append({
                "start_node": start.get("name"),
                "start_labels": start.get("labels") or [],
                "end_node": end.get("name"),
                "end_labels": end.get("labels") or [],
                "type": relationship.get("type"),
                "relation_kind": relationship.get("relation_kind"),
                "description": relationship.get("description"),
                "traversal_direction": "forward" if traversed_forward else "reverse",
            })
        return result

    @staticmethod
    def _mapped_evidence_entities(
        anchors: Sequence[RTLAnchor],
        ranked: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked_span_paths = {
            str(item.get("file_path") or ""): item.get("path") or []
            for item in ranked
        }
        result = []
        seen = set()
        for anchor in anchors:
            for node in anchor.matched_nodes:
                key = str(node.get("element_id") or "")
                if not key or key in seen:
                    continue
                seen.add(key)
                labels = list(node.get("labels") or [])
                entity_type = next((label for label in labels if label not in {"Issue"}), "entity").lower()
                result.append({
                    "type": entity_type,
                    "name": node.get("name"),
                    "file_path": node.get("file_path"),
                    "module_name": node.get("module_name"),
                    "signal_name": node.get("signal_name"),
                    "start_line": node.get("start_line"),
                    "end_line": node.get("end_line"),
                    "similarity": anchor.mapping_confidence,
                    "distance": 0,
                    "entity_role": "evidence",
                    "matched_anchor": anchor.to_dict(include_nodes=False),
                    "path": ranked_span_paths.get(str(node.get("file_path") or ""), []),
                })
        result.sort(key=lambda item: (-float(item.get("similarity") or 0.0), str(item.get("file_path") or "")))
        return result[:50]

    @staticmethod
    def _mode_counts(ranked: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, int]]:
        counts = defaultdict(int)
        for candidate in ranked:
            counts[str(candidate.get("distance_mode") or "unknown")] += 1
        return sorted(counts.items(), key=lambda item: MODE_PRIORITY.get(item[0], 99))
