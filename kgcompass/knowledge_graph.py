from neo4j import GraphDatabase
import os
import re
try:
    from embedding import Embedding
    from utils import relative_path
    from config import (
        DECAY_FACTOR,
        VECTOR_SIMILARITY_WEIGHT,
    )
    from term import info as tinfo, warn as twarn
except Exception:  # pragma: no cover - package import path
    from .embedding import Embedding
    from .utils import relative_path
    from .config import (
        DECAY_FACTOR,
        VECTOR_SIMILARITY_WEIGHT,
    )
    from .term import info as tinfo, warn as twarn
try:
    from kg_schema import (
        HDL_ENTITY_LABELS,
        KG_SCHEMA_VERSION,
        RELATION_REVERSE_DESCRIPTIONS,
        SEMANTIC_RELATION_TYPES,
        node_projection_labels,
        schema_summary,
        semantic_relation_values,
    )
except Exception:  # pragma: no cover
    from .kg_schema import (
        HDL_ENTITY_LABELS,
        KG_SCHEMA_VERSION,
        RELATION_REVERSE_DESCRIPTIONS,
        SEMANTIC_RELATION_TYPES,
        node_projection_labels,
        schema_summary,
        semantic_relation_values,
    )
class KnowledgeGraph:
    KG_SCHEMA_VERSION = KG_SCHEMA_VERSION
    HDL_ENTITY_LABELS = set(HDL_ENTITY_LABELS)
    SEMANTIC_RELATION_TYPES = dict(SEMANTIC_RELATION_TYPES)
    GDS_RELATION_TYPES = semantic_relation_values()
    LEGACY_RELATION_TYPES = ['RELATED']
    NODE_PROJECTION_LABELS = node_projection_labels()

    def __init__(self, uri, user, password, database_name):
        database_name = database_name.replace('-', '').replace('_', '')
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.encountered_issues = set()
        self.last_localization_metadata = {}
        try:
            self.embedder = Embedding()
            tinfo("Embedding model ready")
        except Exception as e:
            twarn(f"Embedding init failed: {e}")
            raise
    
    def add_encountered_issue(self, issue_id):
        self.encountered_issues.add(issue_id)

    def close(self):
        self.driver.close()

    def schema_summary(self):
        return schema_summary()
    
    def _get_embedding(self, text):
        return self.embedder.get_embedding(text[:4000])

    def _safe_hdl_label(self, label):
        if label not in self.HDL_ENTITY_LABELS:
            raise ValueError(f"Unsupported HDL entity label: {label}")
        return label

    @classmethod
    def _semantic_relation_type(cls, relation_kind):
        normalized = (relation_kind or '').lower()
        if normalized not in cls.SEMANTIC_RELATION_TYPES:
            raise ValueError(f"Unsupported KG relation kind: {relation_kind}")
        return cls.SEMANTIC_RELATION_TYPES[normalized]

    @classmethod
    def _relationship_projection(cls, existing_relation_types=None, include_related=False):
        if existing_relation_types is None:
            existing_relation_types = set(cls.GDS_RELATION_TYPES)
        else:
            existing_relation_types = set(existing_relation_types)
        relationship_configs = []
        candidate_types = cls.GDS_RELATION_TYPES + (cls.LEGACY_RELATION_TYPES if include_related else [])
        for rel_type in candidate_types:
            if rel_type not in existing_relation_types:
                continue
            relationship_configs.append(f"""
                        {rel_type}: {{
                            type: '{rel_type}',
                            orientation: 'NATURAL',
                            properties: {{
                                weight: {{
                                    property: 'weight',
                                    defaultValue: 1.0
                                }}
                            }}
                        }}""")
        if not relationship_configs:
            raise ValueError("No supported relationship types exist in the KG for GDS projection.")
        return "{\n" + ",\n".join(relationship_configs) + "\n                    }"

    @staticmethod
    def _issue_text_boost_pattern(issue_text):
        tokens = []
        for raw in re.findall(r"\b([A-Za-z_][A-Za-z0-9_./-]{3,})\b", issue_text or ""):
            token = raw.strip().lower()
            if not token or token in {
                'module', 'signal', 'always', 'assign', 'reset', 'clock', 'state',
                'register', 'error', 'issue', 'logic', 'input', 'output', 'wire',
                'reg', 'bit', 'byte', 'test', 'case', 'description', 'condition',
                'conditions', 'function', 'affected', 'trigger', 'actual',
                'memory', 'physical', 'protection', 'scenario', 'observed',
                'result', 'results', 'problem', 'summary', 'steps', 'reproduce',
                'expected', 'baseline', 'buggy', 'fixed', 'failure', 'fails',
                'failed', 'pass', 'passes', 'passing', 'implementation',
                'configuration', 'config', 'feature', 'option', 'value', 'values',
                'field', 'fields', 'ibex', 'cva6', 'risc', 'riscv', 'caliptra',
                'opentitan', 'repo', 'should', 'would', 'could', 'there', 'their',
                'with', 'from', 'this', 'that', 'have', 'has', 'when', 'then'
            }:
                continue
            if re.fullmatch(r'[A-Z][a-z]+', raw):
                continue
            if re.fullmatch(r'[A-Z]{2,5}', raw) and '_' not in raw and not re.search(r'\d', raw):
                continue
            tokens.append(token)
        return list(dict.fromkeys(tokens))[:12]

    @classmethod
    def _merge_semantic_and_compat_relationship(
        cls,
        tx,
        match_clause,
        source_alias,
        target_alias,
        relation_kind,
        description,
        weight=1,
        params=None,
        properties=None,
        return_clause='',
    ):
        semantic_type = cls._semantic_relation_type(relation_kind)
        params = dict(params or {})
        properties = {
            key: value
            for key, value in (properties or {}).items()
            if value is not None
        }
        params.update({
            'description': description,
            'relation_kind': relation_kind,
            'semantic_type': semantic_type,
            'weight': weight,
            'relationship_properties': properties,
        })
        query = f"""
        {match_clause}
        MERGE ({source_alias})-[semantic:{semantic_type} {{description: $description, weight: $weight}}]->({target_alias})
        SET semantic.relation_kind = $relation_kind,
            semantic.semantic_type = $semantic_type,
            semantic += $relationship_properties
        MERGE ({source_alias})-[compat:RELATED {{description: $description, weight: $weight}}]->({target_alias})
        SET compat.relation_kind = $relation_kind,
            compat.semantic_type = $semantic_type,
            compat += $relationship_properties
        {return_clause}
        """
        return tx.run(query, **params)

    @staticmethod
    def _attach_context_metadata(item, group_name):
        enriched = dict(item)
        if not enriched.get('entity_role'):
            enriched['entity_role'] = 'edit_target' if group_name in {'methods', 'patchable_spans'} else (
                'evidence' if group_name in {'rtl_entities', 'classes'} else 'issue'
            )
        return enriched

    def create_method_entity(self, method_name, method_signature, file_path, start_line, end_line, source_code, doc_string='', weight=1, verilog_kind=None, semantic_summary=None, parse_source=None, parse_confidence=None, repair_role=None, timing_tags=None, timing_priority=None, timing_summary=None):
        # First check if method already exists
        with self.driver.session() as session:
            exists_query = """
            MATCH (m:Method {name: $name, signature: $signature, file_path: $file_path})
            RETURN count(m) > 0 as exists
            """
            exists = session.run(exists_query, 
                               name=method_name, 
                               signature=method_signature, 
                               file_path=file_path).single()['exists']
            
            if not exists:
                # If method doesn't exist, calculate embedding and create new method
                repair_role, timing_tags, timing_priority, timing_summary = self._normalize_timing_metadata(
                    repair_role,
                    timing_tags,
                    timing_priority,
                    timing_summary,
                )
                embedding_source = semantic_summary or source_code
                text_for_embedding = f"{method_name}\\n{doc_string or ''}\\n{embedding_source}"
                embedding = self._get_embedding(text_for_embedding)
                
                session.execute_write(self._create_and_link, 
                                    method_name, 
                                    method_signature, 
                                    file_path, 
                                    start_line, 
                                    end_line, 
                                     source_code, 
                                      doc_string or '',  # Ensure doc_string is not None
                                      verilog_kind,
                                      semantic_summary,
                                      parse_source,
                                      parse_confidence,
                                      repair_role,
                                      timing_tags,
                                      timing_priority,
                                      timing_summary,
                                      embedding,
                                      weight)
                # Create method-file relationship
                session.execute_write(self._link_method_to_file, 
                                    method_name, 
                                    method_signature, 
                                    file_path,
                                    weight)

    @staticmethod
    def _create_and_link(tx, method_name, method_signature, file_path, start_line, end_line, source_code, doc_string, verilog_kind, semantic_summary, parse_source, parse_confidence, repair_role, timing_tags, timing_priority, timing_summary, embedding, weight):
        query = (
            "MERGE (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path, "
            "start_line: $start_line, end_line: $end_line, source_code: $source_code, doc_string: $doc_string, embedding: $embedding}) "
            "SET m.verilog_kind = $verilog_kind, "
            "    m.semantic_summary = $semantic_summary, "
            "    m.parse_source = $parse_source, "
            "    m.parse_confidence = $parse_confidence, "
            "    m.repair_role = $repair_role, "
            "    m.timing_tags = $timing_tags, "
            "    m.timing_priority = $timing_priority, "
            "    m.timing_summary = $timing_summary "
        )
        tx.run(query, method_name=method_name, method_signature=method_signature, 
               file_path=file_path, start_line=start_line, end_line=end_line, source_code=source_code, doc_string=doc_string or '',
               verilog_kind=verilog_kind, semantic_summary=semantic_summary or source_code,
               parse_source=parse_source, parse_confidence=parse_confidence,
               repair_role=repair_role or '',
               timing_tags=timing_tags or [],
               timing_priority=timing_priority,
               timing_summary=timing_summary or '',
               embedding=embedding, weight=weight) # Ensure doc_string is not None

    def clear_graph(self):
        with self.driver.session() as session:
            # Delete all nodes and relationships in batches
            # to avoid overwhelming Neo4j transaction memory on large graphs
            while True:
                result = session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted")
                summary = result.single()
                if summary is None or summary["deleted"] == 0:
                    break
            
            try:
                # Delete all indexes
                for index in session.run("SHOW INDEXES"):
                    session.run(f"DROP INDEX {index['name']}")
            except Exception:
                pass
            
            try:
                # Delete all constraints
                for constraint in session.run("SHOW CONSTRAINTS"):
                    session.run(f"DROP CONSTRAINT {constraint['name']}")
            except Exception:
                pass

    def create_issue(self, issue_id, title, content=None):
        with self.driver.session() as session:
            # First check if issue already exists
            exists_query = """
            MATCH (i:Issue {id: $id})
            RETURN count(i) > 0 as exists
            """
            exists = session.run(exists_query, id=issue_id).single()['exists']
            
            if not exists:
                # If issue doesn't exist, calculate embedding and create new issue
                text_for_embedding = f"{title}\n{content}"
                embedding = self._get_embedding(text_for_embedding)
                session.execute_write(self._create_issue, issue_id, title, content, embedding)

    @staticmethod
    def _create_issue(tx, issue_id, title, content=None, embedding=None):
        query = (
            "MERGE (i:Issue {id: $issue_id}) "
            "SET i.title = $title, "
            "    i.content = $content, "
            "    i.name = $name, "
            "    i.embedding = $embedding "
        )
        tx.run(query, issue_id=issue_id, title=title, content=content, name=f"Issue:{issue_id}", embedding=embedding)

    def create_file_entity(self, file_path):
        """
        Create code file entity
        
        Args:
            file_path (str): File path
        """
        with self.driver.session() as session:
            session.execute_write(self._create_file, file_path)

    @staticmethod
    def _create_file(tx, file_path):
        query = (
            "MERGE (f:File {path: $file_path}) "
            "SET f.name = $name"
        )
        tx.run(query, file_path=file_path, name=relative_path(file_path))

    def create_directory_structure(self, base_path, code_analyzer, process_detail=False, weight=1):
        """
        Create directory structure, including directories and files and their relationships
        
        Args:
            base_path (str): Base path
        """
        with self.driver.session() as session:
            file_paths = session.execute_write(self._create_directory_structure, base_path)
            if process_detail and file_paths:
                for file_path in file_paths:
                    code_analyzer._build_file_class_methods(file_path)

    @staticmethod
    def _create_directory_structure(tx, base_path, weight=1):
        for root, dirs, files in os.walk(base_path):
            # Create current directory
            abs_dir_path = root.replace('\\', '/')
            rel_dir_path = relative_path(abs_dir_path)
            if os.path.basename(root).startswith('.'):
                continue
            # Create current directory node
            query = (
                "MERGE (d:Directory {path: $dir_path}) "
                "SET d.name = $name"
            )
            tx.run(query, 
                dir_path=rel_dir_path or '/', 
                name=os.path.basename(root) or '/'
            )
            file_paths = []
            # If not root directory, create relationship with parent directory
            if rel_dir_path:
                parent_dir_abs = os.path.dirname(abs_dir_path)
                parent_dir_path = relative_path(parent_dir_abs)
                match_clause = (
                    "MATCH (parent:Directory {path: $parent_path}) "
                    "MATCH (child:Directory {path: $child_path}) "
                )
                params = {
                    'parent_path': parent_dir_path or '/',
                    'child_path': rel_dir_path,
                }
                KnowledgeGraph._merge_semantic_and_compat_relationship(
                    tx, match_clause, 'parent', 'child', 'contains',
                    'contains directory', weight, params
                )
                KnowledgeGraph._merge_semantic_and_compat_relationship(
                    tx, match_clause, 'child', 'parent', 'contains',
                    'contained in directory', weight, params
                )
            
            source_extensions = ('.py', '.cpp', '.cc', '.cxx', '.java', '.h', '.hpp', '.v', '.sv', '.vh', '.svh')
            source_files = [f for f in files if f.endswith(source_extensions)]
            total_files = len(source_files)
            for idx, file in enumerate(source_files, 1):
                file_abs_path = os.path.join(abs_dir_path, file)
                rel_file_path = relative_path(file_abs_path)
                
                # Create file node
                query = (
                    "MERGE (f:File {path: $file_path}) "
                    "SET f.name = $name"
                )
                tx.run(query, 
                    file_path=rel_file_path,
                    name=rel_file_path
                )
                
                # Create directory-file relationship
                match_clause = (
                    "MATCH (d:Directory {path: $dir_path}) "
                    "MATCH (f:File {path: $file_path}) "
                )
                params = {
                    'dir_path': rel_dir_path or '/',
                    'file_path': rel_file_path,
                }
                file_paths.append(file_abs_path)
                KnowledgeGraph._merge_semantic_and_compat_relationship(
                    tx, match_clause, 'd', 'f', 'contains',
                    'contains file', weight, params
                )
                KnowledgeGraph._merge_semantic_and_compat_relationship(
                    tx, match_clause, 'f', 'd', 'contains',
                    'contained in directory', weight, params
                )
        return file_paths

    def create_class_entity(self, class_name, file_path, start_line, end_line, source_code, doc_string="", weight=1, verilog_kind=None, semantic_summary=None, parse_source=None, parse_confidence=None, repair_role=None, timing_tags=None, timing_priority=None, timing_summary=None):
        with self.driver.session() as session:
            # First check if class already exists
            exists_query = """
            MATCH (c:Class {name: $name, file_path: $file_path})
            RETURN count(c) > 0 as exists
            """
            exists = session.run(exists_query, 
                               name=class_name, 
                               file_path=file_path).single()['exists']
            
            if not exists:
                # If class doesn't exist, calculate embedding and create new class
                repair_role, timing_tags, timing_priority, timing_summary = self._normalize_timing_metadata(
                    repair_role,
                    timing_tags,
                    timing_priority,
                    timing_summary,
                )
                embedding_source = semantic_summary or source_code
                text_for_embedding = f"{class_name}\\n{doc_string or ''}\\n{embedding_source}"
                text_for_embedding = text_for_embedding[:8000]
                embedding = self._get_embedding(text_for_embedding)
                
                session.execute_write(self._create_class, 
                                    class_name, 
                                    file_path, 
                                    start_line, 
                                    end_line, 
                                     source_code, 
                                      doc_string or '',  # Ensure doc_string is not None
                                      verilog_kind,
                                      semantic_summary,
                                      parse_source,
                                      parse_confidence,
                                      repair_role,
                                      timing_tags,
                                      timing_priority,
                                      timing_summary,
                                    embedding,
                                    weight)

    def _normalize_timing_metadata(self, repair_role=None, timing_tags=None, timing_priority=None, timing_summary=None):
        normalized_role = str(repair_role or "").strip()
        normalized_tags = timing_tags or []
        if isinstance(normalized_tags, str):
            normalized_tags = [tag.strip() for tag in normalized_tags.split(",") if tag.strip()]
        normalized_summary = timing_summary or ""
        return normalized_role, normalized_tags, timing_priority, normalized_summary

    @staticmethod
    def _create_class(tx, class_name, file_path, start_line, end_line, source_code, doc_string="", verilog_kind=None, semantic_summary=None, parse_source=None, parse_confidence=None, repair_role=None, timing_tags=None, timing_priority=None, timing_summary=None, embedding=None, weight=1):
        # Create class node
        query = (
            "MERGE (c:Class {name: $class_name, file_path: $file_path, "
            "start_line: $start_line, end_line: $end_line, source_code: $source_code, "
            "doc_string: $doc_string, embedding: $embedding}) "
            "SET c.short_name = $short_name, "
            "    c.verilog_kind = $verilog_kind, "
            "    c.semantic_summary = $semantic_summary, "
            "    c.parse_source = $parse_source, "
            "    c.parse_confidence = $parse_confidence, "
            "    c.repair_role = $repair_role, "
            "    c.timing_tags = $timing_tags, "
            "    c.timing_priority = $timing_priority, "
            "    c.timing_summary = $timing_summary"
        )
        tx.run(query, 
            class_name=class_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            doc_string=doc_string or '',  # Ensure doc_string is not None
            embedding=embedding,
            short_name=class_name.split('.')[-1],
            verilog_kind=verilog_kind,
            semantic_summary=semantic_summary or source_code,
            parse_source=parse_source,
            parse_confidence=parse_confidence,
            repair_role=repair_role or '',
            timing_tags=timing_tags or [],
            timing_priority=timing_priority,
            timing_summary=timing_summary or '',
            weight=weight
        )
        
        # Create relationship with file
        match_clause = (
            "MATCH (f:File {path: $file_path}) "
            "MATCH (c:Class {name: $class_name, file_path: $file_path}) "
        )
        params = {'file_path': file_path, 'class_name': class_name}
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'f', 'c', 'contains',
            'contains class', weight, params
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'c', 'f', 'contains',
            'contained in file', weight, params
        )

    def link_class_to_method(self, class_name, file_path, method_name, method_signature, weight=1):
        """
        Establish association between class and method
        
        Args:
            class_name (str): Class name
            file_path (str): File path
            method_name (str): Method name
            method_signature (str): Method signature
        """
        with self.driver.session() as session:
            session.execute_write(self._link_class_to_method, 
                                class_name, file_path, method_name, method_signature, weight)

    @staticmethod
    def _link_class_to_method(tx, class_name, file_path, method_name, method_signature, weight=1):
        match_clause = (
            "MATCH (c:Class {name: $class_name, file_path: $file_path}) "
            "MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path}) "
        )
        params = {
            'class_name': class_name,
            'file_path': file_path,
            'method_name': method_name,
            'method_signature': method_signature,
        }
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'c', 'm', 'contains',
            'contains method', weight, params
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'm', 'c', 'contains',
            'contained in class', weight, params
        )

    def link_issues(self, source_id, target_id, weight=1):
        """
        Establish relationship between two issues/PRs
        
        Args:
            source_id (str): Source issue/PR ID
            target_id (str): Target issue/PR ID
        """
        with self.driver.session() as session:
            match_clause = """
            MATCH (source:Issue {id: $source_id})
            MATCH (target:Issue {id: $target_id})
            """
            params = {
                'source_id': source_id,
                'target_id': target_id,
            }
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'source', 'target', 'mentions',
                'points to issue', weight, params
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'target', 'source', 'mentions',
                'referenced by issue', weight, params
            )

    def create_issue_entity_by_github_issue(self, issue):
        self.create_issue_entity(
            str(issue.number),
            issue.title,
            issue.full_body or "",
            issue.created_at.timestamp(),
            issue.state,
            issue.pull_request is not None,
            f"{'pr' if issue.pull_request else 'issue'}#{issue.number}"
        )

    def create_issue_entity(self, issue_id, title, content, created_at, state, is_pr, name):
        """
        Create unified issue entity (including PRs)
        
        Args:
            issue_id (str): Issue/PR ID
            title (str): Title
            content (str): Content
            created_at (float): Creation timestamp
            state (str): State
            is_pr (bool): Whether it's a PR
            name (str): Entity name
        """
        with self.driver.session() as session:
            # First check if issue already exists
            exists_query = """
            MATCH (i:Issue {id: $id})
            RETURN count(i) > 0 as exists
            """
            exists = session.run(exists_query, id=issue_id).single()['exists']
            
            if not exists:
                # If issue doesn't exist, calculate embedding and create new issue
                text_for_embedding = f"{title}\n{content}"
                embedding = self._get_embedding(text_for_embedding)
                
                session.execute_write(self._create_issue_entity, 
                                    issue_id, title, content, 
                                    created_at, state, is_pr, name, embedding)

    @staticmethod
    def _create_issue_entity(tx, issue_id, title, content, created_at, state, is_pr, name, embedding):
        # Create or update entity
        query = """
        MERGE (i:Issue {id: $issue_id})
        ON CREATE SET 
            i.title = $title,
            i.content = $content,
            i.created_at = $created_at,
            i.state = $state,
            i.is_pr = $is_pr,
            i.type = $type,
            i.name = $name,
            i.embedding = $embedding
        ON MATCH SET 
            i.title = $title,
            i.content = $content,
            i.is_pr = $is_pr,
            i.type = $type,
            i.name = $name
        """
        tx.run(query, 
               issue_id=issue_id,
               title=title,
               content=content,
               created_at=created_at,
               state=state,
               is_pr=is_pr,
               type='issue',
               name=name,
               embedding=embedding)

    def get_all_methods(self, top_k):
        """
        Get the 200 most relevant method entities for the given text
        
        Args:
            root_text (str): Base text for calculating similarity
            
        Returns:
            list: List of methods sorted by similarity
        """
        with self.driver.session() as session:
            query = """
            MATCH (root:Issue {id: 'root'})
            WHERE root.embedding IS NOT NULL
            WITH DISTINCT root, root.embedding as root_embedding, 
                 root.title + ' ' + root.content as root_text
            
            MATCH (m:Method)
            WHERE m.embedding IS NOT NULL
            AND (NOT m.name CONTAINS 'test' OR m.name CONTAINS 'pytest')
            
            WITH m, root_embedding, root_text,
                 coalesce(m.semantic_summary, m.source_code, '') as m_text
            WITH m, root_embedding, root_text,
                 (gds.similarity.cosine(root_embedding, m.embedding) * $VECTOR_SIMILARITY_WEIGHT +
                  apoc.text.levenshteinSimilarity(root_text, m_text) * (1 - $VECTOR_SIMILARITY_WEIGHT)) as similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            
            RETURN m.name as name,
                   m.file_path as file_path,
                   m.signature as signature,
                   m.source_code as source_code,
                   m.doc_string as doc_string,
                   m.title as title,
                   similarity
            """
            
            result = session.run(query, top_k=top_k, VECTOR_SIMILARITY_WEIGHT=VECTOR_SIMILARITY_WEIGHT)
            methods = [dict(record) for record in result]
            return methods

    def link_issue_to_file(self, issue_id, file_path, weight=1):
        with self.driver.session() as session:
            session.execute_write(self._link_issue_to_file, issue_id, file_path, weight)

    def search_file_by_path(self, file_path):
        parts = file_path.replace('\\', '/').split('/')
        if '~' in parts:
            parts = parts[parts.index('~')+1:]
        if len(parts) > 3:
            parts = parts[-4:]
        target_filename = parts[-1]
        query = """
        MATCH (f:File)
        WITH f, $file_parts as parts, $target_filename as target
        WITH f, parts, target, f.path as path,
             last(split(f.path, '/')) as file_name,
             split(f.path, '/') as path_parts
        WITH f, parts, target, path, file_name, path_parts,
             [p in parts WHERE p IN path_parts] as matched_parts,
             CASE 
                WHEN file_name = target THEN 3
                WHEN file_name STARTS WITH 'test_' THEN 0
                WHEN file_name CONTAINS replace(target, '_', '') THEN 1
                ELSE 0
             END as filename_match_score,
             apoc.coll.indexOf(path_parts, last(parts[..-1])) as dir_match
        WHERE size(matched_parts) >= 1 
        WITH f, matched_parts, filename_match_score,
             CASE WHEN dir_match >= 0 THEN 2 ELSE 0 END as same_dir,
             reduce(s = 0, i IN range(0, size(matched_parts)-1) |
                s + CASE WHEN apoc.coll.indexOf(path_parts, matched_parts[i]) < apoc.coll.indexOf(path_parts, matched_parts[i+1])
                    THEN 1 ELSE 0 END
             ) as consecutive_count,
             size(matched_parts) as match_count
        RETURN {
            file: f,
            match_count: match_count,
            consecutive_count: consecutive_count,
            score: same_dir * 1000 + filename_match_score * 100 + match_count * 10 + consecutive_count
        } as result
        ORDER BY result.score DESC
        LIMIT 3
        """
        with self.driver.session() as session:
            results = session.run(query, file_parts=parts, target_filename=target_filename)
            matches = []
            for record in results:
                matches.append({
                    'file': record['result']['file'],
                    'score': record['result']['score']
                })
            return matches if matches else None

    @staticmethod
    def _link_issue_to_file(tx, issue_id, file_path, weight=1):
        match_clause = (
            "MERGE (f:File {path: $file_path}) "
            "WITH f "
            "MATCH (i:Issue {id: $issue_id}) "
        )
        params = {'file_path': file_path, 'issue_id': issue_id}
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'i', 'f', 'mentions',
            'points to file', weight, params
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'f', 'i', 'mentions',
            'referenced by issue', weight, params
        )

    def create_commit_entity(self, commit_id, commit_message):
        query = """
        MERGE (c:Commit {id: $commit_id})
        SET c.message = $message
        """
        with self.driver.session() as session:
            session.run(query, commit_id=commit_id, message=commit_message)

    def link_method_to_commit(self, method_name, method_signature, file_path, commit_id, commit_message):
        match_clause = """
        MATCH (m:Method {name: $method_name, signature: $signature, file_path: $file_path})
        MATCH (c:Commit {id: $commit_id})
        """
        params = {
            'method_name': method_name,
            'signature': method_signature,
            'file_path': file_path,
            'commit_id': commit_id,
        }
        with self.driver.session() as session:
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'm', 'c', 'modifies',
                'modified by commit', 1, params, {'message': commit_message}
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'c', 'm', 'modifies',
                'modified method', 1, params, {'message': commit_message}
            )

    def link_method_to_issue(self, method_name, method_signature, file_path, issue_id, weight=1):
        with self.driver.session() as session:
            session.execute_write(self._link_method_to_issue, 
                                method_name, method_signature, file_path, issue_id, weight)

    @staticmethod
    def _link_method_to_issue(tx, method_name, method_signature, file_path, issue_id, weight=1):
        match_clause = (
            "MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path}) "
            "MATCH (i:Issue {id: $issue_id}) "
        )
        params = {
            'method_name': method_name,
            'method_signature': method_signature,
            'file_path': file_path,
            'issue_id': issue_id,
        }
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'm', 'i', 'mentions',
            'referenced by issue', weight, params
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'i', 'm', 'mentions',
            'points to method', weight, params
        )

    def link_class_to_issue(self, class_name, file_path, issue_id, weight=1):
        with self.driver.session() as session:
            session.execute_write(self._link_class_to_issue, 
                                class_name, file_path, issue_id, weight)

    @staticmethod
    def _link_class_to_issue(tx, class_name, file_path, issue_id, weight=1):
        match_clause = (
            "MATCH (c:Class {name: $class_name, file_path: $file_path}) "
            "MATCH (i:Issue {id: $issue_id}) "
        )
        params = {
            'class_name': class_name,
            'file_path': file_path,
            'issue_id': issue_id,
        }
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'c', 'i', 'mentions',
            'referenced by issue', weight, params
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'i', 'c', 'mentions',
            'points to class', weight, params
        )

    @staticmethod
    def _link_method_to_file(tx, method_name, method_signature, file_path, weight=1):
        match_clause = (
            "MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path}) "
            "MATCH (f:File {path: $file_path}) "
        )
        params = {
            'method_name': method_name,
            'method_signature': method_signature,
            'file_path': file_path,
        }
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'f', 'm', 'contains',
            'contains method', weight, params
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'm', 'f', 'contains',
            'contained in file', weight, params
        )

    def link_method_calls(self, caller_name, caller_signature, 
                         callee_name, callee_signature):
        with self.driver.session() as session:
            session.execute_write(self._link_method_calls, 
                                caller_name, caller_signature,
                                callee_name, callee_signature)

    @staticmethod
    def _link_method_calls(tx, caller_name, caller_signature,
                          callee_name, callee_signature):
        match_clause = (
            "MATCH (caller:Method {name: $caller_name, signature: $caller_signature}) "
            "MATCH (callee:Method {name: $callee_name, signature: $callee_signature}) "
        )
        params = {
            'caller_name': caller_name,
            'caller_signature': caller_signature,
            'callee_name': callee_name,
            'callee_signature': callee_signature,
        }
        result = KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'caller', 'callee', 'calls',
            'calls method', 1, params,
            return_clause='RETURN caller.name as caller, callee.name as callee'
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'callee', 'caller', 'calls',
            'called by method', 1, params
        )
        
        
        record = result.single()
        if record:
            pass

    def create_hdl_entity(
        self,
        label,
        name,
        file_path,
        module_name='',
        rtl_kind='',
        signal_name='',
        direction='',
        width='',
        declaration='',
        start_line=None,
        end_line=None,
        start_column=None,
        end_column=None,
        source_code='',
        semantic_summary='',
        parse_source=None,
        parse_confidence=None,
        repair_role=None,
        timing_tags=None,
        timing_priority=None,
        timing_summary=None,
        span_kind='',
        patch_granularity='',
        parent_span='',
        reads_signals=None,
        writes_signals=None,
        drives_signals=None,
        connects_signals=None,
        guards_signals=None,
        port_bindings=None,
        projection_source='',
        weight=1,
    ):
        label = self._safe_hdl_label(label)
        embedding_source = semantic_summary or source_code or declaration or name
        embedding = self._get_embedding(embedding_source)
        repair_role, timing_tags, timing_priority, timing_summary = self._normalize_timing_metadata(
            repair_role,
            timing_tags,
            timing_priority,
            timing_summary,
        )
        with self.driver.session() as session:
            session.execute_write(
                self._create_hdl_entity,
                label,
                name,
                file_path,
                module_name or '',
                rtl_kind or '',
                signal_name or '',
                direction or '',
                width or '',
                declaration or '',
                start_line,
                end_line,
                start_column,
                end_column,
                source_code or declaration or '',
                semantic_summary or embedding_source,
                parse_source,
                parse_confidence,
                repair_role,
                timing_tags,
                timing_priority,
                timing_summary,
                span_kind or '',
                patch_granularity or '',
                parent_span or '',
                list(reads_signals or []),
                list(writes_signals or []),
                list(drives_signals or []),
                list(connects_signals or []),
                list(guards_signals or []),
                list(port_bindings or []),
                projection_source or '',
                embedding,
                weight,
            )

    @staticmethod
    def _create_hdl_entity(
        tx,
        label,
        name,
        file_path,
        module_name,
        rtl_kind,
        signal_name,
        direction,
        width,
        declaration,
        start_line,
        end_line,
        start_column,
        end_column,
        source_code,
        semantic_summary,
        parse_source,
        parse_confidence,
        repair_role,
        timing_tags,
        timing_priority,
        timing_summary,
        span_kind,
        patch_granularity,
        parent_span,
        reads_signals,
        writes_signals,
        drives_signals,
        connects_signals,
        guards_signals,
        port_bindings,
        projection_source,
        embedding,
        weight,
    ):
        file_relation_map = {
            'Macro': ('defines macro', 'defines', 'defined in file'),
            'Include': ('includes header', 'includes', 'included from file'),
            'Import': ('imports package symbol', 'imports', 'imported by file'),
            'Typedef': ('declares type', 'declares', 'declared in file'),
        }
        file_description, file_relation_kind, reverse_file_description = file_relation_map.get(
            label,
            (f"contains {label.lower()}", 'contains', 'contained in file'),
        )
        query = f"""
        MERGE (n:{label} {{name: $name, file_path: $file_path, module_name: $module_name}})
        SET n.rtl_kind = $rtl_kind,
            n.verilog_kind = $rtl_kind,
            n.signal_name = $signal_name,
            n.direction = $direction,
            n.width = $width,
            n.declaration = $declaration,
            n.start_line = $start_line,
            n.end_line = $end_line,
            n.start_column = $start_column,
            n.end_column = $end_column,
            n.source_code = $source_code,
            n.semantic_summary = $semantic_summary,
            n.parse_source = $parse_source,
            n.parse_confidence = $parse_confidence,
            n.repair_role = $repair_role,
            n.timing_tags = $timing_tags,
            n.timing_priority = $timing_priority,
            n.timing_summary = $timing_summary,
            n.span_kind = $span_kind,
            n.patch_granularity = $patch_granularity,
            n.parent_span = $parent_span,
            n.reads_signals = $reads_signals,
            n.writes_signals = $writes_signals,
            n.drives_signals = $drives_signals,
            n.connects_signals = $connects_signals,
            n.guards_signals = $guards_signals,
            n.port_bindings = $port_bindings,
            n.projection_source = $projection_source,
            n.embedding = $embedding
        """
        tx.run(
            query,
            name=name,
            file_path=file_path,
            module_name=module_name,
            rtl_kind=rtl_kind,
            signal_name=signal_name,
            direction=direction,
            width=width,
            declaration=declaration,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            source_code=source_code,
            semantic_summary=semantic_summary,
            parse_source=parse_source,
            parse_confidence=parse_confidence,
            repair_role=repair_role or '',
            timing_tags=timing_tags or [],
            timing_priority=timing_priority,
            timing_summary=timing_summary or '',
            span_kind=span_kind,
            patch_granularity=patch_granularity,
            parent_span=parent_span,
            reads_signals=reads_signals,
            writes_signals=writes_signals,
            drives_signals=drives_signals,
            connects_signals=connects_signals,
            guards_signals=guards_signals,
            port_bindings=port_bindings,
            projection_source=projection_source,
            embedding=embedding,
            file_description=file_description,
            file_relation_kind=file_relation_kind,
            weight=weight,
        )
        match_clause = f"""
        MATCH (f:File {{path: $file_path}})
        MATCH (n:{label} {{name: $name, file_path: $file_path, module_name: $module_name}})
        """
        params = {
            'file_path': file_path,
            'name': name,
            'module_name': module_name,
        }
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'f', 'n', file_relation_kind,
            file_description, weight, params
        )
        KnowledgeGraph._merge_semantic_and_compat_relationship(
            tx, match_clause, 'n', 'f', file_relation_kind,
            reverse_file_description, weight, params
        )

    def link_class_to_hdl_entity(
        self,
        class_name,
        class_file_path,
        entity_label,
        entity_name,
        entity_file_path,
        entity_module_name='',
        description='contains rtl entity',
        reverse_description='contained in module',
        relation_kind='contains',
        parse_source=None,
        parse_confidence=None,
        weight=1,
    ):
        entity_label = self._safe_hdl_label(entity_label)
        match_clause = f"""
        MATCH (c:Class {{name: $class_name, file_path: $class_file_path}})
        MATCH (e:{entity_label} {{name: $entity_name, file_path: $entity_file_path, module_name: $entity_module_name}})
        """
        params = {
            'class_name': class_name,
            'class_file_path': class_file_path,
            'entity_name': entity_name,
            'entity_file_path': entity_file_path,
            'entity_module_name': entity_module_name or '',
        }
        with self.driver.session() as session:
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'c', 'e', relation_kind,
                description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'e', 'c', relation_kind,
                reverse_description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )

    def link_method_to_hdl_entity(
        self,
        method_name,
        method_signature,
        method_file_path,
        entity_label,
        entity_name,
        entity_file_path,
        entity_module_name='',
        description='uses rtl entity',
        reverse_description='used by method',
        relation_kind='uses',
        parse_source=None,
        parse_confidence=None,
        weight=1,
    ):
        entity_label = self._safe_hdl_label(entity_label)
        match_clause = f"""
        MATCH (m:Method {{name: $method_name, signature: $method_signature, file_path: $method_file_path}})
        MATCH (e:{entity_label} {{name: $entity_name, file_path: $entity_file_path, module_name: $entity_module_name}})
        """
        params = {
            'method_name': method_name,
            'method_signature': method_signature,
            'method_file_path': method_file_path,
            'entity_name': entity_name,
            'entity_file_path': entity_file_path,
            'entity_module_name': entity_module_name or '',
        }
        with self.driver.session() as session:
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'm', 'e', relation_kind,
                description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'e', 'm', relation_kind,
                reverse_description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )

    def link_hdl_entities(
        self,
        source_label,
        source_name,
        source_file_path,
        source_module_name,
        target_label,
        target_name,
        target_file_path,
        target_module_name,
        description='feeds signal',
        reverse_description='fed by signal',
        relation_kind='feeds',
        parse_source=None,
        parse_confidence=None,
        weight=1,
    ):
        source_label = self._safe_hdl_label(source_label)
        target_label = self._safe_hdl_label(target_label)
        match_clause = f"""
        MATCH (s:{source_label} {{name: $source_name, file_path: $source_file_path, module_name: $source_module_name}})
        MATCH (t:{target_label} {{name: $target_name, file_path: $target_file_path, module_name: $target_module_name}})
        """
        params = {
            'source_name': source_name,
            'source_file_path': source_file_path,
            'source_module_name': source_module_name or '',
            'target_name': target_name,
            'target_file_path': target_file_path,
            'target_module_name': target_module_name or '',
        }
        with self.driver.session() as session:
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 's', 't', relation_kind,
                description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 't', 's', relation_kind,
                reverse_description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )

    def link_method_to_class(
        self,
        method_name,
        method_signature,
        method_file_path,
        class_name,
        description='instantiates module',
        reverse_description='instantiated by method',
        relation_kind='instantiates',
        parse_source=None,
        parse_confidence=None,
        weight=1,
    ):
        match_clause = """
        MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $method_file_path})
        MATCH (c:Class {name: $class_name})
        """
        params = {
            'method_name': method_name,
            'method_signature': method_signature,
            'method_file_path': method_file_path,
            'class_name': class_name,
        }
        with self.driver.session() as session:
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'm', 'c', relation_kind,
                description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'c', 'm', relation_kind,
                reverse_description, weight, params,
                {'parse_source': parse_source, 'parse_confidence': parse_confidence}
            )

    def get_method_by_name(self, method_name):
        with self.driver.session() as session:
            query = """
            MATCH (m:Method)
            WHERE m.name = $method_name
            RETURN m.name as name,
                    m.signature as signature,
                    m.file_path as file_path,
                    m.start_line as start_line,
                    m.end_line as end_line,
                    m.source_code as source_code,
                    m.doc_string as doc_string
            """
            result = session.run(query, method_name=method_name)
            return [{
                'name': record['name'],
                'signature': record['signature'],
                'file_path': record['file_path'], 
                'start_line': record['start_line'],
                'end_line': record['end_line'],
                'source_code': record['source_code'],
                'doc_string': record['doc_string']
            } for record in result]

    def get_all_similarities_to_root(self, max_hops=2, limit=None, sort=False, issue_text="", issue_mentioned_files=None):
        localization_mode = os.getenv("KGVERI_LOCALIZATION_MODE", "rtl_causal_b2").strip().lower()
        if localization_mode not in {"kgcompass_b0", "b0", "legacy"}:
            try:
                from rtl_causal_localization import RTLCausalLocalizer
            except Exception:  # pragma: no cover
                from .rtl_causal_localization import RTLCausalLocalizer
            localizer = RTLCausalLocalizer(
                self.driver,
                vector_weight=VECTOR_SIMILARITY_WEIGHT,
                decay_factor=DECAY_FACTOR,
                mode=localization_mode,
            )
            results, metadata = localizer.localize(
                issue_text=issue_text,
                issue_mentioned_files=issue_mentioned_files or [],
                limit=limit or 50,
                max_cost=float(max_hops),
                capture_ablation=os.getenv("KGVERI_CAPTURE_ABLATION", "0") == "1",
            )
            self.last_localization_metadata = metadata
            return results

        self.last_localization_metadata = {
            "ranking_version": "kgcompass-b0",
            "localization_mode": localization_mode,
            "max_root_cost": float(max_hops),
        }
        limit = limit or 500
        max_target_nodes = min(1000, limit * 2)
        issue_mentioned_files = [
            str(item or "").replace('\\', '/').strip()
            for item in (issue_mentioned_files or [])
            if str(item or "").strip()
        ]
        issue_tokens = self._issue_text_boost_pattern(issue_text)
        
        with self.driver.session() as session:
            try:
                # 1. Ensure old graph projection is deleted
                session.run("CALL gds.graph.drop('graph', false)")
                
                # 2. Create a canonical semantic graph projection. RELATED is kept in
                # Neo4j as a compatibility layer, but only projected as a fallback.
                existing_relation_types = session.run(
                    "MATCH ()-[r]->() RETURN collect(DISTINCT type(r)) AS types"
                ).single()['types']
                include_related_projection = os.getenv('KGCOMPASS_GDS_INCLUDE_RELATED', '').lower() in {'1', 'true', 'yes'}
                try:
                    relationship_projection = self._relationship_projection(existing_relation_types, include_related=include_related_projection)
                except ValueError:
                    relationship_projection = self._relationship_projection(existing_relation_types, include_related=True)
                projection_labels = repr(self.NODE_PROJECTION_LABELS)
                hdl_entity_labels = sorted(self.HDL_ENTITY_LABELS)
                session.run(f"""
                CALL gds.graph.project(
                    'graph',
                    {projection_labels},
                    {relationship_projection}
                )
                """)

                # 3. Execute optimized query
                method_query = """
                MATCH (root:Issue {id: 'root'})
                WHERE root.embedding IS NOT NULL
                WITH root, root.embedding as root_embedding,
                    root.title + ' ' + root.content as root_text

                MATCH (m)
                WHERE (m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels)
                       OR (m:Issue AND m.id <> 'root')) 
                AND m.embedding IS NOT NULL

                CALL gds.shortestPath.dijkstra.stream('graph', {
                    sourceNode: root,
                    targetNode: m,
                    relationshipWeightProperty: 'weight'
                })
                YIELD nodeIds, totalCost

                WITH nodeIds, totalCost, root_embedding, root_text,
                    gds.util.asNode(nodeIds[-1]) as m
                WHERE (any(label IN labels(m) WHERE label IN $hdl_entity_labels)
                       OR (m:Class AND NOT EXISTS((m)-[:CONTAINS|RELATED]->(:Method))) OR m:Issue)
                  AND totalCost <= $max_hops
                  AND (m:Issue AND m.id <> 'root' OR NOT m:Issue)

                WITH m, nodeIds, totalCost, root_embedding, root_text,
                    [i IN range(0, size(nodeIds)-2) | 
                    coalesce(
                        [
                            (start)-[rel]-(end) 
                            WHERE id(start) = nodeIds[i] AND id(end) = nodeIds[i+1]
                              AND type(rel) <> 'RELATED' |
                            {
                                start_node: CASE 
                                    WHEN start:Commit THEN 'Commit#' + start.id
                                    WHEN start:Issue THEN start.name
                                    ELSE start.name
                                END,
                                end_node: CASE 
                                    WHEN end:Commit THEN 'Commit#' + end.id
                                    WHEN end:Issue THEN end.name
                                    ELSE end.name
                                END,
                                type: type(rel),
                                relation_kind: coalesce(rel.relation_kind, toLower(type(rel))),
                                description: CASE
                                    WHEN id(start) = id(startNode(rel)) THEN rel.description
                                    ELSE CASE
                                        WHEN rel.description = 'contains method' THEN 'contained in method'
                                        WHEN rel.description = 'contained in method' THEN 'contains method'
                                        WHEN rel.description = 'contains class' THEN 'contained in class'
                                        WHEN rel.description = 'contained in class' THEN 'contains class'
                                        WHEN rel.description = 'contains file' THEN 'contained in file'
                                        WHEN rel.description = 'contained in file' THEN 'contains file'
                                        WHEN rel.description = 'points to issue' THEN 'referenced by issue'
                                        WHEN rel.description = 'referenced by issue' THEN 'points to issue'
                                        WHEN rel.description = 'calls method' THEN 'called by method'
                                        WHEN rel.description = 'called by method' THEN 'calls method'
                                        WHEN rel.description = 'reads signal' THEN 'read by method'
                                        WHEN rel.description = 'read by method' THEN 'reads signal'
                                        WHEN rel.description = 'writes signal' THEN 'written by method'
                                        WHEN rel.description = 'written by method' THEN 'writes signal'
                                        WHEN rel.description = 'drives signal' THEN 'driven by method'
                                        WHEN rel.description = 'driven by method' THEN 'drives signal'
                                        WHEN rel.description = 'feeds signal' THEN 'fed by signal'
                                        WHEN rel.description = 'fed by signal' THEN 'feeds signal'
                                        WHEN rel.description = 'connects signal' THEN 'connected by instance'
                                        WHEN rel.description = 'connected by instance' THEN 'connects signal'
                                        WHEN rel.description = 'instantiates module' THEN 'instantiated by method'
                                         WHEN rel.description = 'instantiated by method' THEN 'instantiates module'
                                         WHEN rel.description = 'contained in module' THEN 'contains rtl entity'
                                         WHEN rel.description = 'includes header' THEN 'included from file'
                                         WHEN rel.description = 'included from file' THEN 'includes header'
                                         WHEN rel.description = 'imports package symbol' THEN 'imported by file'
                                         WHEN rel.description = 'imported by file' THEN 'imports package symbol'
                                         WHEN rel.description = 'declares type' THEN 'declared in file'
                                         WHEN rel.description = 'declared in file' THEN 'declares type'
                                         WHEN rel.description = 'clocked by signal' THEN 'clocks logic'
                                         WHEN rel.description = 'clocks logic' THEN 'clocked by signal'
                                         WHEN rel.description = 'reset by signal' THEN 'resets logic'
                                         WHEN rel.description = 'resets logic' THEN 'reset by signal'
                                         WHEN rel.description = 'guards conditional compilation' THEN 'guarded by macro'
                                         WHEN rel.description = 'guarded by macro' THEN 'guards conditional compilation'
                                         WHEN rel.description = 'transitions to state' THEN 'transitioned from state'
                                         WHEN rel.description = 'transitioned from state' THEN 'transitions to state'
                                         WHEN rel.description = 'exercises module' THEN 'exercised by testbench'
                                         WHEN rel.description = 'exercised by testbench' THEN 'exercises module'
                                         ELSE rel.description
                                     END
                                 END
                            }
                        ][0],
                        [
                            (start)-[rel:RELATED]-(end) 
                            WHERE id(start) = nodeIds[i] AND id(end) = nodeIds[i+1] |
                            {
                                start_node: CASE 
                                    WHEN start:Commit THEN 'Commit#' + start.id
                                    WHEN start:Issue THEN start.name
                                    ELSE start.name
                                END,
                                end_node: CASE 
                                    WHEN end:Commit THEN 'Commit#' + end.id
                                    WHEN end:Issue THEN end.name
                                    ELSE end.name
                                END,
                                type: type(rel),
                                relation_kind: coalesce(rel.relation_kind, toLower(type(rel))),
                                description: CASE
                                    WHEN id(start) = id(startNode(rel)) THEN rel.description
                                    ELSE CASE
                                        WHEN rel.description = 'contains method' THEN 'contained in method'
                                        WHEN rel.description = 'contained in method' THEN 'contains method'
                                        WHEN rel.description = 'contains class' THEN 'contained in class'
                                        WHEN rel.description = 'contained in class' THEN 'contains class'
                                        WHEN rel.description = 'contains file' THEN 'contained in file'
                                        WHEN rel.description = 'contained in file' THEN 'contains file'
                                        WHEN rel.description = 'points to issue' THEN 'referenced by issue'
                                        WHEN rel.description = 'referenced by issue' THEN 'points to issue'
                                        WHEN rel.description = 'calls method' THEN 'called by method'
                                        WHEN rel.description = 'called by method' THEN 'calls method'
                                        WHEN rel.description = 'reads signal' THEN 'read by method'
                                        WHEN rel.description = 'read by method' THEN 'reads signal'
                                        WHEN rel.description = 'writes signal' THEN 'written by method'
                                        WHEN rel.description = 'written by method' THEN 'writes signal'
                                        WHEN rel.description = 'drives signal' THEN 'driven by method'
                                        WHEN rel.description = 'driven by method' THEN 'drives signal'
                                        WHEN rel.description = 'feeds signal' THEN 'fed by signal'
                                        WHEN rel.description = 'fed by signal' THEN 'feeds signal'
                                        WHEN rel.description = 'connects signal' THEN 'connected by instance'
                                        WHEN rel.description = 'connected by instance' THEN 'connects signal'
                                        WHEN rel.description = 'instantiates module' THEN 'instantiated by method'
                                         WHEN rel.description = 'instantiated by method' THEN 'instantiates module'
                                         WHEN rel.description = 'contained in module' THEN 'contains rtl entity'
                                         WHEN rel.description = 'includes header' THEN 'included from file'
                                         WHEN rel.description = 'included from file' THEN 'includes header'
                                         WHEN rel.description = 'imports package symbol' THEN 'imported by file'
                                         WHEN rel.description = 'imported by file' THEN 'imports package symbol'
                                         WHEN rel.description = 'declares type' THEN 'declared in file'
                                         WHEN rel.description = 'declared in file' THEN 'declares type'
                                         WHEN rel.description = 'clocked by signal' THEN 'clocks logic'
                                         WHEN rel.description = 'clocks logic' THEN 'clocked by signal'
                                         WHEN rel.description = 'reset by signal' THEN 'resets logic'
                                         WHEN rel.description = 'resets logic' THEN 'reset by signal'
                                         WHEN rel.description = 'guards conditional compilation' THEN 'guarded by macro'
                                         WHEN rel.description = 'guarded by macro' THEN 'guards conditional compilation'
                                         WHEN rel.description = 'transitions to state' THEN 'transitioned from state'
                                         WHEN rel.description = 'transitioned from state' THEN 'transitions to state'
                                         WHEN rel.description = 'exercises module' THEN 'exercised by testbench'
                                         WHEN rel.description = 'exercised by testbench' THEN 'exercises module'
                                         ELSE rel.description
                                     END
                                 END
                            }
                        ][0]
                    )
                    ] as path_details

                WITH m, path_details, totalCost as cost, root_embedding, root_text,
                    coalesce(m.semantic_summary, m.source_code, m.declaration, m.name) as m_text
                WITH m, path_details, cost,
                    CASE 
                        WHEN m:Issue THEN 
                            gds.similarity.cosine(root_embedding, m.embedding) * ($DECAY_FACTOR ^ cost)
                        ELSE
                            (gds.similarity.cosine(root_embedding, m.embedding) * $VECTOR_SIMILARITY_WEIGHT +
                            apoc.text.levenshteinSimilarity(root_text, m_text) * (1 - $VECTOR_SIMILARITY_WEIGHT)) *
                            ($DECAY_FACTOR ^ cost)
                    END as similarity_score
                ORDER BY similarity_score DESC
                LIMIT 10000

                RETURN collect({
                    type: CASE 
                        WHEN m:PatchableSpan THEN 'patchable_span'
                        WHEN m:Method THEN 'method' 
                        WHEN m:Class THEN 'class'
                        WHEN m:Signal THEN 'signal'
                        WHEN m:Port THEN 'port'
                        WHEN m:Parameter THEN 'parameter'
                        WHEN m:Macro THEN 'macro'
                        WHEN m:State THEN 'state'
                        WHEN m:Branch THEN 'branch'
                        WHEN m:Condition THEN 'condition'
                        WHEN m:GenerateBlock THEN 'generate_block'
                        WHEN m:ConditionalCompilationScope THEN 'conditional_compilation'
                        WHEN m:Testbench THEN 'testbench'
                        WHEN m:Assertion THEN 'assertion'
                        WHEN m:Instance THEN 'instance'
                        WHEN m:ClockDomain THEN 'clock_domain'
                        WHEN m:ResetDomain THEN 'reset_domain'
                        WHEN m:Typedef THEN 'typedef'
                        WHEN m:Import THEN 'import'
                        WHEN m:Include THEN 'include'
                        ELSE 'issue'
                    END,
                    name: m.name,
                    signature: CASE WHEN m:Method THEN m.signature WHEN m:PatchableSpan THEN m.name ELSE null END,
                    file_path: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.file_path ELSE null END,
                    documentation: CASE WHEN m:Method OR m:Class THEN m.doc_string ELSE null END,
                    source_code: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.source_code ELSE null END,
                    semantic_summary: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.semantic_summary ELSE null END,
                    repair_role: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.repair_role ELSE null END,
                    timing_tags: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.timing_tags ELSE null END,
                    timing_priority: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.timing_priority ELSE null END,
                    timing_summary: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.timing_summary ELSE null END,
                    verilog_kind: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.verilog_kind ELSE null END,
                    signal_name: CASE WHEN any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.signal_name ELSE null END,
                    module_name: CASE WHEN any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.module_name ELSE null END,
                    direction: CASE WHEN m:Port THEN m.direction ELSE null END,
                    width: CASE WHEN m:Signal OR m:Port OR m:Parameter THEN m.width ELSE null END,
                    declaration: CASE WHEN any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.declaration ELSE null END,
                    start_line: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.start_line ELSE null END,
                    end_line: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.end_line ELSE null END,
                    start_column: CASE WHEN m:PatchableSpan THEN m.start_column ELSE null END,
                    end_column: CASE WHEN m:PatchableSpan THEN m.end_column ELSE null END,
                    parse_source: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.parse_source ELSE null END,
                    parse_confidence: CASE WHEN m:Method OR m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.parse_confidence ELSE null END,
                    span_kind: CASE WHEN m:PatchableSpan THEN m.span_kind ELSE null END,
                    patch_granularity: CASE WHEN m:PatchableSpan THEN m.patch_granularity ELSE null END,
                    parent_span: CASE WHEN m:PatchableSpan THEN m.parent_span ELSE null END,
                    reads_signals: CASE WHEN m:PatchableSpan THEN m.reads_signals ELSE null END,
                    writes_signals: CASE WHEN m:PatchableSpan THEN m.writes_signals ELSE null END,
                    drives_signals: CASE WHEN m:PatchableSpan THEN m.drives_signals ELSE null END,
                    connects_signals: CASE WHEN m:PatchableSpan THEN m.connects_signals ELSE null END,
                    guards_signals: CASE WHEN m:PatchableSpan THEN m.guards_signals ELSE null END,
                    port_bindings: CASE WHEN m:PatchableSpan THEN m.port_bindings ELSE null END,
                    projection_source: CASE WHEN m:PatchableSpan THEN m.projection_source ELSE null END,
                    issue_id: CASE WHEN m:Issue THEN m.id ELSE null END,
                    title: CASE WHEN m:Issue THEN m.title ELSE null END,
                    content: CASE WHEN m:Issue THEN m.content ELSE null END,
                    similarity: similarity_score,
                    distance: cost,
                    path: path_details
                }) as methods
                """
                
                # 4. Execute query and get results
                method_result = session.run(
                    method_query,
                    max_hops=float(max_hops),
                    max_target_nodes=max_target_nodes,
                    VECTOR_SIMILARITY_WEIGHT=VECTOR_SIMILARITY_WEIGHT,
                    DECAY_FACTOR=DECAY_FACTOR,
                    issue_tokens=issue_tokens,
                    issue_mentioned_files=issue_mentioned_files,
                    hdl_entity_labels=hdl_entity_labels,
                )
                method_record = method_result.single()
                method_similarities = method_record['methods'] if method_record else []
                if issue_tokens or issue_mentioned_files:
                    fallback_query = """
                    MATCH (m)
                    WHERE (m:Class OR any(label IN labels(m) WHERE label IN $hdl_entity_labels))
                      AND coalesce(m.file_path, '') <> ''
                    WITH m,
                         toLower(coalesce(m.file_path, '')) AS file_path,
                         toLower(coalesce(m.name, '')) AS name,
                         toLower(coalesce(m.module_name, '')) AS module_name,
                         toLower(coalesce(m.signature, '')) AS signature
                    WITH m, file_path, name, module_name, signature,
                         [token IN $issue_tokens WHERE token <> '' AND (
                            file_path CONTAINS token OR name CONTAINS token OR
                            module_name CONTAINS token OR signature CONTAINS token
                         )] AS token_hits,
                         [hint IN $issue_mentioned_files WHERE hint <> '' AND (
                            file_path = toLower(hint) OR file_path ENDS WITH toLower(hint)
                         )] AS hint_hits
                    WHERE size(token_hits) > 0 OR size(hint_hits) > 0
                    WITH m, size(token_hits) AS token_score, size(hint_hits) AS hint_score
                    ORDER BY hint_score DESC, token_score DESC
                    LIMIT $fallback_limit
                    RETURN collect({
                        type: CASE
                            WHEN m:PatchableSpan THEN 'patchable_span'
                            WHEN m:Method THEN 'method'
                            WHEN m:Class THEN 'class'
                            WHEN m:Signal THEN 'signal'
                            WHEN m:Port THEN 'port'
                            WHEN m:Parameter THEN 'parameter'
                            WHEN m:Macro THEN 'macro'
                            WHEN m:State THEN 'state'
                            WHEN m:Branch THEN 'branch'
                            WHEN m:Condition THEN 'condition'
                            WHEN m:GenerateBlock THEN 'generate_block'
                            WHEN m:ConditionalCompilationScope THEN 'conditional_compilation'
                            WHEN m:Testbench THEN 'testbench'
                            WHEN m:Assertion THEN 'assertion'
                            WHEN m:Instance THEN 'instance'
                            WHEN m:ClockDomain THEN 'clock_domain'
                            WHEN m:ResetDomain THEN 'reset_domain'
                            WHEN m:Typedef THEN 'typedef'
                            WHEN m:Import THEN 'import'
                            WHEN m:Include THEN 'include'
                            ELSE 'entity'
                        END,
                        name: m.name,
                        signature: CASE WHEN m:Method THEN m.signature WHEN m:PatchableSpan THEN m.name ELSE null END,
                        file_path: m.file_path,
                        documentation: CASE WHEN m:Method OR m:Class THEN m.doc_string ELSE null END,
                        source_code: m.source_code,
                        semantic_summary: m.semantic_summary,
                        repair_role: m.repair_role,
                        timing_tags: m.timing_tags,
                        timing_priority: m.timing_priority,
                        timing_summary: m.timing_summary,
                        verilog_kind: m.verilog_kind,
                        signal_name: CASE WHEN any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.signal_name ELSE null END,
                        module_name: CASE WHEN any(label IN labels(m) WHERE label IN $hdl_entity_labels) THEN m.module_name ELSE null END,
                        direction: CASE WHEN m:Port THEN m.direction ELSE null END,
                        width: CASE WHEN m:Signal OR m:Port OR m:Parameter THEN m.width ELSE null END,
                        declaration: m.declaration,
                        start_line: m.start_line,
                        end_line: m.end_line,
                        start_column: CASE WHEN m:PatchableSpan THEN m.start_column ELSE null END,
                        end_column: CASE WHEN m:PatchableSpan THEN m.end_column ELSE null END,
                        parse_source: m.parse_source,
                        parse_confidence: m.parse_confidence,
                        span_kind: CASE WHEN m:PatchableSpan THEN m.span_kind ELSE null END,
                        patch_granularity: CASE WHEN m:PatchableSpan THEN m.patch_granularity ELSE null END,
                        parent_span: CASE WHEN m:PatchableSpan THEN m.parent_span ELSE null END,
                        reads_signals: CASE WHEN m:PatchableSpan THEN m.reads_signals ELSE null END,
                        writes_signals: CASE WHEN m:PatchableSpan THEN m.writes_signals ELSE null END,
                        drives_signals: CASE WHEN m:PatchableSpan THEN m.drives_signals ELSE null END,
                        connects_signals: CASE WHEN m:PatchableSpan THEN m.connects_signals ELSE null END,
                        guards_signals: CASE WHEN m:PatchableSpan THEN m.guards_signals ELSE null END,
                        port_bindings: CASE WHEN m:PatchableSpan THEN m.port_bindings ELSE null END,
                        projection_source: CASE WHEN m:PatchableSpan THEN m.projection_source ELSE null END,
                        similarity: 0.35 + token_score * 0.05 + hint_score * 0.30,
                        distance: 0.5,
                        path: []
                    }) AS fallback
                    """
                    fallback_record = session.run(
                        fallback_query,
                        issue_tokens=issue_tokens,
                        issue_mentioned_files=[item.lower() for item in issue_mentioned_files],
                        fallback_limit=min(100, limit),
                        hdl_entity_labels=hdl_entity_labels,
                    ).single()
                    fallback_entities = fallback_record['fallback'] if fallback_record else []
                    seen_similarity_keys = {
                        (item.get('type'), item.get('name'), item.get('file_path'), item.get('start_line'))
                        for item in method_similarities
                        if isinstance(item, dict)
                    }
                    for item in fallback_entities:
                        key = (item.get('type'), item.get('name'), item.get('file_path'), item.get('start_line'))
                        if key not in seen_similarity_keys:
                            method_similarities.append(item)
                            seen_similarity_keys.add(key)
                if issue_tokens or issue_mentioned_files:
                    for sim in method_similarities:
                        if not isinstance(sim, dict):
                            continue
                        boost = 0.0
                        file_path = str(sim.get('file_path') or '').replace('\\', '/').lower()
                        name = str(sim.get('name') or '').lower()
                        module_name = str(sim.get('module_name') or '').lower()
                        signature = str(sim.get('signature') or '').lower()
                        for token in issue_tokens:
                            if not token:
                                continue
                            if token in name or token in module_name or token in signature or token in file_path:
                                boost += 0.12
                        for path_hint in issue_mentioned_files:
                            hint = str(path_hint or '').replace('\\', '/').lower()
                            if hint and (file_path == hint or file_path.endswith(hint)):
                                boost += 0.30
                        if boost:
                            try:
                                sim['similarity'] = float(sim.get('similarity') or 0.0) + boost
                            except (TypeError, ValueError):
                                sim['similarity'] = boost
                # 5. Process and organize results
                results = {
                    'patchable_spans': list({
                        (sim['name'], sim.get('file_path'), sim.get('start_line'), sim.get('end_line')): sim
                        for sim in method_similarities
                        if sim['type'] == 'patchable_span' and sim['similarity'] is not None
                    }.values()),
                    'methods': list({
                        (sim['name'], sim.get('signature')): sim
                        for sim in method_similarities
                        if sim['type'] == 'method' and sim['similarity'] is not None
                    }.values()),
                    'classes': list({
                        sim['name']: sim
                        for sim in method_similarities
                        if sim['type'] == 'class' and sim['similarity'] is not None
                    }.values()),
                    'rtl_entities': list({
                        (sim['type'], sim['name'], sim.get('file_path'), sim.get('start_line')): sim
                        for sim in method_similarities
                        if sim['type'] in {
                            'signal',
                            'port',
                            'parameter',
                            'macro',
                            'state',
                            'branch',
                            'condition',
                            'generate_block',
                            'conditional_compilation',
                            'testbench',
                            'assertion',
                            'instance',
                            'clock_domain',
                            'reset_domain',
                            'typedef',
                            'import',
                            'include',
                        } and sim['similarity'] is not None
                    }.values()),
                    'issues': list({
                        sim['issue_id']: sim
                        for sim in method_similarities
                        if sim['type'] == 'issue' and sim['similarity'] is not None
                    }.values())
                }

                results['methods'] = [
                    self._attach_context_metadata(item, 'methods')
                    for item in sorted(results['methods'], key=lambda x: x.get('similarity', 0), reverse=True)
                ]
                results['patchable_spans'] = [
                    self._attach_context_metadata(item, 'patchable_spans')
                    for item in sorted(results['patchable_spans'], key=lambda x: x.get('similarity', 0), reverse=True)
                ]
                results['classes'] = [
                    self._attach_context_metadata(item, 'classes')
                    for item in sorted(results['classes'], key=lambda x: x.get('similarity', 0), reverse=True)
                ]
                results['rtl_entities'] = [
                    self._attach_context_metadata(item, 'rtl_entities')
                    for item in sorted(results['rtl_entities'], key=lambda x: x.get('similarity', 0), reverse=True)
                ]
                results['issues'] = [
                    self._attach_context_metadata(item, 'issues')
                    for item in sorted(results['issues'], key=lambda x: x.get('similarity', 0), reverse=True)
                ]
                results['edit_targets'] = list(results['patchable_spans'])
                results['evidence_entities'] = list(results['rtl_entities'])
                
                # Retrieve root issue
                root_query = """
                MATCH (root:Issue {id: 'root'})
                RETURN {
                    type: 'issue',
                    name: root.name,
                    issue_id: root.id,
                    title: root.title,
                    content: root.content,
                    similarity: 2.0,
                    distance: 0,
                    path: []
                } as root_issue
                """
                root_result = session.run(root_query)
                root_record = root_result.single()
                if root_record:
                    results['issues'].insert(0, root_record['root_issue'])
                
                # 6. Sort and limit results by the original KGCompass similarity score.
                if sort or limit:
                    for key in list(results.keys()):
                        results[key] = sorted(
                            results[key],
                            key=lambda x: x.get('similarity', 0),
                            reverse=True,
                        )
                        if limit:
                            results[key] = results[key][:limit]
                
                return results
                
            finally:
                # 7. Cleanup: Delete graph projection to free memory
                session.run("CALL gds.graph.drop('graph', false)")

    def _create_indexes(self):
        """Create database indexes to improve query performance"""
        with self.driver.session() as session:
            # Method node index
            session.run("""
                CREATE INDEX method_composite IF NOT EXISTS
                FOR (m:Method)
                ON (m.name, m.signature, m.file_path)
            """)
            
            # Issue node index
            session.run("""
                CREATE INDEX issue_id IF NOT EXISTS
                FOR (i:Issue)
                ON (i.id)
            """)
            
            # File node index
            session.run("""
                CREATE INDEX file_path IF NOT EXISTS
                FOR (f:File)
                ON (f.path)
            """)
            
            # Class node index
            session.run("""
                CREATE INDEX class_composite IF NOT EXISTS
                FOR (c:Class)
                ON (c.name, c.file_path)
            """)
            
            # Commit node index
            session.run("""
                CREATE INDEX commit_id IF NOT EXISTS
                FOR (c:Commit)
                ON (c.id)
            """)
            
            # Directory node index
            session.run("""
                CREATE INDEX directory_path IF NOT EXISTS
                FOR (d:Directory)
                ON (d.path)
            """)

            for label in self.HDL_ENTITY_LABELS:
                session.run(f"""
                    CREATE INDEX {label.lower()}_composite IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.name, n.file_path, n.module_name)
                """)
             
            tinfo("KG indexes created")
                
    def link_class_to_file(self, class_name, file_path, weight=1):
        """
        Establish relationship between class and file
        
        Args:
            class_name (str): Class name
            file_path (str): File path
        """
        with self.driver.session() as session:
            match_clause = """
            MATCH (c:Class {name: $class_name, file_path: $file_path})
            MATCH (f:File {path: $file_path})
            """
            params = {
                'class_name': class_name,
                'file_path': file_path,
            }
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'c', 'f', 'contains',
                'contained in file', weight, params
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'f', 'c', 'contains',
                'contains class', weight, params
            )
            
    def link_method_to_file(self, method_name, method_signature, file_path, weight=1):
        """
        Establish relationship between method and file
        
        Args:
            method_name (str): Method name
            method_signature (str): Method signature
            file_path (str): File path
        """
        with self.driver.session() as session:
            match_clause = """
            MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path})
            MATCH (f:File {path: $file_path})
            """
            params = {
                'method_name': method_name,
                'method_signature': method_signature,
                'file_path': file_path,
            }
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'm', 'f', 'contains',
                'contained in file', weight, params
            )
            session.execute_write(
                self._merge_semantic_and_compat_relationship,
                match_clause, 'f', 'm', 'contains',
                'contains method', weight, params
            )
