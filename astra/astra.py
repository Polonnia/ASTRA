import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken


from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    gpt_35_turbo_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    build_dual_level_graphs,
    chunking_by_token_size,
    extract_entities,
    get_chunks,
    naive_query,
    dual_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    TextChunkSchema,
)
from .prompt import GRAPH_FIELD_SEP
from .token_usage import token_usage_stage


@dataclass
class ASTRA:
    working_dir: str = field(
        default_factory=lambda: f"./astra_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_hierachical_mode: bool = True

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    language: str = "english"
    enable_structural_graph: bool = False

    # dual retrieval runtime controls
    enable_tree_traversal: bool = False
    detail_top_k_entity: int = 10
    detail_top_k_relation: int = 10
    macro_top_k_entity: int = 10
    macro_top_k_relation: int = 10
    keyword_top_k_entity: int = 10
    keyword_top_k_relation: int = 10
    max_token_for_text_unit: int = 20000
    max_token_for_graph: int = 20000
    max_token_for_summary: int = 20000

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 8
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    # best_model_func: callable = gpt_35_turbo_complete
    best_model_func: callable = gpt_4o_mini_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 8
    cheap_model_func: callable = gpt_35_turbo_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 8

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # export
    export_workbench_json: bool = True

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"ASTRA init with param:\n\n  {_print_config}\n")

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name", "entity_type", "level", "kind", "source", "target", "graph_level"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "dual" and self.entities_vdb is None:
            raise ValueError("dual mode requires enable_local so the mixed vector store exists")

        entities_vdb = self.entities_vdb

        if param.mode == "naive":                     # retrieve with only text units
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "dual":
            assert entities_vdb is not None
            response = await dual_query(
                query,
                self.chunk_entity_relation_graph,
                entities_vdb,
                self.full_docs,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        level_graph_exports = None
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {    # dict: {hash: ori_content}
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))     # filter the docs that has already in the storage.
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if len(new_docs):
                logger.info(f"[New Docs] inserting {len(new_docs)} docs")
            else:
                logger.info("No new docs found, checking pending chunks for resume")

            # ---------- chunking
            inserting_chunks = {}
            if len(new_docs):
                inserting_chunks = get_chunks(
                    new_docs=new_docs,
                    chunk_func=self.chunk_func,
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    language=self.language,
                )

                _add_chunk_keys = await self.text_chunks.filter_keys(
                    list(inserting_chunks.keys())
                )
                inserting_chunks = {
                    k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
                }
                for _, chunk_data in inserting_chunks.items():
                    chunk_data["processed"] = False
                    chunk_data["dual_processed"] = False

                if len(inserting_chunks):
                    logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
                    if self.enable_naive_rag:
                        logger.info("Insert chunks for naive RAG")
                        await self.chunks_vdb.upsert(inserting_chunks)
                else:
                    logger.info("No new chunks created from new docs")

            # Persist new docs/chunks first so per-chunk processing state can be resumed.
            if len(new_docs):
                await self.full_docs.upsert(new_docs)
            if len(inserting_chunks):
                await self.text_chunks.upsert(inserting_chunks)

            # Collect pending chunks from storage (processed != True)
            all_chunk_keys = await self.text_chunks.all_keys()
            all_chunk_values = await self.text_chunks.get_by_ids(all_chunk_keys)
            pending_chunks = {
                chunk_id: chunk_data
                for chunk_id, chunk_data in zip(all_chunk_keys, all_chunk_values)
                if isinstance(chunk_data, dict) and not bool(chunk_data.get("processed", False))
            }
            pending_dual_chunks = {
                chunk_id: chunk_data
                for chunk_id, chunk_data in zip(all_chunk_keys, all_chunk_values)
                if isinstance(chunk_data, dict) and not bool(chunk_data.get("dual_processed", False))
            }
            all_chunks = {
                chunk_id: chunk_data
                for chunk_id, chunk_data in zip(all_chunk_keys, all_chunk_values)
                if isinstance(chunk_data, dict)
            }

            dual_mode_enabled = self.enable_structural_graph and str(self.language).strip().lower() == "english"

            if dual_mode_enabled:
                dual_build_chunks = cast(
                    Dict[str, TextChunkSchema],
                    pending_dual_chunks if pending_dual_chunks else all_chunks,
                )
                if not dual_build_chunks:
                    logger.info("No chunks available for dual-level graph")
                    return
                if pending_dual_chunks:
                    logger.info(f"[Pending Dual Chunks] processing {len(pending_dual_chunks)} chunks")
                else:
                    logger.info("No pending dual chunks, run dual build for Step8 resume check")

                logger.info("\033[94m[Dual-Level Graph Extraction]...\033[0m")
                level_graph_exports = await build_dual_level_graphs(
                    new_docs=new_docs,
                    chunks=dual_build_chunks,
                    knwoledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    global_config=asdict(self),
                    text_chunks_storage=self.text_chunks,
                    full_docs_db=self.full_docs,
                )
            else:
                if not pending_chunks:
                    logger.info("No pending chunks to process")
                    return
                logger.info(f"[Pending Chunks] processing {len(pending_chunks)} chunks")

                # ---------- extract/summary entity and upsert to graph
                logger.info("\033[94m[Entity Extraction]...\033[0m")
                with token_usage_stage("entity_relation_extraction"):
                    maybe_new_kg = await self.entity_extraction_func(
                        pending_chunks,
                        knwoledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        global_config=asdict(self),
                        text_chunks_storage=self.text_chunks,
                    )
                if maybe_new_kg is None:
                    logger.warning("No new entities found")
                    return
                self.chunk_entity_relation_graph = maybe_new_kg

            # ---------- export graph entities/relationships for graphrag-workbench
            if self.export_workbench_json:
                try:
                    self._export_graph_for_workbench_json()
                    if level_graph_exports:
                        self._export_dual_level_graph_json(level_graph_exports)
                except Exception as e:
                    logger.warning(f"Failed to export workbench JSON: {e}")
        finally:
            await self._insert_done()

    def _split_text_unit_ids(self, source_id: object) -> List[str]:
        text = str(source_id or "").strip()
        if not text:
            return []
        return [x.strip() for x in text.split(GRAPH_FIELD_SEP) if x.strip()]

    def _parse_section_ids(self, section_value: object) -> List[str]:
        if section_value is None:
            return []
        if isinstance(section_value, list):
            return [str(s).strip() for s in section_value if str(s).strip()]
        text = str(section_value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except Exception:
            pass
        return [x.strip() for x in text.split(GRAPH_FIELD_SEP) if x.strip()]

    def _to_int(self, value: object, default: int = 0) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _to_float(self, value: object, default: float = 0.0) -> float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _resolve_workbench_output_dir(self) -> str:
        configured = str(self.addon_params.get("workbench_output_dir", "")).strip()
        if configured:
            return configured
        return self.working_dir

    def _infer_section_ids_from_chunks(self, text_unit_ids: List[str]) -> List[str]:
        if not text_unit_ids:
            return []
        chunks_data = getattr(self.text_chunks, "_data", {}) if self.text_chunks else {}
        if not isinstance(chunks_data, dict):
            return []
        section_ids: List[str] = []
        for chunk_id in text_unit_ids:
            chunk = chunks_data.get(chunk_id)
            if not isinstance(chunk, dict):
                continue
            section = str(chunk.get("structure_leaf_node_id", "")).strip()
            if section:
                section_ids.append(section)
        return sorted(set(section_ids))

    def _export_graph_for_workbench_json(self) -> None:
        graph = getattr(self.chunk_entity_relation_graph, "_graph", None)
        if graph is None:
            logger.warning(
                "Skip workbench JSON export: current graph storage does not expose in-memory graph"
            )
            return

        entities = []
        is_chinese = str(self.language).strip().lower() in {"chinese", "zh", "中文"}
        for i, (node_id, node_data) in enumerate(graph.nodes(data=True), start=1):
            entity_types = [
                t.strip()
                for t in str(node_data.get("entity_type", "unnamed")).split(
                    GRAPH_FIELD_SEP
                )
                if t.strip()
            ]
            primary_type = entity_types[0] if entity_types else "unnamed"
            if is_chinese and primary_type.lower() == "unknown":
                primary_type = "未知"
            if primary_type == "未知":
                pass
            elif primary_type.lower() not in {"unnamed", "unknown"} and primary_type.isascii():
                primary_type = primary_type.upper()

            text_unit_ids = self._split_text_unit_ids(node_data.get("source_id", ""))
            section_ids = self._parse_section_ids(node_data.get("section", []))
            if not section_ids:
                section_ids = self._infer_section_ids_from_chunks(text_unit_ids)

            entities.append(
                {
                    "id": str(node_id),
                    "human_readable_id": str(i),
                    "title": str(node_id),
                    "type": primary_type,
                    "description": str(node_data.get("description", "")),
                    "text_unit_ids": text_unit_ids,
                    "frequency": len(text_unit_ids),
                    "degree": int(graph.degree(node_id)),
                    "level": self._to_int(node_data.get("level", 0), 0),
                    "section": section_ids,
                }
            )

        relationships = []
        for i, (source, target, edge_data) in enumerate(graph.edges(data=True), start=1):
            text_unit_ids = self._split_text_unit_ids(edge_data.get("source_id", ""))
            relationships.append(
                {
                    "id": str(edge_data.get("id", f"{source}->{target}")),
                    "human_readable_id": str(i),
                    "source": str(source),
                    "target": str(target),
                    "description": str(edge_data.get("description", "")),
                    "weight": self._to_float(edge_data.get("weight", 1.0), 1.0),
                    "combined_degree": int(graph.degree(source) + graph.degree(target)),
                    "text_unit_ids": text_unit_ids,
                }
            )

        output_dir = self._resolve_workbench_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        entities_path = os.path.join(output_dir, "entities.json")
        relationships_path = os.path.join(output_dir, "relationships.json")

        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        with open(relationships_path, "w", encoding="utf-8") as f:
            json.dump(relationships, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Exported workbench JSON: entities={len(entities)}, relationships={len(relationships)} -> {output_dir}"
        )

    def _export_dual_level_graph_json(self, level_graph_exports: dict) -> None:
        if not isinstance(level_graph_exports, dict):
            return

        output_dir = os.path.join(self._resolve_workbench_output_dir(), "graphs")
        os.makedirs(output_dir, exist_ok=True)

        file_mappings = {
            "structural_graph_entities": "structural_graph_entities.json",
            "structural_graph_relations": "structural_graph_relations.json",
            "content_graph_entities": "content_graph_entities.json",
            "content_graph_relations": "content_graph_relations.json",
        }

        for key, filename in file_mappings.items():
            data = level_graph_exports.get(key, [])
            if not isinstance(data, list):
                data = []
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Exported dual-level graph JSON files to %s",
            output_dir,
        )

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
