import re
import json
import asyncio
import tiktoken
import networkx as nx
import time
import logging
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Union
from collections import defaultdict, deque
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    normalize_extracted_field,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


@contextmanager
def timer():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"\033[94m[Retrieval Time: {elapsed_time:.6f} seconds]\033[0m")


def _count_tokens_safe(text: str, model_name: str) -> int:
    try:
        return len(encode_string_by_tiktoken(str(text or ""), model_name=model_name))
    except Exception:
        return 0


def _sum_tokens(items: list[Any], key: callable, model_name: str) -> int:
    total = 0
    for item in items:
        total += _count_tokens_safe(key(item), model_name)
    return total


def _write_dual_query_log(global_config: dict, payload: dict[str, Any]) -> None:
    working_dir = str(global_config.get("working_dir", ".")).strip() or "."
    log_file = str(global_config.get("dual_query_log_file", "")).strip()
    if not log_file:
        log_file = os.path.join(working_dir, "dual_query_retrieval_log.jsonl")
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[Dual][Log] Failed to write retrieval log: %s", e)


def _write_extraction_parse_log(global_config: dict, payload: dict[str, Any]) -> None:
    working_dir = str(global_config.get("working_dir", ".")).strip() or "."
    log_file = str(global_config.get("extraction_parse_log_file", "")).strip()
    if not log_file:
        log_file = os.path.join(working_dir, "extraction_parse_failures.jsonl")
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[Extraction][Log] Failed to write parse failure log: %s", e)


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
  # tokenizer
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    language = str(chunk_func_params.get("language", "english")).strip().lower()
    is_chinese = language in {"chinese", "zh", "中文"}
    # Keep language for structured-doc branch only; default token chunkers don't accept it.
    runtime_chunk_params = dict(chunk_func_params)
    runtime_chunk_params.pop("language", None)

    for doc_key, doc_value in new_docs_list:
        doc_content = doc_value["content"]
        print(doc_content[:100], flush=True)  # for debugging, print the beginning of the content
        structured_chunks = _extract_leaf_chunks_from_structured_doc(
            doc_key=doc_key,
            content=doc_content,
            tiktoken_model=ENCODER,
            overlap_token_size=chunk_func_params.get("overlap_token_size", 128),
            max_token_size=chunk_func_params.get("max_token_size", 1024),
            use_summary=is_chinese,
        )
        if structured_chunks is not None:
            inserting_chunks.update(structured_chunks)
            continue

        tokens = ENCODER.encode_batch([doc_content], num_threads=16)
        chunks = chunk_func(
            tokens,
            doc_keys=[doc_key],
            tiktoken_model=ENCODER,
            **runtime_chunk_params,
        )
        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            )

    return inserting_chunks


def _extract_leaf_chunks_from_structured_doc(
    doc_key: str,
    content: str,
    tiktoken_model: tiktoken.Encoding,
    overlap_token_size: int = 128,
    max_token_size: int = 4096,
    use_summary: bool = False,
) -> Union[dict[str, TextChunkSchema], None]:
    """Parse a structured JSON document and create chunks from structure nodes.

    The expected schema is similar to:
    {
      "structure": [
        {"title": ..., "start_index": ..., "end_index": ..., "summary": ..., "text": ...},
        ...
      ]
    }
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    structure = parsed.get("structure") if isinstance(parsed, dict) else None
    if not isinstance(structure, list) or not structure:
        return None
    doc_name = str(parsed.get("doc_name", "")).strip() if isinstance(parsed, dict) else ""
    doc_description = str(parsed.get("doc_description", "")).strip() if isinstance(parsed, dict) else ""

    nodes = []

    def _collect_nodes(raw_nodes: list, parent_path: str = ""):
        for i, raw_node in enumerate(raw_nodes):
            if not isinstance(raw_node, dict):
                continue
            title = raw_node.get("title")
            summary = raw_node.get("summary")
            text = raw_node.get("text")
            start_index = raw_node.get("start_index")
            end_index = raw_node.get("end_index")

            path = f"{parent_path}.{i}" if parent_path else str(i)
            node_id = str(raw_node.get("node_id", "")).strip() or path
            raw_children = raw_node.get("nodes")
            if not isinstance(raw_children, list):
                raw_children = raw_node.get("children")
            child_nodes: list[dict] = raw_children if isinstance(raw_children, list) else []
            node_title = title.strip() if isinstance(title, str) and title.strip() else node_id

            nodes.append(
                {
                    "path": path,
                    "title": node_title,
                    "node_id": node_id,
                    "summary": summary.strip() if isinstance(summary, str) else "",
                    "text": text.strip() if isinstance(text, str) else "",
                    "start": start_index,
                    "end": end_index,
                    "has_children": any(isinstance(c, dict) for c in child_nodes),
                }
            )

            if child_nodes:
                _collect_nodes(child_nodes, parent_path=path)

    _collect_nodes(structure)

    if not nodes:
        return None

    if use_summary:
        target_nodes = [n for n in nodes if n["summary"]]
        if not target_nodes:
            return None
        root_node_count = sum(1 for n in target_nodes if "." not in n["path"])
        nested_node_count = len(target_nodes) - root_node_count
        logger.info(
            f"Structured summary scan for {doc_key}: total={len(target_nodes)}, root={root_node_count}, nested={nested_node_count}"
        )
    else:
        # Leaf nodes are nodes with content text and no valid children.
        target_nodes = [n for n in nodes if n["text"] and not n["has_children"]]
        if not target_nodes:
            return None
        root_leaf_count = sum(1 for n in target_nodes if "." not in n["path"])
        nested_leaf_count = len(target_nodes) - root_leaf_count
        logger.info(
            f"Structured leaf scan for {doc_key}: total={len(target_nodes)}, root={root_leaf_count}, nested={nested_leaf_count}"
        )

    chunks: dict[str, TextChunkSchema] = {}
    global_chunk_order = 0
    for node in target_nodes:
        node_content = node["summary"] if use_summary else node["text"]
        node_tokens = tiktoken_model.encode(node_content)
        split_stride = max(1, max_token_size - overlap_token_size)
        if len(node_tokens) <= max_token_size:
            sub_chunks = [node_content]
        else:
            sub_chunks = []
            for start in range(0, len(node_tokens), split_stride):
                sub_tokens = node_tokens[start : start + max_token_size]
                sub_text = tiktoken_model.decode(sub_tokens).strip()
                if sub_text:
                    sub_chunks.append(sub_text)

        for sub_index, sub_text in enumerate(sub_chunks):
            chunk_id = compute_mdhash_id(
                f"{doc_key}|{node['path']}|{node['title']}|{node['start']}|{node['end']}|{sub_index}|{'summary' if use_summary else 'text'}",
                prefix="chunk-",
            )
            chunks[chunk_id] = {
                "tokens": len(tiktoken_model.encode(sub_text)),
                "content": sub_text,
                "chunk_order_index": global_chunk_order,
                "full_doc_id": doc_key,
                "doc_name": doc_name,
                "structure_leaf_title": node["title"],
                "structure_leaf_node_id": node.get("node_id", ""),
                "structure_leaf_subchunk_index": sub_index,
                "structure_leaf_subchunk_total": len(sub_chunks),
            }
            global_chunk_order += 1

    if use_summary:
        logger.info(
            f"Structured doc detected for {doc_key}: extracted {len(chunks)} summary chunks from JSON structure"
        )
    else:
        logger.info(
            f"Structured doc detected for {doc_key}: extracted {len(chunks)} leaf chunks from JSON structure"
        )
    return chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Summarize the entity or relation description,is used during entity extraction and when merging nodes or edges in the knowledge graph

    Args:
        entity_or_relation_name: entity or relation name
        description: description
        global_config: global configuration
    """
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    section: Union[str, None] = None,
    level: int = 0,
):
    record_type = normalize_extracted_field(record_attributes[0]).lower() if record_attributes else ""
    if len(record_attributes) < 3 or record_type != "entity":
        return None
    # add this record as a node in the G
    entity_name = normalize_extracted_field(record_attributes[1]).upper()
    if not entity_name.strip():
        return None
    if len(record_attributes) >= 4:
        entity_type = normalize_extracted_field(record_attributes[2]).lower()
        entity_description = normalize_extracted_field(record_attributes[3])
    else:
        # Compatibility for malformed output like: ("entity"<|>name<|>type|>description)
        # where '<|>' between type and description is partially broken into '|>'.
        third_attr = normalize_extracted_field(record_attributes[2])
        recovered = None
        for broken_sep in ["|>", "｜>"]:
            if broken_sep in third_attr:
                left, right = third_attr.split(broken_sep, 1)
                left = normalize_extracted_field(left)
                right = normalize_extracted_field(right)
                if left and right:
                    recovered = (left, right)
                    break

        if recovered is not None:
            entity_type, entity_description = recovered[0].lower(), recovered[1]
            logger.warning(
                "Recovered malformed entity tuple by splitting third attribute with broken delimiter. chunk_key=%s, entity_name=%s, raw_third_attr=%s",
                chunk_key,
                entity_name,
                third_attr,
            )
        else:
            # Backward compatibility: older prompts may output only name + description.
            logger.warning(
                "Detected old/invalid entity extraction format with only 3 attributes; fallback entity_type=unknown. chunk_key=%s, entity_name=%s, raw_record=%s",
                chunk_key,
                entity_name,
                json.dumps(record_attributes, ensure_ascii=False),
            )
            entity_type = "unknown"
            entity_description = third_attr
    if not entity_type.strip():
        print(f"\033[93m[Warning] Empty entity_type detected, defaulting to 'unknown'. record_attributes={record_attributes}\033[0m")
        entity_type = "unknown"
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
        section=[section] if isinstance(section, str) and section.strip() else [],
        level=level,
    )


def _normalize_section_ids(section_value) -> list[str]:
    if section_value is None:
        return []
    if isinstance(section_value, list):
        return [str(s).strip() for s in section_value if str(s).strip()]
    if isinstance(section_value, str):
        text = section_value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except Exception:
            pass
        return [text]
    return []


def _compose_doc_section_id(doc_id: Union[str, None], section_id: Union[str, None]) -> str:
    doc_text = str(doc_id or "").strip()
    section_text = str(section_id or "").strip()
    if not section_text:
        return ""
    if not doc_text:
        return section_text
    if section_text.startswith(f"{doc_text}-"):
        return section_text
    return f"{doc_text}-{section_text}"


def _is_extraction_parse_error(
    response_text: str,
    parsed_nodes: dict,
    parsed_edges: dict,
    tuple_delimiter: str,
) -> bool:
    if parsed_nodes or parsed_edges:
        return False

    text = str(response_text or "").strip()
    if not text:
        return True

    lower_text = text.lower()
    # Typical signs that model returned content in a wrong format (e.g., JSON/markdown/narrative).
    if "```" in text or "{" in text or "[" in text:
        return True
    if tuple_delimiter in text and ("(" in text or ")" in text):
        return True
    if "entity" in lower_text or "relationship" in lower_text:
        return True
    return False


async def _llm_extract_with_retry(
    use_llm_func: callable,
    prompt: str,
    tuple_delimiter: str,
    record_delimiter: str,
    completion_delimiter: str,
    chunk_key: str,
    section_node_id: Union[str, None] = None,
    doc_id: Union[str, None] = None,
    max_retries: int = 3,
    initial_response: Union[str, None] = None,
    global_config: Union[dict, None] = None,
    extraction_stage: str = "entity_extraction",
):
    parsed_nodes: dict = {}
    parsed_edges: dict = {}
    attempt_logs: list[dict[str, Any]] = []
    last_response_text = ""

    for attempt in range(1, max_retries + 1):
        if attempt == 1 and initial_response is not None:
            response = initial_response
        else:
            response = await use_llm_func(prompt)
        response_text = str(response or "")
        last_response_text = response_text
        parsed_nodes, parsed_edges = await _parse_extraction_output(
            response,
            tuple_delimiter,
            record_delimiter,
            completion_delimiter,
            chunk_key=chunk_key,
            section_node_id=section_node_id,
            doc_id=doc_id,
        )

        if not _is_extraction_parse_error(response, parsed_nodes, parsed_edges, tuple_delimiter):
            return parsed_nodes, parsed_edges

        attempt_logs.append(
            {
                "attempt": attempt,
                "parsed_node_count": len(parsed_nodes),
                "parsed_edge_count": len(parsed_edges),
                "response_preview": response_text[:2000],
            }
        )

        if attempt < max_retries:
            logger.warning(
                "[Extraction Parse Retry] chunk=%s attempt=%s/%s format parse error, retrying...",
                chunk_key,
                attempt,
                max_retries,
            )
        else:
            logger.warning(
                "[Extraction Parse Retry] chunk=%s exceeded max retries (%s), keeping last parse result",
                chunk_key,
                max_retries,
            )
            if isinstance(global_config, dict):
                _write_extraction_parse_log(
                    global_config,
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "stage": extraction_stage,
                        "chunk_key": chunk_key,
                        "section_node_id": str(section_node_id or ""),
                        "doc_id": str(doc_id or ""),
                        "max_retries": max_retries,
                        "prompt_hash": compute_mdhash_id(prompt, prefix="prompt-"),
                        "parsed_node_count": len(parsed_nodes),
                        "parsed_edge_count": len(parsed_edges),
                        "attempts": attempt_logs,
                        "final_response": last_response_text,
                    },
                )

    return parsed_nodes, parsed_edges


def _debug_abort_if_entity_missing_type_or_section(
    entity_data: dict,
    chunk_key: str,
    record_attributes: list[str],
    section_node_id: Union[str, None],
):
    entity_type = str(entity_data.get("entity_type", "")).strip().lower()
    section_ids = _normalize_section_ids(entity_data.get("section"))
    type_is_missing = entity_type in {"", "unknown", "未知", "unk", "n/a", "na", "none", "null"}
    section_is_missing = len(section_ids) == 0
    if not type_is_missing and not section_is_missing:
        return

    missing_fields = []
    if type_is_missing:
        missing_fields.append("entity_type")
    if section_is_missing:
        missing_fields.append("section")

    logger.error(
        "[Entity Extraction Debug Abort] missing=%s | chunk_key=%s | structure_leaf_node_id=%s",
        ",".join(missing_fields),
        chunk_key,
        str(section_node_id or ""),
    )
    logger.error(
        "[Entity Extraction Debug Abort] raw_record=%s",
        json.dumps(record_attributes, ensure_ascii=False),
    )
    logger.error(
        "[Entity Extraction Debug Abort] entity=%s",
        json.dumps(entity_data, ensure_ascii=False),
    )
    if section_is_missing:
        raise RuntimeError(
            f"Entity extraction failed: missing {','.join(missing_fields)} for entity={entity_data.get('entity_name', '')}"
        )

    logger.warning(
        "[Entity Extraction Debug Warning] Continue with fallback entity_type for entity=%s",
        str(entity_data.get("entity_name", "")),
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    section: Union[str, None] = None,
):
    record_type = normalize_extracted_field(record_attributes[0]).lower() if record_attributes else ""
    if len(record_attributes) < 5 or record_type != "relationship":
        return None
    # add this record as edge
    source = normalize_extracted_field(record_attributes[1]).upper()
    target = normalize_extracted_field(record_attributes[2]).upper()
    edge_description = normalize_extracted_field(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
        section=[section] if isinstance(section, str) and section.strip() else [],
    )


async def _parse_extraction_output(
    final_result: str,
    tuple_delimiter: str,
    record_delimiter: str,
    completion_delimiter: str,
    chunk_key: str,
    section_node_id: Union[str, None] = None,
    doc_id: Union[str, None] = None,
):
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    records = split_string_by_multi_markers(
        final_result,
        [record_delimiter, completion_delimiter],
    )

    normalized_section_id = _compose_doc_section_id(doc_id, section_node_id)

    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        if_entities = await _handle_single_entity_extraction(
            record_attributes, chunk_key, section=normalized_section_id
        )
        if if_entities is not None:
            _debug_abort_if_entity_missing_type_or_section(
                if_entities,
                chunk_key,
                record_attributes,
                normalized_section_id,
            )
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
            continue

        if_relation = await _handle_single_relationship_extraction(
            record_attributes, chunk_key, section=normalized_section_id
        )
        if if_relation is not None:
            maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)

    return dict(maybe_nodes), dict(maybe_edges)


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    input_levels = [int(dp.get("level", 0) or 0) for dp in nodes_data if "level" in dp]
    node_level = max(input_levels) if input_levels else 0
    node_id = _compose_level_aware_node_id(entity_name, node_level)

    already_source_ids = []
    already_description = []
    already_embeddings = []
    already_sections = []
    already_entity_types = []
    already_level = None

    already_node = await knwoledge_graph_inst.get_node(node_id)
    if already_node is not None:                                            # already exist
        already_source_ids.extend(
            split_string_by_multi_markers(str(already_node.get("source_id", "")), [GRAPH_FIELD_SEP])
        )
        already_description.append(str(already_node.get("description", "")))
        if "embedding" in already_node and already_node["embedding"]:
            already_embeddings.append(already_node["embedding"])
        already_sections.extend(_normalize_section_ids(already_node.get("section")))
        already_entity_types.extend(
            [
                t.strip()
                for t in split_string_by_multi_markers(
                    str(already_node.get("entity_type", "")), [GRAPH_FIELD_SEP]
                )
                if t.strip()
            ]
        )
        if "level" in already_node and already_node["level"] is not None:
            try:
                already_level = int(already_node["level"])
            except (ValueError, TypeError):
                already_level = None

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    section_ids = sorted(
        set(
            [s for dp in nodes_data for s in _normalize_section_ids(dp.get("section"))]
            + already_sections
        )
    )

    is_chinese = _is_chinese_language(global_config)

    def _normalize_entity_type(value: object) -> str:
        raw = str(value).strip().lower()
        if not raw:
            return "未知" if is_chinese else "unknown"
        if raw in {"unknown", "unk", "n/a", "na", "none", "null", "未知"}:
            return "未知" if is_chinese else "unknown"
        return str(value).strip().lower()

    entity_types = sorted(
        set(
            [
                _normalize_entity_type(dp.get("entity_type", ""))
                for dp in nodes_data
                if str(dp.get("entity_type", "")).strip()
            ]
            + [_normalize_entity_type(t) for t in already_entity_types if str(t).strip()]
        )
    )
    if not entity_types:
        entity_types = ["未知" if is_chinese else "unknown"]
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )

    embedding = None
    for dp in reversed(nodes_data):
        if "embedding" in dp and dp["embedding"] is not None:
            embedding = dp["embedding"]
            break
    if embedding is None and already_embeddings:
        embedding = already_embeddings[-1]

    # Get the maximum level from all input nodes (keep the highest clustering layer)
    node_level = max(
        ([max(dp.get("level", 0) for dp in nodes_data if "level" in dp)] if any("level" in dp for dp in nodes_data) else [0])
        + ([already_level] if already_level is not None else [])
    )

    node_data = dict(
        entity_name=entity_name,
        entity_type=GRAPH_FIELD_SEP.join(entity_types),
        description=description,
        source_id=source_id,
        section=json.dumps(section_ids, ensure_ascii=False),
        level=str(node_level),
        vectorized="false",
    )
    if embedding is not None:
        if isinstance(embedding, str):
            node_data["embedding"] = embedding
        else:
            node_data["embedding"] = json.dumps([float(v) for v in embedding])

    await knwoledge_graph_inst.upsert_node(
        node_id,
        node_data=node_data,
    )
    node_data["id"] = node_id
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_levels = [int(dp.get("level", 0) or 0) for dp in edges_data if "level" in dp]
    edge_level = max(edge_levels) if edge_levels else 0
    src_node_id = _compose_level_aware_node_id(src_id, edge_level)
    tgt_node_id = _compose_level_aware_node_id(tgt_id, edge_level)

    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    already_sections = []
    if await knwoledge_graph_inst.has_edge(src_node_id, tgt_node_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_node_id, tgt_node_id)
        if already_edge is not None:
            already_weights.append(float(already_edge.get("weight", 0.0) or 0.0))
            already_source_ids.extend(
                split_string_by_multi_markers(str(already_edge.get("source_id", "")), [GRAPH_FIELD_SEP])
            )
            already_description.append(str(already_edge.get("description", "")))
            already_order.append(already_edge.get("order", 1))

    section_ids = sorted(
        set(
            [s for dp in edges_data for s in _normalize_section_ids(dp.get("section"))]
            + already_sections
        )
    )

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    is_chinese = _is_chinese_language(global_config)
    unknown_type = "未知" if is_chinese else "unknown"
    for need_insert_id, need_insert_name in [(src_node_id, src_id), (tgt_node_id, tgt_id)]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "entity_name": need_insert_name,
                    "entity_type": unknown_type,
                    "source_id": source_id,
                    "description": description,
                    "section": json.dumps(section_ids, ensure_ascii=False),
                    "level": str(edge_level),
                    "vectorized": "false",
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_node_id,
        tgt_node_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
            section=json.dumps(section_ids, ensure_ascii=False),
            order=order,
            vectorized="false",
        ),
    )


def _is_chinese_language(global_config: dict) -> bool:
    language = str(global_config.get("language", "english")).strip().lower()
    return language in {"chinese", "zh", "中文"}


def _normalize_query_term_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [value.strip() for value in re.split(r"[\n,;，；]+", values) if value.strip()]
    if isinstance(values, list):
        normalized = []
        for value in values:
            text = str(value).strip()
            if text:
                normalized.append(text)
        return normalized
    return []


def _compose_level_aware_node_id(entity_name: str, node_level: int) -> str:
    return f"{entity_name}::level::{node_level}"


def _compute_structure_node_levels(nodes: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Compute structure level with leaf as level 1 and parents incrementing upward."""
    level_cache: dict[str, int] = {}

    def _dfs(node_id: str) -> int:
        if node_id in level_cache:
            return level_cache[node_id]
        node = nodes.get(node_id, {})
        children = node.get("children", []) if isinstance(node, dict) else []
        child_ids = [str(cid).strip() for cid in children if str(cid).strip() in nodes]
        if not child_ids:
            level_cache[node_id] = 1
            return 1
        level_cache[node_id] = 1 + max(_dfs(cid) for cid in child_ids)
        return level_cache[node_id]

    for node_id in nodes.keys():
        _dfs(node_id)
    return level_cache


def _split_level_aware_node_id(node_id: str) -> tuple[str, int | None]:
    marker = "::level::"
    if marker not in node_id:
        return node_id, None
    entity_name, raw_level = node_id.rsplit(marker, 1)
    try:
        return entity_name, int(raw_level)
    except (TypeError, ValueError):
        return entity_name, None


def _is_vectorized_flag_true(vectorized_value: object) -> bool:
    return str(vectorized_value or "").strip().lower() in {"1", "true", "yes", "y"}


async def _mark_graph_records_vectorized(
    knowledge_graph_inst: BaseGraphStorage,
    entity_names: list[str],
    edge_pairs: list[tuple[str, str]],
    level: int,
) -> None:
    entity_tasks = [
        knowledge_graph_inst.upsert_node(
            _compose_level_aware_node_id(entity_name, level),
            {"vectorized": "true"},
        )
        for entity_name in entity_names
        if str(entity_name).strip()
    ]
    edge_tasks = [
        knowledge_graph_inst.upsert_edge(
            _compose_level_aware_node_id(src_id, level),
            _compose_level_aware_node_id(tgt_id, level),
            {"vectorized": "true"},
        )
        for src_id, tgt_id in edge_pairs
        if str(src_id).strip() and str(tgt_id).strip()
    ]
    if entity_tasks:
        await asyncio.gather(*entity_tasks)
    if edge_tasks:
        await asyncio.gather(*edge_tasks)


def _vdb_hit_kind(hit: dict) -> str:
    return str(hit.get("kind", "entity") or "entity").strip().lower() or "entity"


def _filter_vdb_hits_by_kind(results: list[dict], kind: str) -> list[dict]:
    normalized_kind = str(kind).strip().lower()
    filtered = []
    for hit in results:
        if _vdb_hit_kind(hit) == normalized_kind:
            filtered.append({**hit, "kind": normalized_kind})
    return filtered


def _dedupe_vdb_hits(results: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for hit in results:
        dedupe_key = (
            _vdb_hit_kind(hit),
            hit.get("id") or hit.get("__id__"),
            hit.get("entity_name"),
            hit.get("source"),
            hit.get("target"),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(hit)
    return deduped


async def _query_mixed_vdb(
    entities_vdb: BaseVectorStorage,
    query_terms: list[str],
    top_k: int,
    kind: str | None = None,
) -> list[dict]:
    hits: list[dict] = []
    logger.info(
        "[Dual][Vector] Start mixed VDB query | terms=%s | top_k=%s | kind=%s",
        len(query_terms),
        top_k,
        kind or "all",
    )
    for term in query_terms:
        if not term:
            continue
        results = await entities_vdb.query(term, top_k=top_k)
        logger.info(
            "[Dual][Vector] term=%s | raw_hits=%s",
            term,
            len(results),
        )
        if kind is not None:
            results = _filter_vdb_hits_by_kind(results, kind)
            logger.info(
                "[Dual][Vector] term=%s | filtered_hits(kind=%s)=%s",
                term,
                kind,
                len(results),
            )
        hits.extend(results)
    deduped = _dedupe_vdb_hits(hits)
    logger.info(
        "[Dual][Vector] Finished mixed VDB query | total_hits=%s | deduped_hits=%s",
        len(hits),
        len(deduped),
    )
    return deduped


def _normalize_keyword_relations(values: Any) -> list[dict[str, str]]:
    if values is None:
        return []
    relations: list[dict[str, str]] = []

    if isinstance(values, list):
        for item in values:
            head = ""
            tail = ""
            if isinstance(item, dict):
                head = str(item.get("head", "")).strip()
                tail = str(item.get("tail", "")).strip()
            else:
                text = str(item).strip()
                for marker in ["->", "→", "=>", "-"]:
                    if marker in text:
                        left, right = text.split(marker, 1)
                        head = str(left).strip()
                        tail = str(right).strip()
                        break
            if head and tail:
                relations.append({"head": head, "tail": tail})
        return relations

    text = str(values).strip()
    if not text:
        return []
    for row in re.split(r"[\n;；]+", text):
        row = row.strip()
        if not row:
            continue
        for marker in ["->", "→", "=>", "-"]:
            if marker in row:
                left, right = row.split(marker, 1)
                head = str(left).strip()
                tail = str(right).strip()
                if head and tail:
                    relations.append({"head": head, "tail": tail})
                break
    return relations


def _resolve_dual_runtime_config(query_param: QueryParam, global_config: dict) -> dict[str, Any]:
    def _int_cfg(key: str, default: int) -> int:
        gp = global_config.get(key, None)
        qp = getattr(query_param, key, None)
        for value in [gp, qp, default]:
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                continue
        return default

    def _bool_cfg(key: str, default: bool) -> bool:
        gp = global_config.get(key, None)
        qp = getattr(query_param, key, None)
        for value in [qp, gp, default]:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                if value.strip().lower() in {"1", "true", "yes", "y", "on"}:
                    return True
                if value.strip().lower() in {"0", "false", "no", "n", "off"}:
                    return False
        return default

    return {
        "enable_tree_traversal": _bool_cfg("enable_tree_traversal", False),
        "detail_top_k_entity": _int_cfg("detail_top_k_entity", 10),
        "detail_top_k_relation": _int_cfg("detail_top_k_relation", 10),
        "macro_top_k_entity": _int_cfg("macro_top_k_entity", 10),
        "macro_top_k_relation": _int_cfg("macro_top_k_relation", 10),
        "keyword_top_k_entity": _int_cfg("keyword_top_k_entity", 10),
        "keyword_top_k_relation": _int_cfg("keyword_top_k_relation", 10),
        "max_token_for_text_unit": _int_cfg("max_token_for_text_unit", 20000),
        "max_token_for_graph": _int_cfg("max_token_for_graph", 20000),
        "max_token_for_summary": _int_cfg("max_token_for_summary", 20000),
    }


async def _decompose_dual_query(
    query: str,
    global_config: dict,
    enable_tree_traversal: bool,
) -> dict[str, Any]:
    use_llm_func: callable = global_config["cheap_model_func"]
    prompt_key = (
        "dual_query_decomposition"
        if enable_tree_traversal
        else "dual_query_decomposition_with_keyword_relations"
    )
    prompt = PROMPTS[prompt_key].format(query=query)
    response = await use_llm_func(prompt)
    parser = global_config.get("convert_response_to_json_func")
    parsed = parser(response) if callable(parser) else None
    if not isinstance(parsed, dict):
        parsed = {}
    decomposition = {
        "detail_subquestions": _normalize_query_term_list(parsed.get("detail_subquestions")),
        "macro_subquestions": _normalize_query_term_list(parsed.get("macro_subquestions")),
        "keywords": _normalize_query_term_list(parsed.get("keywords")),
        "keyword_relations": _normalize_keyword_relations(parsed.get("keyword_relations")),
    }
    logger.info(
        "[Dual][Decompose] tree=%s | detail=%s | macro=%s | keywords=%s | keyword_relations=%s",
        enable_tree_traversal,
        len(decomposition["detail_subquestions"]),
        len(decomposition["macro_subquestions"]),
        len(decomposition["keywords"]),
        len(decomposition["keyword_relations"]),
    )
    logger.debug("[Dual][Decompose] payload=%s", json.dumps(decomposition, ensure_ascii=False))
    return decomposition


async def _match_graph_nodes_by_keywords(
    knowledge_graph_inst: BaseGraphStorage,
    keywords: list[str],
    top_k: int,
) -> list[dict]:
    if not keywords:
        logger.info("[Dual][Keyword] Skip keyword match because keywords is empty")
        return []
    all_nodes = await knowledge_graph_inst.get_all_nodes()
    if not all_nodes:
        logger.info("[Dual][Keyword] Skip keyword match because graph has no nodes")
        return []
    normalized_keywords = [keyword.lower() for keyword in keywords if keyword.strip()]
    matched = []
    for node in all_nodes:
        node_id = str(node.get("id", "")).strip()
        node_name = str(node.get("entity_name", node_id)).strip()
        # Case-insensitive substring match: entity name contains any keyword.
        entity_name_text = node_name.lower()
        if any(keyword in entity_name_text for keyword in normalized_keywords):
            matched.append({"entity_name": node_name or node_id, **node})
    if not matched:
        return []
    degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(node["entity_name"]) for node in matched]
    )
    matched = [
        {**node, "rank": degree}
        for node, degree in zip(matched, degrees)
        if node.get("entity_name")
    ]
    matched = sorted(
        matched,
        key=lambda x: (x["rank"], len(str(x.get("description", "")))),
        reverse=True,
    )
    top_nodes = matched[:top_k]
    logger.info(
        "[Dual][Keyword] keywords=%s | matched=%s | top_k=%s",
        keywords,
        len(matched),
        len(top_nodes),
    )
    logger.info(
        "[Dual][Keyword] top_entities=%s",
        [str(node.get("entity_name", "")).strip() for node in top_nodes],
    )
    return top_nodes


def _format_structure_sections_for_prompt(sections: list[dict]) -> str:
    return json.dumps(sections, ensure_ascii=False, indent=2)


def _safe_parse_json_response(response: str, global_config: dict) -> dict:
    parser = global_config.get("convert_response_to_json_func")
    parsed = parser(response) if callable(parser) else None
    if isinstance(parsed, dict):
        return parsed
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _load_structured_docs_from_store(
    full_docs_db: BaseKVStorage,
) -> list[dict[str, Any]]:
    doc_ids = await full_docs_db.all_keys()
    if not doc_ids:
        return []
    raw_docs = await full_docs_db.get_by_ids(doc_ids)
    docs = {
        doc_id: doc_value
        for doc_id, doc_value in zip(doc_ids, raw_docs)
        if isinstance(doc_value, dict)
    }
    return _extract_structured_nodes_from_docs(docs)


def _structured_doc_root_ids(structured_doc: dict[str, Any]) -> list[str]:
    nodes = structured_doc.get("nodes", {})
    return [
        node_id
        for node_id, node in nodes.items()
        if node.get("parent_id") is None
    ]


def _structure_sections_at_level(
    structured_doc: dict[str, Any],
    node_ids: list[str],
) -> list[dict[str, str]]:
    nodes = structured_doc.get("nodes", {})
    sections = []
    for node_id in node_ids:
        node = nodes.get(node_id)
        if not node:
            continue
        summary = str(node.get("summary", "")).strip()
        if not summary:
            summary = str(node.get("text", "")).strip()[:500]
        sections.append(
            {
                "node_id": node_id,
                "title": str(node.get("title", node_id)).strip() or node_id,
                "summary": summary,
            }
        )
    return sections


def _structure_leaf_sections_with_text_at_level(
    structured_doc: dict[str, Any],
    node_ids: list[str],
) -> list[dict[str, str]]:
    nodes = structured_doc.get("nodes", {})
    sections = []
    for node_id in node_ids:
        node = nodes.get(node_id)
        if not node:
            continue
        summary = str(node.get("summary", "")).strip()
        text = str(node.get("text", "")).strip()
        if not text:
            text = summary
        sections.append(
            {
                "node_id": node_id,
                "title": str(node.get("title", node_id)).strip() or node_id,
                "summary": summary,
                "text": text,
            }
        )
    return sections


async def _collect_detail_evidence_from_leaf_sections(
    leaf_sections: list[dict[str, str]],
    detail_subquestions: list[str],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    if not leaf_sections or not detail_subquestions:
        logger.info(
            "[Dual][Detail] Skip detail evidence | leaf_sections=%s | detail_subquestions=%s",
            len(leaf_sections),
            len(detail_subquestions),
        )
        return ""

    use_llm_func: callable = global_config["cheap_model_func"]
    tiktoken_model_name = str(global_config.get("tiktoken_model_name", "gpt-4o")).strip() or "gpt-4o"
    max_section_tokens = max(
        512,
        int(
            getattr(
                query_param,
                "max_token_for_section_text",
                getattr(query_param, "max_token_for_detail_qa", 10086),
            )
            or 10086
        ),
    )
    logger.info(
        "[Dual][Detail] Start detail evidence | leaf_sections=%s | detail_subquestions=%s | max_section_tokens=%s",
        len(leaf_sections),
        len(detail_subquestions),
        max_section_tokens,
    )

    async def _ask_section(section: dict[str, str]) -> str:
        section_text = str(section.get("text", "")).strip()
        if not section_text:
            logger.info(
                "[Dual][Detail] Skip empty section text | section_id=%s",
                str(section.get("node_id", "")).strip(),
            )
            return ""
        section_tokens = encode_string_by_tiktoken(section_text, model_name=tiktoken_model_name)
        section_id_for_log = str(section.get("node_id", "")).strip()
        original_token_count = len(section_tokens)
        if len(section_tokens) > max_section_tokens:
            section_text = decode_tokens_by_tiktoken(
                section_tokens[:max_section_tokens],
                model_name=tiktoken_model_name,
            )
        logger.info(
            "[Dual][Detail] Query section | section_id=%s | original_tokens=%s | used_tokens=%s",
            section_id_for_log,
            original_token_count,
            min(original_token_count, max_section_tokens),
        )

        prompt = PROMPTS["dual_leaf_detail_question_answer"].format(
            section_id=section.get("node_id", ""),
            section_title=section.get("title", section.get("node_id", "")),
            section_text=section_text,
            detail_questions=json.dumps(detail_subquestions, ensure_ascii=False, indent=2),
        )
        parsed = _safe_parse_json_response(await use_llm_func(prompt), global_config)
        answers = parsed.get("answers", []) if isinstance(parsed.get("answers", []), list) else []
        useful_information = str(parsed.get("useful_information", "")).strip()
        if not useful_information and answers:
            useful_information = "\n".join(
                [
                    f"Q: {str(item.get('question', '')).strip()}\nA: {str(item.get('answer', '')).strip()}"
                    for item in answers
                    if isinstance(item, dict)
                ]
            ).strip()
        if not useful_information:
            logger.info(
                "[Dual][Detail] No useful info from section | section_id=%s",
                section_id_for_log,
            )
            return ""
        logger.info(
            "[Dual][Detail] Section answered | section_id=%s | answers=%s | useful_info_chars=%s",
            section_id_for_log,
            len(answers),
            len(useful_information),
        )

        section_id = str(parsed.get("section_id", section.get("node_id", ""))).strip() or section.get("node_id", "")
        section_title = str(parsed.get("section_title", section.get("title", section_id))).strip() or section.get("title", section_id)
        return (
            f"[Leaf Section]\n"
            f"section_id: {section_id}\n"
            f"section_title: {section_title}\n"
            f"useful_information:\n{useful_information}"
        )

    section_results = await asyncio.gather(*[_ask_section(section) for section in leaf_sections])
    section_results = [result.strip() for result in section_results if result and result.strip()]
    logger.info(
        "[Dual][Detail] Finished detail evidence | non_empty_sections=%s",
        len(section_results),
    )
    return "\n\n".join(section_results).strip()


async def _collect_macro_evidence_from_structure(
    full_docs_db: BaseKVStorage,
    macro_subquestions: list[str],
    detail_subquestions: list[str],
    query_param: QueryParam,
    global_config: dict,
) -> list[dict[str, Any]]:
    if not macro_subquestions and not detail_subquestions:
        logger.info("[Dual][Macro] Skip macro evidence because macro/detail subquestions are empty")
        return []

    structured_docs = await _load_structured_docs_from_store(full_docs_db)
    if not structured_docs:
        logger.info("[Dual][Macro] Skip macro evidence because structured docs are empty")
        return []

    use_llm_func: callable = global_config["cheap_model_func"]
    results: list[dict[str, Any]] = []

    def _merge_useful_information(base: str, additions: list[str]) -> str:
        merged_parts: list[str] = []
        base_text = str(base or "").strip()
        if base_text and base_text != "NONE":
            merged_parts.append(base_text)
        for item in additions:
            item_text = str(item or "").strip()
            if not item_text or item_text == "NONE":
                continue
            if item_text not in merged_parts:
                merged_parts.append(item_text)
        return "\n\n".join(merged_parts) if merged_parts else "NONE"

    for macro_subquestion in macro_subquestions:
        logger.info("[Dual][Macro] Start macro_subquestion=%s", macro_subquestion)
        # ---- Layer 0: document selection ----
        if len(structured_docs) == 1:
            candidate_docs = structured_docs
        else:
            doc_list_for_prompt = [
                {
                    "doc_id": sd.get("doc_id", ""),
                    "doc_name": sd.get("doc_name", ""),
                    "doc_description": sd.get("doc_description", ""),
                }
                for sd in structured_docs
            ]
            doc_list_for_prompt = truncate_list_by_token_size(
                doc_list_for_prompt,
                key=lambda x: f"{x.get('doc_id', '')} {x.get('doc_name', '')} {x.get('doc_description', '')}",
                max_token_size=max(512, getattr(query_param, "max_token_for_macro_traversal", 2048) // 2),
            )
            doc_sel_prompt = PROMPTS["dual_macro_doc_selection"].format(
                macro_subquestion=macro_subquestion,
                documents=_format_structure_sections_for_prompt(doc_list_for_prompt),
            )
            doc_sel_parsed = _safe_parse_json_response(
                await use_llm_func(doc_sel_prompt), global_config
            )
            selected_doc_ids = {
                str(d).strip()
                for d in doc_sel_parsed.get("doc_ids", [])
                if str(d).strip()
            }
            candidate_docs = (
                [sd for sd in structured_docs if sd.get("doc_id", "") in selected_doc_ids]
                if selected_doc_ids
                else []
            )
        logger.info(
            "[Dual][Macro] doc_selection | macro_subquestion=%s | candidates=%s",
            macro_subquestion,
            len(candidate_docs),
        )

        # ---- Layer 1+: section traversal within each selected document ----
        for structured_doc in candidate_docs:
            current_node_ids = _structured_doc_root_ids(structured_doc)
            useful_information = "NONE"
            depth = 0
            final_status = "leaf"
            final_leaf_node_ids: list[str] = []
            logger.info(
                "[Dual][Macro] Start doc traversal | doc=%s | root_nodes=%s",
                structured_doc.get("doc_name", structured_doc.get("doc_id", "")),
                len(current_node_ids),
            )

            while current_node_ids:
                current_sections = _structure_sections_at_level(structured_doc, current_node_ids)
                current_sections = truncate_list_by_token_size(
                    current_sections,
                    key=lambda x: f"{x.get('node_id', '')} {x.get('title', '')} {x.get('summary', '')}",
                    max_token_size=max(
                        256,
                        getattr(
                            query_param,
                            "max_token_for_section_summary",
                            getattr(query_param, "max_token_for_macro_traversal", 4096),
                        ),
                    ),
                )
                if not current_sections:
                    break

                current_level_ids = {section["node_id"] for section in current_sections}

                async def _traverse_one_section(section: dict[str, str]) -> dict[str, Any]:
                    section_node_id = str(section.get("node_id", "")).strip()
                    prompt = PROMPTS["dual_macro_traversal"].format(
                        doc_name=structured_doc.get("doc_name", structured_doc.get("doc_id", "")),
                        doc_description=structured_doc.get("doc_description", ""),
                        macro_subquestion=macro_subquestion,
                        useful_information=useful_information,
                        current_sections=_format_structure_sections_for_prompt([section]),
                    )
                    parsed = _safe_parse_json_response(await use_llm_func(prompt), global_config)
                    selected_section_ids = [
                        str(node_id).strip()
                        for node_id in parsed.get("section_ids", [])
                        if str(node_id).strip() == section_node_id
                    ]
                    return {
                        "sufficient": bool(parsed.get("sufficient", False)),
                        "selected_section_ids": selected_section_ids,
                        "useful_information": str(parsed.get("useful_information", "")).strip(),
                    }

                level_results = await asyncio.gather(
                    *[_traverse_one_section(section) for section in current_sections]
                )

                useful_information = _merge_useful_information(
                    useful_information,
                    [result.get("useful_information", "") for result in level_results],
                )

                selected_section_ids: list[str] = []
                sufficient = False
                for result in level_results:
                    if result.get("sufficient", False):
                        sufficient = True
                    for section_id in result.get("selected_section_ids", []):
                        if section_id not in selected_section_ids:
                            selected_section_ids.append(section_id)
                logger.info(
                    "[Dual][Macro] level_result | doc=%s | depth=%s | sections=%s | selected=%s | sufficient=%s | useful_info_chars=%s",
                    structured_doc.get("doc_name", structured_doc.get("doc_id", "")),
                    depth,
                    len(current_sections),
                    len(selected_section_ids),
                    sufficient,
                    len(useful_information) if useful_information and useful_information != "NONE" else 0,
                )

                if sufficient:
                    final_status = "sufficient"
                    final_leaf_node_ids = selected_section_ids if selected_section_ids else list(current_level_ids)
                    break

                next_node_ids = []
                for section_id in selected_section_ids:
                    node = structured_doc.get("nodes", {}).get(section_id, {})
                    child_ids = node.get("children", [])
                    for child_id in child_ids:
                        if child_id not in next_node_ids:
                            next_node_ids.append(child_id)

                if not next_node_ids:
                    final_status = "leaf"
                    final_leaf_node_ids = selected_section_ids if selected_section_ids else list(current_level_ids)
                    break

                current_node_ids = next_node_ids
                depth += 1

            if detail_subquestions and final_leaf_node_ids:
                leaf_sections = _structure_leaf_sections_with_text_at_level(structured_doc, final_leaf_node_ids)
                leaf_detail_information = await _collect_detail_evidence_from_leaf_sections(
                    leaf_sections=leaf_sections,
                    detail_subquestions=detail_subquestions,
                    query_param=query_param,
                    global_config=global_config,
                )
                if leaf_detail_information:
                    useful_information = (
                        f"{useful_information}\n\n{leaf_detail_information}"
                        if useful_information and useful_information != "NONE"
                        else leaf_detail_information
                    )

            if useful_information and useful_information != "NONE":
                logger.info(
                    "[Dual][Macro] End doc traversal | doc=%s | status=%s | depth=%s | final_leaf_nodes=%s | useful_info_chars=%s",
                    structured_doc.get("doc_name", structured_doc.get("doc_id", "")),
                    final_status,
                    depth,
                    len(final_leaf_node_ids),
                    len(useful_information),
                )
                results.append(
                    {
                        "macro_subquestion": macro_subquestion,
                        "doc_id": structured_doc.get("doc_id", ""),
                        "doc_name": structured_doc.get("doc_name", ""),
                        "status": final_status,
                        "depth": depth,
                        "useful_information": useful_information,
                    }
                )

    truncated_results = truncate_list_by_token_size(
        results,
        key=lambda x: f"{x.get('macro_subquestion', '')} {x.get('doc_name', '')} {x.get('useful_information', '')}",
        max_token_size=getattr(query_param, "max_token_for_macro_traversal", 4096),
    )
    logger.info(
        "[Dual][Macro] Finished macro evidence | raw_results=%s | truncated_results=%s",
        len(results),
        len(truncated_results),
    )
    return truncated_results


def _build_dual_context_section(title: str, rows: list[list[Any]]) -> str:
    return f"""
-----{title}-----
```csv
{list_of_list_to_csv(rows)}
```
"""


async def _build_dual_query_context(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    full_docs_db: BaseKVStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str | None:
    logger.info("[Dual] Start building dual query context")
    tiktoken_model_name = str(global_config.get("tiktoken_model_name", "gpt-4o")).strip() or "gpt-4o"
    runtime_cfg = _resolve_dual_runtime_config(query_param, global_config)
    enable_tree_traversal = bool(runtime_cfg["enable_tree_traversal"])
    decomposition = await _decompose_dual_query(
        query,
        global_config,
        enable_tree_traversal=enable_tree_traversal,
    )

    detail_terms = decomposition["detail_subquestions"] or [query]
    macro_terms = decomposition["macro_subquestions"] or [query]

    detail_hits = await _query_mixed_vdb(
        entities_vdb,
        detail_terms,
        top_k=max(runtime_cfg["detail_top_k_entity"], runtime_cfg["detail_top_k_relation"]),
    )
    macro_hits = await _query_mixed_vdb(
        entities_vdb,
        macro_terms,
        top_k=max(runtime_cfg["macro_top_k_entity"], runtime_cfg["macro_top_k_relation"]),
    )

    detail_entity_hits = _filter_vdb_hits_by_kind(detail_hits, "entity")[: runtime_cfg["detail_top_k_entity"]]
    detail_relation_hits = _filter_vdb_hits_by_kind(detail_hits, "relation")[: runtime_cfg["detail_top_k_relation"]]
    macro_entity_hits = _filter_vdb_hits_by_kind(macro_hits, "entity")[: runtime_cfg["macro_top_k_entity"]]
    macro_relation_hits = _filter_vdb_hits_by_kind(macro_hits, "relation")[: runtime_cfg["macro_top_k_relation"]]

    all_nodes = await knowledge_graph_inst.get_all_nodes()
    node_lookup_by_exact_id: dict[str, dict] = {}
    node_lookup_by_name: dict[str, list[dict]] = defaultdict(list)
    for node in all_nodes:
        node_id = str(node.get("id", "")).strip()
        if node_id:
            node_lookup_by_exact_id[node_id] = node
        entity_name = str(node.get("entity_name", "")).strip()
        if entity_name:
            node_lookup_by_name[entity_name.lower()].append(node)

    async def _resolve_entity_hit(hit: dict, source_tag: str) -> dict[str, Any] | None:
        entity_name = str(hit.get("entity_name", "")).strip()
        if not entity_name:
            return None
        level_text = str(hit.get("level", "")).strip()
        candidate_nodes: list[dict] = []
        if level_text and level_text.lstrip("-").isdigit():
            node_id = _compose_level_aware_node_id(entity_name, int(level_text))
            node = node_lookup_by_exact_id.get(node_id)
            if node is not None:
                candidate_nodes.append(node)
        if not candidate_nodes:
            candidate_nodes.extend(node_lookup_by_name.get(entity_name.lower(), []))
        if not candidate_nodes:
            fallback = await knowledge_graph_inst.get_node(entity_name)
            if isinstance(fallback, dict):
                candidate_nodes.append({**fallback, "id": entity_name, "entity_name": entity_name})
        if not candidate_nodes:
            return None

        best_node = candidate_nodes[0]
        if len(candidate_nodes) > 1:
            best_node = max(candidate_nodes, key=lambda x: int(str(x.get("level", 0)).strip() or 0))
        node_id = str(best_node.get("id", _compose_level_aware_node_id(entity_name, 0))).strip()
        degree = await knowledge_graph_inst.node_degree(node_id)
        return {
            "entity": str(best_node.get("entity_name", entity_name)).strip() or entity_name,
            "node_id": node_id,
            "description": str(best_node.get("description", "UNKNOWN")),
            "entity_type": str(best_node.get("entity_type", "unknown")),
            "source_id": str(best_node.get("source_id", "")),
            "section": str(best_node.get("section", "")),
            "level": str(best_node.get("level", "0")),
            "rank": degree,
            "sources": source_tag,
            "kind": "entity",
        }

    async def _resolve_relation_hit(hit: dict, source_tag: str) -> dict[str, Any] | None:
        src_name = str(hit.get("source", "")).strip()
        tgt_name = str(hit.get("target", "")).strip()
        if not src_name or not tgt_name:
            return None

        candidate_pairs: list[tuple[str, str]] = []
        level_text = str(hit.get("level", "")).strip()
        if level_text and level_text.lstrip("-").isdigit():
            level_value = int(level_text)
            candidate_pairs.append(
                (
                    _compose_level_aware_node_id(src_name, level_value),
                    _compose_level_aware_node_id(tgt_name, level_value),
                )
            )
        for src_node in node_lookup_by_name.get(src_name.lower(), []):
            for tgt_node in node_lookup_by_name.get(tgt_name.lower(), []):
                src_id = str(src_node.get("id", "")).strip()
                tgt_id = str(tgt_node.get("id", "")).strip()
                if src_id and tgt_id:
                    candidate_pairs.append((src_id, tgt_id))

        dedup_pairs = []
        seen = set()
        for src_id, tgt_id in candidate_pairs:
            key = (src_id, tgt_id)
            if key in seen:
                continue
            seen.add(key)
            dedup_pairs.append((src_id, tgt_id))

        best: dict[str, Any] | None = None
        for src_id, tgt_id in dedup_pairs:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data is None:
                edge_data = await knowledge_graph_inst.get_edge(tgt_id, src_id)
                if edge_data is None:
                    continue
                src_id, tgt_id = tgt_id, src_id
            rank = await knowledge_graph_inst.edge_degree(src_id, tgt_id)
            candidate = {
                "source": src_name,
                "target": tgt_name,
                "src_node_id": src_id,
                "tgt_node_id": tgt_id,
                "description": str(edge_data.get("description", hit.get("description", ""))),
                "source_id": str(edge_data.get("source_id", "")),
                "section": str(edge_data.get("section", "")),
                "weight": float(edge_data.get("weight", 1.0) or 1.0),
                "rank": rank,
                "sources": source_tag,
                "kind": "relation",
            }
            if best is None or candidate["rank"] > best["rank"]:
                best = candidate
        return best

    async def _match_keyword_entities(keywords: list[str]) -> list[dict[str, Any]]:
        if not keywords:
            return []
        keyword_nodes = await _match_graph_nodes_by_keywords(
            knowledge_graph_inst,
            keywords,
            top_k=max(1, runtime_cfg["keyword_top_k_entity"]),
        )
        results = []
        for node in keyword_nodes[: runtime_cfg["keyword_top_k_entity"]]:
            entity_name = str(node.get("entity_name", node.get("id", ""))).strip()
            node_id = str(node.get("id", "")).strip() or _compose_level_aware_node_id(entity_name, 0)
            if not entity_name:
                continue
            results.append(
                {
                    "entity": entity_name,
                    "node_id": node_id,
                    "description": str(node.get("description", "UNKNOWN")),
                    "entity_type": str(node.get("entity_type", "unknown")),
                    "source_id": str(node.get("source_id", "")),
                    "section": str(node.get("section", "")),
                    "level": str(node.get("level", "0")),
                    "rank": int(node.get("rank", 0)),
                    "sources": "keyword-single",
                    "kind": "entity",
                }
            )
        return results

    async def _match_keyword_relation_paths(
        keyword_relations: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not keyword_relations:
            return [], []
        graph = getattr(knowledge_graph_inst, "_graph", None)
        if graph is None:
            logger.warning("[Dual][KeywordPath] Skip keyword relation path search: graph backend has no _graph")
            return [], []

        def _find_candidate_nodes(keyword: str) -> list[dict]:
            term = str(keyword).strip().lower()
            if not term:
                return []
            candidates = []
            for node in all_nodes:
                name = str(node.get("entity_name", "")).strip().lower()
                if term in name:
                    candidates.append(node)
            candidates = sorted(
                candidates,
                key=lambda x: (
                    int(str(x.get("level", 0)).strip() or 0),
                    len(str(x.get("description", ""))),
                ),
                reverse=True,
            )
            return candidates[: max(3, runtime_cfg["keyword_top_k_entity"])]

        keyword_entities: list[dict[str, Any]] = []
        keyword_rel_edges: list[dict[str, Any]] = []

        for rel in keyword_relations[: runtime_cfg["keyword_top_k_relation"]]:
            head = str(rel.get("head", "")).strip()
            tail = str(rel.get("tail", "")).strip()
            if not head or not tail:
                continue
            head_candidates = _find_candidate_nodes(head)
            tail_candidates = _find_candidate_nodes(tail)
            best_path: list[str] | None = None
            best_pair: tuple[str, str] | None = None

            for head_node in head_candidates:
                for tail_node in tail_candidates:
                    head_id = str(head_node.get("id", "")).strip()
                    tail_id = str(tail_node.get("id", "")).strip()
                    if not head_id or not tail_id:
                        continue
                    try:
                        path = nx.shortest_path(graph, source=head_id, target=tail_id)
                    except Exception:
                        continue
                    if best_path is None or len(path) < len(best_path):
                        best_path = path
                        best_pair = (head_id, tail_id)

            if not best_path or not best_pair:
                continue

            for node_id in best_path:
                node = node_lookup_by_exact_id.get(node_id)
                if node is None:
                    continue
                keyword_entities.append(
                    {
                        "entity": str(node.get("entity_name", "")).strip() or node_id,
                        "node_id": node_id,
                        "description": str(node.get("description", "UNKNOWN")),
                        "entity_type": str(node.get("entity_type", "unknown")),
                        "source_id": str(node.get("source_id", "")),
                        "section": str(node.get("section", "")),
                        "level": str(node.get("level", "0")),
                        "rank": max(0, len(best_path) - 1),
                        "sources": f"keyword-path:{head}->{tail}",
                        "kind": "entity",
                    }
                )

            for i in range(len(best_path) - 1):
                src_id = best_path[i]
                tgt_id = best_path[i + 1]
                edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
                if edge_data is None:
                    edge_data = await knowledge_graph_inst.get_edge(tgt_id, src_id)
                    if edge_data is None:
                        continue
                src_node = node_lookup_by_exact_id.get(src_id, {})
                tgt_node = node_lookup_by_exact_id.get(tgt_id, {})
                src_name = str(src_node.get("entity_name", src_id)).strip() or src_id
                tgt_name = str(tgt_node.get("entity_name", tgt_id)).strip() or tgt_id
                keyword_rel_edges.append(
                    {
                        "source": src_name,
                        "target": tgt_name,
                        "src_node_id": src_id,
                        "tgt_node_id": tgt_id,
                        "description": str(edge_data.get("description", "")),
                        "source_id": str(edge_data.get("source_id", "")),
                        "section": str(edge_data.get("section", "")),
                        "weight": float(edge_data.get("weight", 1.0) or 1.0),
                        "rank": max(0, len(best_path) - 1),
                        "sources": f"keyword-path:{head}->{tail}",
                        "kind": "relation",
                    }
                )

        return keyword_entities, keyword_rel_edges

    def _merge_entity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for record in records:
            node_id = str(record.get("node_id", "")).strip()
            if not node_id:
                continue
            if node_id not in merged:
                merged[node_id] = {**record, "sources": [str(record.get("sources", "")).strip()]}
                continue
            existing = merged[node_id]
            existing["rank"] = max(float(existing.get("rank", 0) or 0), float(record.get("rank", 0) or 0))
            if str(existing.get("description", "")).strip() in {"", "UNKNOWN"}:
                existing["description"] = record.get("description", "UNKNOWN")
            source_tag = str(record.get("sources", "")).strip()
            if source_tag and source_tag not in existing["sources"]:
                existing["sources"].append(source_tag)
            if not str(existing.get("source_id", "")).strip() and str(record.get("source_id", "")).strip():
                existing["source_id"] = record.get("source_id", "")
            if not str(existing.get("section", "")).strip() and str(record.get("section", "")).strip():
                existing["section"] = record.get("section", "")
        return [
            {**item, "sources": "+".join([s for s in item.get("sources", []) if s])}
            for item in merged.values()
        ]

    def _merge_relation_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            src_id = str(record.get("src_node_id", "")).strip()
            tgt_id = str(record.get("tgt_node_id", "")).strip()
            if not src_id or not tgt_id:
                continue
            key = (src_id, tgt_id) if src_id <= tgt_id else (tgt_id, src_id)
            if key not in merged:
                merged[key] = {**record, "sources": [str(record.get("sources", "")).strip()]}
                continue
            existing = merged[key]
            existing["rank"] = max(float(existing.get("rank", 0) or 0), float(record.get("rank", 0) or 0))
            if str(existing.get("description", "")).strip() in {"", "UNKNOWN"}:
                existing["description"] = record.get("description", "")
            source_tag = str(record.get("sources", "")).strip()
            if source_tag and source_tag not in existing["sources"]:
                existing["sources"].append(source_tag)
            if not str(existing.get("source_id", "")).strip() and str(record.get("source_id", "")).strip():
                existing["source_id"] = record.get("source_id", "")
            if not str(existing.get("section", "")).strip() and str(record.get("section", "")).strip():
                existing["section"] = record.get("section", "")
        return [
            {**item, "sources": "+".join([s for s in item.get("sources", []) if s])}
            for item in merged.values()
        ]

    vector_entities: list[dict[str, Any]] = []
    for hit in detail_entity_hits:
        resolved = await _resolve_entity_hit(hit, "vector-detail")
        if resolved is not None:
            vector_entities.append(resolved)
    for hit in macro_entity_hits:
        resolved = await _resolve_entity_hit(hit, "vector-macro")
        if resolved is not None:
            vector_entities.append(resolved)

    vector_relations: list[dict[str, Any]] = []
    for hit in detail_relation_hits:
        resolved = await _resolve_relation_hit(hit, "vector-detail")
        if resolved is not None:
            vector_relations.append(resolved)
    for hit in macro_relation_hits:
        resolved = await _resolve_relation_hit(hit, "vector-macro")
        if resolved is not None:
            vector_relations.append(resolved)

    keyword_entities = await _match_keyword_entities(decomposition["keywords"])
    keyword_path_entities, keyword_path_relations = await _match_keyword_relation_paths(
        decomposition.get("keyword_relations", [])
    )

    graph_entities = _merge_entity_records(vector_entities + keyword_entities + keyword_path_entities)
    graph_relations = _merge_relation_records(vector_relations + keyword_path_relations)

    if not graph_entities and not graph_relations:
        logger.info("[Dual] Empty context: no graph evidence")
        _write_dual_query_log(
            global_config,
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": "dual",
                "original_query": query,
                "final_entities": [],
                "final_section_ids": [],
                "token_totals": {
                    "graph_evidence": 0,
                    "text_chunk_evidence": 0,
                    "section_summary_evidence": 0,
                    "macro_structure_evidence": 0,
                    "context_total": 0,
                },
            },
        )
        return None

    graph_rows_items: list[dict[str, Any]] = []
    for entity in graph_entities:
        graph_rows_items.append(
            {
                "kind": "entity",
                "node_id": entity.get("node_id", ""),
                "payload": [
                    "entity",
                    entity.get("entity", ""),
                    entity.get("description", ""),
                    entity.get("level", ""),
                    entity.get("sources", ""),
                ],
                "text": f"{entity.get('entity', '')} {entity.get('description', '')}",
            }
        )
    for relation in graph_relations:
        graph_rows_items.append(
            {
                "kind": "relation",
                "src_node_id": relation.get("src_node_id", ""),
                "tgt_node_id": relation.get("tgt_node_id", ""),
                "payload": [
                    "relation",
                    relation.get("source", ""),
                    relation.get("target", ""),
                    relation.get("description", ""),
                    relation.get("sources", ""),
                ],
                "text": f"{relation.get('source', '')} {relation.get('target', '')} {relation.get('description', '')}",
            }
        )

    graph_rows_items = truncate_list_by_token_size(
        graph_rows_items,
        key=lambda x: x.get("text", ""),
        max_token_size=runtime_cfg["max_token_for_graph"],
    )
    selected_entity_node_ids = {
        str(item.get("node_id", "")).strip()
        for item in graph_rows_items
        if item.get("kind") == "entity" and str(item.get("node_id", "")).strip()
    }
    selected_relation_edge_keys = {
        tuple(sorted((str(item.get("src_node_id", "")).strip(), str(item.get("tgt_node_id", "")).strip())))
        for item in graph_rows_items
        if item.get("kind") == "relation"
        and str(item.get("src_node_id", "")).strip()
        and str(item.get("tgt_node_id", "")).strip()
    }
    selected_graph_entities = [
        entity for entity in graph_entities if str(entity.get("node_id", "")).strip() in selected_entity_node_ids
    ]
    selected_graph_relations = [
        relation
        for relation in graph_relations
        if tuple(sorted((str(relation.get("src_node_id", "")).strip(), str(relation.get("tgt_node_id", "")).strip())))
        in selected_relation_edge_keys
    ]
    graph_entity_rows = [
        [
            i,
            item["payload"][1],
            item["payload"][2],
            item["payload"][3],
            item["payload"][4],
        ]
        for i, item in enumerate([it for it in graph_rows_items if it["kind"] == "entity"])
    ]
    graph_relation_rows = [
        [
            i,
            item["payload"][1],
            item["payload"][2],
            item["payload"][3],
            item["payload"][4],
        ]
        for i, item in enumerate([it for it in graph_rows_items if it["kind"] == "relation"])
    ]

    def _extract_chunk_ids(raw_source_id: object) -> list[str]:
        return [
            c.strip()
            for c in split_string_by_multi_markers(str(raw_source_id or ""), [GRAPH_FIELD_SEP])
            if c.strip().startswith("chunk-")
        ]

    chunk_hit_count: dict[str, int] = defaultdict(int)
    section_hit_count: dict[str, int] = defaultdict(int)

    for entity in selected_graph_entities:
        for chunk_id in _extract_chunk_ids(entity.get("source_id", "")):
            chunk_hit_count[chunk_id] += 1
        for section_id in _normalize_section_ids(entity.get("section")):
            section_hit_count[section_id] += 1
    for relation in selected_graph_relations:
        for chunk_id in _extract_chunk_ids(relation.get("source_id", "")):
            chunk_hit_count[chunk_id] += 1
        for section_id in _normalize_section_ids(relation.get("section")):
            section_hit_count[section_id] += 1

    chunk_candidates = sorted(chunk_hit_count.items(), key=lambda x: x[1], reverse=True)
    chunk_records: list[dict[str, Any]] = []
    for chunk_id, mapped_count in chunk_candidates:
        chunk_data = await text_chunks_db.get_by_id(chunk_id)
        if not isinstance(chunk_data, dict):
            continue
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "mapped_count": mapped_count,
                "content": str(chunk_data.get("content", "")),
                "doc_id": str(chunk_data.get("full_doc_id", "")),
                "section_id": str(chunk_data.get("structure_leaf_node_id", "")).strip(),
            }
        )

    chunk_records = truncate_list_by_token_size(
        chunk_records,
        key=lambda x: x.get("content", ""),
        max_token_size=runtime_cfg["max_token_for_text_unit"],
    )
    chunk_rows = [
        [
            i,
            record.get("chunk_id", ""),
            record.get("mapped_count", 0),
            record.get("doc_id", ""),
            record.get("section_id", ""),
            record.get("content", ""),
        ]
        for i, record in enumerate(chunk_records)
    ]

    section_rows: list[list[Any]] = []
    section_records: list[dict[str, Any]] = []
    structured_docs = await _load_structured_docs_from_store(full_docs_db)
    if structured_docs:
        section_to_doc: dict[str, dict[str, Any]] = {}
        for doc in structured_docs:
            nodes = doc.get("nodes", {})
            for node_id, node in nodes.items():
                section_to_doc[str(node_id)] = {
                    "doc_name": doc.get("doc_name", doc.get("doc_id", "")),
                    "title": str(node.get("title", node_id)).strip() or node_id,
                    "summary": str(node.get("summary", "")).strip() or str(node.get("text", "")).strip(),
                    "parent_id": str(node.get("parent_id", "")).strip() if node.get("parent_id") is not None else "",
                }

        # Only keep sections that are directly hit by retrieved entities/relations
        matched_section_ids = sorted(
            [sid for sid in section_hit_count.keys() if sid],
            key=lambda x: section_hit_count.get(x, 0),
            reverse=True,
        )
        for sid in matched_section_ids:
            meta = section_to_doc.get(sid)
            if not meta:
                continue
            summary = str(meta.get("summary", "")).strip()
            if not summary:
                continue
            section_records.append(
                {
                    "section_id": sid,
                    "depth": 0,
                    "priority": "matched",
                    "doc_name": str(meta.get("doc_name", "")),
                    "title": str(meta.get("title", sid)),
                    "summary": summary,
                    "mapped_count": section_hit_count.get(sid, 0),
                }
            )

        section_records = truncate_list_by_token_size(
            section_records,
            key=lambda x: x.get("summary", ""),
            max_token_size=runtime_cfg["max_token_for_summary"],
        )
        section_rows = [
            [
                i,
                record.get("section_id", ""),
                record.get("depth", 0),
                record.get("priority", ""),
                record.get("mapped_count", 0),
                record.get("doc_name", ""),
                record.get("title", ""),
                record.get("summary", ""),
            ]
            for i, record in enumerate(section_records)
        ]

    macro_rows: list[list[Any]] = []
    macro_evidence: list[dict[str, Any]] = []
    if enable_tree_traversal:
        macro_evidence = await _collect_macro_evidence_from_structure(
            full_docs_db=full_docs_db,
            macro_subquestions=decomposition["macro_subquestions"],
            detail_subquestions=decomposition["detail_subquestions"],
            query_param=query_param,
            global_config=global_config,
        )
        macro_rows = [
            [
                i,
                item.get("macro_subquestion", ""),
                item.get("doc_name", item.get("doc_id", "")),
                item.get("status", ""),
                item.get("depth", 0),
                item.get("useful_information", ""),
            ]
            for i, item in enumerate(macro_evidence)
        ]

    decomposition_rows = [["detail_subquestions", "macro_subquestions", "keywords", "keyword_relations"]]
    decomposition_rows.append([
        " | ".join(decomposition["detail_subquestions"]) or "NONE",
        " | ".join(decomposition["macro_subquestions"]) or "NONE",
        " | ".join(decomposition["keywords"]) or "NONE",
        " | ".join([f"{r.get('head','')}->{r.get('tail','')}" for r in decomposition.get("keyword_relations", [])]) or "NONE",
    ])

    context_sections = [
        _build_dual_context_section("Query Decomposition", decomposition_rows),
        _build_dual_context_section(
            "Graph Evidence - Entities",
            [["id", "entity", "description", "level", "sources"], *graph_entity_rows],
        ),
        _build_dual_context_section(
            "Graph Evidence - Relations",
            [["id", "source", "target", "description", "sources"], *graph_relation_rows],
        ),
        _build_dual_context_section(
            "Text Chunk Evidence",
            [["id", "chunk_id", "mapped_count", "doc_id", "section_id", "content"], *chunk_rows],
        ),
        _build_dual_context_section(
            "Section Summary Evidence",
            [["id", "section_id", "depth", "priority", "mapped_count", "doc_name", "title", "summary"], *section_rows],
        ),
    ]
    if macro_rows:
        context_sections.append(
            _build_dual_context_section(
                "Macro Structure Evidence",
                [["id", "macro_subquestion", "document", "status", "depth", "useful_information"], *macro_rows],
            )
        )

    context = "\n".join(context_sections)

    entity_section_ids: list[str] = []
    final_entities: list[dict[str, Any]] = []
    for entity in selected_graph_entities:
        section_ids = _normalize_section_ids(entity.get("section"))
        entity_section_ids.extend(section_ids)
        final_entities.append(
            {
                "entity": str(entity.get("entity", "")).strip(),
                "node_id": str(entity.get("node_id", "")).strip(),
                "section_ids": section_ids,
                "sources": str(entity.get("sources", "")).strip(),
            }
        )
    summary_section_ids = [str(r.get("section_id", "")).strip() for r in section_records if str(r.get("section_id", "")).strip()]
    final_section_ids = sorted(set(entity_section_ids + summary_section_ids))

    graph_token_total = _sum_tokens(graph_rows_items, lambda x: x.get("text", ""), tiktoken_model_name)
    chunk_token_total = _sum_tokens(chunk_records, lambda x: x.get("content", ""), tiktoken_model_name)
    section_token_total = _sum_tokens(section_records, lambda x: x.get("summary", ""), tiktoken_model_name)
    macro_token_total = _sum_tokens(macro_evidence, lambda x: x.get("useful_information", ""), tiktoken_model_name)
    context_total_tokens = _count_tokens_safe(context, tiktoken_model_name)

    _write_dual_query_log(
        global_config,
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": "dual",
            "original_query": query,
            "final_entities": final_entities,
            "final_section_ids": final_section_ids,
            "token_totals": {
                "graph_evidence": graph_token_total,
                "text_chunk_evidence": chunk_token_total,
                "section_summary_evidence": section_token_total,
                "macro_structure_evidence": macro_token_total,
                "context_total": context_total_tokens,
            },
        },
    )

    logger.info(
        "[Dual] Context built | entities=%s | relations=%s | chunks=%s | sections=%s | macro_rows=%s | chars=%s",
        len(graph_entity_rows),
        len(graph_relation_rows),
        len(chunk_rows),
        len(section_rows),
        len(macro_rows),
        len(context),
    )
    return context


def _extract_structured_nodes_from_docs(
    new_docs: dict[str, dict],
) -> list[dict[str, Any]]:
    structured_docs: list[dict[str, Any]] = []

    for doc_id, doc_value in new_docs.items():
        content = doc_value.get("content", "") if isinstance(doc_value, dict) else ""
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(parsed, dict):
            continue
        raw_structure = parsed.get("structure")
        if not isinstance(raw_structure, list) or not raw_structure:
            continue

        nodes: dict[str, dict[str, Any]] = {}
        traversal: list[str] = []

        def _collect(raw_nodes: list, parent_id: Union[str, None], path_prefix: str = ""):
            for idx, raw_node in enumerate(raw_nodes):
                if not isinstance(raw_node, dict):
                    continue
                path = f"{path_prefix}.{idx}" if path_prefix else str(idx)
                node_id = str(raw_node.get("node_id", "")).strip() or path
                title = str(raw_node.get("title", "")).strip() or node_id
                summary = str(raw_node.get("summary", "")).strip()
                text = str(raw_node.get("text", "")).strip()
                raw_children = raw_node.get("nodes")
                if not isinstance(raw_children, list):
                    raw_children = raw_node.get("children")
                child_nodes = raw_children if isinstance(raw_children, list) else []
                child_ids = []
                for child_idx, child in enumerate(child_nodes):
                    if not isinstance(child, dict):
                        continue
                    child_path = f"{path}.{child_idx}"
                    child_id = str(child.get("node_id", "")).strip() or child_path
                    child_ids.append(child_id)

                nodes[node_id] = {
                    "node_id": node_id,
                    "title": title,
                    "summary": summary,
                    "text": text,
                    "parent_id": parent_id,
                    "children": child_ids,
                }
                traversal.append(node_id)
                if child_nodes:
                    _collect(child_nodes, node_id, path)

        _collect(raw_structure, parent_id=None)
        if not nodes:
            continue

        structured_docs.append(
            {
                "doc_id": doc_id,
                "doc_name": str(parsed.get("doc_name", "")).strip() or doc_id,
                "doc_description": str(parsed.get("doc_description", "")).strip(),
                "nodes": nodes,
                "traversal": traversal,
            }
        )

    return structured_docs


async def _ensure_structure_node_summary(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    doc_name: str,
    global_config: dict,
    cache: dict[str, str],
) -> str:
    if node_id in cache:
        return cache[node_id]

    node = nodes.get(node_id)
    if not node:
        cache[node_id] = ""
        return ""

    existing = str(node.get("summary", "")).strip()
    if existing:
        cache[node_id] = existing
        return existing

    children = node.get("children", [])
    child_summaries = []
    for child_id in children:
        child_summary = await _ensure_structure_node_summary(
            child_id,
            nodes,
            doc_name,
            global_config,
            cache,
        )
        if child_summary:
            child_summaries.append(f"- [{child_id}] {child_summary}")

    if not child_summaries:
        cache[node_id] = ""
        return ""

    use_llm_func: callable = global_config["cheap_model_func"]
    summary_prompt = PROMPTS["structure_summary_generation"].format(
        doc_name=doc_name,
        unit_id=node_id,
        unit_title=str(node.get("title", node_id)),
        child_summaries="\n".join(child_summaries),
    )
    generated = (await use_llm_func(summary_prompt)).strip()
    nodes[node_id]["summary"] = generated
    cache[node_id] = generated
    return generated


def _collect_level_entity_exports(
    level_nodes: dict[str, list[dict]],
    level_name: str,
) -> list[dict]:
    exports = []
    for entity_name, records in level_nodes.items():
        if not records:
            continue
        entity_types = sorted(
            {
                str(r.get("entity_type", "")).strip().lower()
                for r in records
                if str(r.get("entity_type", "")).strip()
            }
        )
        descriptions = sorted(
            {
                str(r.get("description", "")).strip()
                for r in records
                if str(r.get("description", "")).strip()
            }
        )
        unit_ids = sorted(
            {
                s
                for r in records
                for s in _normalize_section_ids(r.get("section"))
            }
        )
        chunk_ids = sorted(
            {
                c.strip()
                for r in records
                for c in split_string_by_multi_markers(str(r.get("source_id", "")), [GRAPH_FIELD_SEP])
                if c.strip().startswith("chunk-")
            }
        )
        exports.append(
            {
                "entity_name": entity_name,
                "entity_type": entity_types[0] if entity_types else "unknown",
                "entity_types": entity_types,
                "description": GRAPH_FIELD_SEP.join(descriptions),
                "unit_ids": unit_ids,
                "chunk_ids": chunk_ids,
                "graph_level": level_name,
            }
        )
    return exports


def _collect_level_relation_exports(
    level_edges: dict[tuple[str, str], list[dict]],
    level_name: str,
) -> list[dict]:
    exports = []
    for (src_id, tgt_id), records in level_edges.items():
        if not records:
            continue
        descriptions = sorted(
            {
                str(r.get("description", "")).strip()
                for r in records
                if str(r.get("description", "")).strip()
            }
        )
        unit_ids = sorted(
            {
                s
                for r in records
                for s in _normalize_section_ids(r.get("section"))
            }
        )
        chunk_ids = sorted(
            {
                c.strip()
                for r in records
                for c in split_string_by_multi_markers(str(r.get("source_id", "")), [GRAPH_FIELD_SEP])
                if c.strip().startswith("chunk-")
            }
        )
        weight = sum(
            [
                float(r.get("weight", 1.0))
                if is_float_regex(str(r.get("weight", 1.0)))
                else 1.0
                for r in records
            ]
        )
        exports.append(
            {
                "source": src_id,
                "target": tgt_id,
                "description": GRAPH_FIELD_SEP.join(descriptions),
                "weight": weight,
                "unit_ids": unit_ids,
                "chunk_ids": chunk_ids,
                "graph_level": level_name,
            }
        )
    return exports


async def build_dual_level_graphs(
    new_docs: dict[str, dict],
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: Union[BaseVectorStorage, None],
    global_config: dict,
    text_chunks_storage: Union[BaseKVStorage, None] = None,
    full_docs_db: Union[BaseKVStorage, None] = None,
) -> dict[str, list[dict]]:
    def _dual_log(step: str, message: str, color: str = "96") -> None:
        logger.info(f"\033[{color}m[DualGraph][{step}] {message}\033[0m")

    _dual_log("Step 1", f"Start build: input_docs={len(new_docs)}, input_chunks={len(chunks)}", "96")

    async def _upsert_dual_exports_to_vector(
        structural_graph_entities: list[dict],
        structural_graph_relations: list[dict],
        content_graph_entities: list[dict],
        content_graph_relations: list[dict],
    ) -> None:
        if entity_vdb is not None:
            _dual_log("Step 8", "Upsert dual-level entities/relations to vector DB", "95")
            pending_records: list[dict[str, Any]] = []

            for vector_item in structural_graph_entities + content_graph_entities:
                try:
                    level_num = int(vector_item.get("level", 0))
                except (TypeError, ValueError):
                    level_num = 0
                node_id = _compose_level_aware_node_id(str(vector_item["entity_name"]), level_num)
                node_data = await knwoledge_graph_inst.get_node(node_id)
                if node_data is not None and _is_vectorized_flag_true(node_data.get("vectorized")):
                    continue
                pending_records.append(
                    {
                        "kind": "entity",
                        "graph_level": vector_item.get("graph_level", "content"),
                        "level_num": level_num,
                        "node_id": node_id,
                        "entity_name": vector_item["entity_name"],
                        "vdb_id": compute_mdhash_id(
                            f"entity|{vector_item['graph_level']}|{level_num}|{vector_item['entity_name']}",
                            prefix="ent-",
                        ),
                        "payload": {
                            "content": f"{vector_item['entity_name']}\n{vector_item['description']}",
                            "entity_name": vector_item["entity_name"],
                            "entity_type": vector_item.get("entity_type", "unknown"),
                            "level": str(level_num),
                            "kind": "entity",
                            "graph_level": vector_item.get("graph_level", "content"),
                        },
                    }
                )

            for vector_item in structural_graph_relations + content_graph_relations:
                try:
                    level_num = int(vector_item.get("level", 0))
                except (TypeError, ValueError):
                    level_num = 0
                src_node_id = _compose_level_aware_node_id(str(vector_item["source"]), level_num)
                tgt_node_id = _compose_level_aware_node_id(str(vector_item["target"]), level_num)
                edge_data = await knwoledge_graph_inst.get_edge(src_node_id, tgt_node_id)
                if edge_data is None:
                    edge_data = await knwoledge_graph_inst.get_edge(tgt_node_id, src_node_id)
                if edge_data is not None and _is_vectorized_flag_true(edge_data.get("vectorized")):
                    continue
                pending_records.append(
                    {
                        "kind": "relation",
                        "graph_level": vector_item.get("graph_level", "content"),
                        "level_num": level_num,
                        "source": vector_item["source"],
                        "target": vector_item["target"],
                        "src_node_id": src_node_id,
                        "tgt_node_id": tgt_node_id,
                        "vdb_id": compute_mdhash_id(
                            f"relation|{vector_item['graph_level']}|{level_num}|{vector_item['source']}|{vector_item['target']}",
                            prefix="rel-",
                        ),
                        "payload": {
                            "content": f"{vector_item['source']}\n{vector_item['target']}\n{vector_item['description']}",
                            "source": vector_item["source"],
                            "target": vector_item["target"],
                            "level": str(level_num),
                            "kind": "relation",
                            "graph_level": vector_item.get("graph_level", "content"),
                        },
                    }
                )

            total_pending = len(pending_records)
            _dual_log("Step 8", f"Pending vector records={total_pending}", "92")
            if total_pending == 0:
                _dual_log("Step 8", "All graph records already vectorized, skip upsert", "93")
                return

            batch_size = max(1, int(global_config.get("embedding_batch_num", 32)))
            inserted = 0
            for batch_start in range(0, total_pending, batch_size):
                batch_records = pending_records[batch_start : batch_start + batch_size]
                batch_payload = {record["vdb_id"]: record["payload"] for record in batch_records}
                await entity_vdb.upsert(batch_payload)
                for record in batch_records:
                    if record["kind"] == "entity":
                        await knwoledge_graph_inst.upsert_node(record["node_id"], {"vectorized": "true"})
                    else:
                        await knwoledge_graph_inst.upsert_edge(
                            record["src_node_id"],
                            record["tgt_node_id"],
                            {"vectorized": "true"},
                        )

                inserted += len(batch_records)
                now_ticks = PROMPTS["process_tickers"][inserted % len(PROMPTS["process_tickers"])]
                print(
                    f"{now_ticks} [Vector] Inserted {inserted}({inserted * 100 // total_pending}%) records\r",
                    end="",
                    flush=True,
                )
            print()
            _dual_log("Step 8", f"Vector upsert records={total_pending}", "92")
        else:
            _dual_log("Step 8", "Skip vector upsert: entity_vdb is None", "93")

    def _extract_chunk_ids_from_source_id(source_id: object) -> list[str]:
        return [
            c.strip()
            for c in split_string_by_multi_markers(str(source_id or ""), [GRAPH_FIELD_SEP])
            if c.strip().startswith("chunk-")
        ]

    def _is_invalid_entity_type(entity_type_value: object) -> bool:
        raw = str(entity_type_value or "").strip().lower()
        return raw in {"", "unknown", "unk", "n/a", "na", "none", "null", "未知"}

    def _is_invalid_description(description_value: object) -> bool:
        return str(description_value or "").strip().lower() in {"", "unknown", "n/a", "none", "null"}

    async def _repair_graph_integrity_before_vectorization(step_label: str = "Step 7") -> None:
        _dual_log(step_label, "Run graph integrity check before vectorization", "95")

        chunk_lookup: dict[str, TextChunkSchema] = {
            chunk_id: chunk_dp
            for chunk_id, chunk_dp in chunks.items()
            if isinstance(chunk_dp, dict) and str(chunk_id).startswith("chunk-")
        }

        all_nodes = await knwoledge_graph_inst.get_all_nodes()

        invalid_node_count = 0
        invalid_relation_count = 0
        repair_chunk_ids: set[str] = set()
        debug_invalid_entities = []

        node_ids: list[str] = []
        for node in all_nodes:
            node_id = str(node.get("id", "") or "").strip()
            if not node_id:
                continue
            node_ids.append(node_id)

            entity_type = node.get("entity_type", "")
            desc = node.get("description", "")
            bad_type = _is_invalid_entity_type(entity_type)
            bad_desc = _is_invalid_description(desc)
            if bad_type or bad_desc:
                invalid_node_count += 1
                repair_chunk_ids.update(_extract_chunk_ids_from_source_id(node.get("source_id", "")))
                debug_invalid_entities.append({
                    "id": node_id,
                    "entity_type": entity_type,
                    "description": desc,
                    "bad_type": bad_type,
                    "bad_desc": bad_desc
                })

        visited_pairs: set[tuple[str, str]] = set()
        for src_node_id in node_ids:
            edge_pairs = await knwoledge_graph_inst.get_node_edges(src_node_id)
            if not edge_pairs:
                continue
            for src, tgt in edge_pairs:
                src_id = str(src)
                tgt_id = str(tgt)
                pair_key = (src_id, tgt_id) if src_id <= tgt_id else (tgt_id, src_id)
                if pair_key in visited_pairs:
                    continue
                visited_pairs.add(pair_key)

                edge_data = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
                if edge_data is None:
                    edge_data = await knwoledge_graph_inst.get_edge(tgt_id, src_id)
                if edge_data is None:
                    continue

                if _is_invalid_description(edge_data.get("description", "")):
                    invalid_relation_count += 1
                    repair_chunk_ids.update(_extract_chunk_ids_from_source_id(edge_data.get("source_id", "")))


        _dual_log(
            step_label,
            f"Integrity scan result: invalid_entities={invalid_node_count}, invalid_relations={invalid_relation_count}, repair_chunks={len(repair_chunk_ids)}",
            "92",
        )
        if not repair_chunk_ids:
            return

        use_llm_func: callable = global_config["best_model_func"]
        tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
        completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

        repair_targets: list[tuple[str, Any]] = []
        for chunk_id in sorted(repair_chunk_ids):
            chunk_dp = chunk_lookup.get(chunk_id)
            if chunk_dp is None and text_chunks_storage is not None:
                loaded_chunk = await text_chunks_storage.get_by_id(chunk_id)
                if isinstance(loaded_chunk, dict):
                    chunk_dp = loaded_chunk
            if chunk_dp is None:
                continue
            repair_targets.append((chunk_id, chunk_dp))

        if not repair_targets:
            _dual_log(step_label, "No available chunk payloads for integrity repair", "93")
            return

        max_parallel = max(1, int(global_config.get("integrity_repair_concurrency", 8)))
        _dual_log(step_label, f"Integrity repair concurrency={max_parallel}", "95")
        semaphore = asyncio.Semaphore(max_parallel)

        async def _repair_single_chunk(chunk_key: str, chunk_dp: dict) -> None:
            async with semaphore:
                doc_id = str(chunk_dp.get("full_doc_id", "")).strip()
                doc_name = str(chunk_dp.get("doc_name", chunk_dp.get("full_doc_id", ""))).strip()
                doc_description = str(chunk_dp.get("doc_description", "")).strip()
                section_node_id = chunk_dp.get("structure_leaf_node_id")

                entity_prompt = PROMPTS["entity_extraction"].format(
                    tuple_delimiter=tuple_delimiter,
                    record_delimiter=record_delimiter,
                    completion_delimiter=completion_delimiter,
                    doc_name=doc_name,
                    doc_description=doc_description,
                    input_text=chunk_dp["content"],
                )
                parsed_nodes, _ = await _llm_extract_with_retry(
                    use_llm_func,
                    entity_prompt,
                    tuple_delimiter,
                    record_delimiter,
                    completion_delimiter,
                    chunk_key=chunk_key,
                    section_node_id=section_node_id,
                    doc_id=doc_id,
                    global_config=global_config,
                    extraction_stage="integrity_repair_entity",
                )

                relation_prompt = PROMPTS["relation_extraction"].format(
                    tuple_delimiter=tuple_delimiter,
                    record_delimiter=record_delimiter,
                    completion_delimiter=completion_delimiter,
                    entities=json.dumps(list(parsed_nodes.keys()), ensure_ascii=False),
                    input_text=chunk_dp["content"],
                )
                _, parsed_edges = await _llm_extract_with_retry(
                    use_llm_func,
                    relation_prompt,
                    tuple_delimiter,
                    record_delimiter,
                    completion_delimiter,
                    chunk_key=chunk_key,
                    section_node_id=section_node_id,
                    doc_id=doc_id,
                    global_config=global_config,
                    extraction_stage="integrity_repair_relation",
                )

                for _, records in parsed_nodes.items():
                    for record in records:
                        record["level"] = 0

                if parsed_nodes:
                    await asyncio.gather(
                        *[
                            _merge_nodes_then_upsert(entity_name, records, knwoledge_graph_inst, global_config)
                            for entity_name, records in parsed_nodes.items()
                        ]
                    )
                if parsed_edges:
                    await asyncio.gather(
                        *[
                            _merge_edges_then_upsert(src_tgt[0], src_tgt[1], records, knwoledge_graph_inst, global_config)
                            for src_tgt, records in parsed_edges.items()
                        ]
                    )

        repair_tasks = [
            asyncio.create_task(_repair_single_chunk(chunk_key, chunk_dp))
            for chunk_key, chunk_dp in repair_targets
        ]

        repaired_chunks = 0
        for future in asyncio.as_completed(repair_tasks):
            await future
            repaired_chunks += 1
            now_ticks = PROMPTS["process_tickers"][
                repaired_chunks % len(PROMPTS["process_tickers"])
            ]
            print(
                f"{now_ticks} [Integrity Repair] Processed {repaired_chunks}({repaired_chunks * 100 // len(repair_targets)}%) chunks\r",
                end="",
                flush=True,
            )

        print()
        _dual_log(step_label, f"Integrity repair finished: repaired_chunks={repaired_chunks}", "92")

    async def _load_dual_exports_from_existing_graph() -> dict[str, list[dict]]:
        all_nodes = await knwoledge_graph_inst.get_all_nodes()
        structural_graph_entities: list[dict] = []
        content_graph_entities: list[dict] = []
        node_level_map: dict[str, int] = {}

        for node in all_nodes:
            entity_name = str(node.get("entity_name") or node.get("id") or "").strip()
            if not entity_name:
                continue
            try:
                node_level = int(node.get("level", 0))
            except (ValueError, TypeError):
                node_level = 0

            graph_level = "structural" if node_level >= 1 else "content"
            export_item = {
                "entity_name": entity_name,
                "entity_type": str(node.get("entity_type", "unknown") or "unknown"),
                "description": str(node.get("description", "") or ""),
                "weight": 1.0,
                "unit_ids": [],
                "chunk_ids": split_string_by_multi_markers(
                    str(node.get("source_id", "") or ""),
                    [GRAPH_FIELD_SEP],
                ),
                "level": node_level,
                "graph_level": graph_level,
            }
            if graph_level == "structural":
                structural_graph_entities.append(export_item)
            else:
                content_graph_entities.append(export_item)

            node_id = str(node.get("id", "") or "").strip() or _compose_level_aware_node_id(entity_name, node_level)
            node_level_map[node_id] = node_level

        structural_graph_relations: list[dict] = []
        content_graph_relations: list[dict] = []
        visited_pairs: set[tuple[str, str]] = set()

        for source in node_level_map.keys():
            edge_pairs = await knwoledge_graph_inst.get_node_edges(source)
            if not edge_pairs:
                continue
            for src, tgt in edge_pairs:
                src_id = str(src)
                tgt_id = str(tgt)
                pair_key = (src_id, tgt_id) if src_id <= tgt_id else (tgt_id, src_id)
                if pair_key in visited_pairs:
                    continue
                visited_pairs.add(pair_key)

                edge_data = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
                if edge_data is None:
                    edge_data = await knwoledge_graph_inst.get_edge(tgt_id, src_id)
                edge_data = edge_data or {}

                src_entity_name, src_level = _split_level_aware_node_id(src_id)
                tgt_entity_name, tgt_level = _split_level_aware_node_id(tgt_id)
                src_level_val = src_level if src_level is not None else node_level_map.get(src_id, 0)
                tgt_level_val = tgt_level if tgt_level is not None else node_level_map.get(tgt_id, 0)

                graph_level = (
                    "structural"
                    if src_level_val >= 1 and tgt_level_val >= 1
                    else "content"
                )
                export_item = {
                    "source": src_entity_name,
                    "target": tgt_entity_name,
                    "description": str(edge_data.get("description", "") or ""),
                    "weight": float(edge_data.get("weight", 0.0) or 0.0),
                    "unit_ids": [],
                    "chunk_ids": split_string_by_multi_markers(
                        str(edge_data.get("source_id", "") or ""),
                        [GRAPH_FIELD_SEP],
                    ),
                    "level": min(src_level_val, tgt_level_val),
                    "graph_level": graph_level,
                }
                if graph_level == "structural":
                    structural_graph_relations.append(export_item)
                else:
                    content_graph_relations.append(export_item)

        return {
            "structural_graph_entities": structural_graph_entities,
            "structural_graph_relations": structural_graph_relations,
            "content_graph_entities": content_graph_entities,
            "content_graph_relations": content_graph_relations,
        }

    language = str(global_config.get("language", "english")).strip().lower()
    if language != "english":
        _dual_log("Step 1", f"Skip build: language={language} (dual only supports english)", "93")
        return {
            "structural_graph_entities": [],
            "structural_graph_relations": [],
            "content_graph_entities": [],
            "content_graph_relations": [],
        }

    ordered_chunks = list(chunks.items())
    if ordered_chunks:
        last_chunk_key, last_chunk_dp = ordered_chunks[-1]
        if bool(last_chunk_dp.get("dual_processed", False)):
            _dual_log(
                "Step 1",
                f"Resume detected: last chunk already dual_processed, skip Step 2-6 and start from integrity check (chunk={last_chunk_key})",
                "93",
            )
            await _repair_graph_integrity_before_vectorization("Step 7")
            level_graph_exports = await _load_dual_exports_from_existing_graph()
            await _upsert_dual_exports_to_vector(
                level_graph_exports["structural_graph_entities"],
                level_graph_exports["structural_graph_relations"],
                level_graph_exports["content_graph_entities"],
                level_graph_exports["content_graph_relations"],
            )
            _dual_log("Step 9", "Dual graph build completed", "92")
            return level_graph_exports

    _dual_log("Step 2", "Parse structured docs from new_docs", "94")
    structured_docs = _extract_structured_nodes_from_docs(new_docs)
    if not structured_docs and full_docs_db is not None:
        _dual_log("Step 2", "No structured docs in new_docs, fallback to full_docs store", "95")
        structured_docs = await _load_structured_docs_from_store(full_docs_db)
    if not structured_docs:
        _dual_log("Step 2", "No structured docs available, abort dual build", "91")
        return {
            "structural_graph_entities": [],
            "structural_graph_relations": [],
            "content_graph_entities": [],
            "content_graph_relations": [],
        }
    _dual_log("Step 2", f"Structured docs ready: count={len(structured_docs)}", "92")

    async def _mark_dual_chunk_processed(chunk_key: str, chunk_dp: dict) -> None:
        if text_chunks_storage is None:
            return
        await text_chunks_storage.upsert({chunk_key: {**chunk_dp, "dual_processed": True}})
        logger.debug("[DualGraph] Marked dual_processed=true for chunk=%s", chunk_key)

    use_llm_func: callable = global_config["best_model_func"]
    tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
    completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    entity_types = ",".join(PROMPTS["DEFAULT_ENTITY_TYPES"])

    structural_nodes: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
    structural_edges: defaultdict[tuple[str, str, int], list[dict]] = defaultdict(list)
    content_nodes: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
    content_edges: defaultdict[tuple[str, str, int], list[dict]] = defaultdict(list)

    unit_meta: dict[tuple[str, str], dict[str, str]] = {}

    _dual_log("Step 3", "Build structural-level entities and relations", "96")

    async def _process_structural_unit(doc, unit_id, nodes, doc_name, global_config, summary_cache):
        node = nodes.get(unit_id, {})
        summary = await _ensure_structure_node_summary(
            unit_id,
            nodes,
            doc_name,
            global_config,
            summary_cache,
        )
        if not summary:
            logger.debug("[DualGraph][Step 3] Skip unit without summary: doc=%s, unit=%s", doc["doc_id"], unit_id)
            return None

        unit_title = str(node.get("title", unit_id)).strip() or unit_id
        summary_with_title = f"{unit_title}\n{summary}" if summary else unit_title
        unit_meta[(doc["doc_id"], unit_id)] = {
            "doc_name": doc_name,
            "unit_title": unit_title,
            "unit_summary": summary,
        }

        extraction_prompt = PROMPTS["entity_relation_extraction"].format(
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
            doc_name=doc_name,
            doc_description=summary,
            input_text=summary_with_title,
        )
        _doc_id = doc["doc_id"]
        parsed_nodes, parsed_edges = await _llm_extract_with_retry(
            use_llm_func,
            extraction_prompt,
            tuple_delimiter,
            record_delimiter,
            completion_delimiter,
            chunk_key=f"struct-{_doc_id}-{unit_id}",
            section_node_id=unit_id,
            doc_id=_doc_id,
            global_config=global_config,
            extraction_stage="structural_level_extraction",
        )

        return (parsed_nodes, parsed_edges, doc["doc_id"], unit_id)

    # Step 3: gather all units for all docs
    structural_tasks = []
    doc_unit_level_map: dict[str, dict[str, int]] = {}
    total_structural_units = 0
    for doc in structured_docs:
        doc_id = doc["doc_id"]
        doc_name = doc["doc_name"]
        nodes = doc["nodes"]
        doc_unit_level_map[doc_id] = _compute_structure_node_levels(nodes)
        traversal = doc["traversal"]
        summary_cache: dict[str, str] = {}
        _dual_log("Step 3", f"Processing doc={doc_id}, traversal_units={len(traversal)}", "95")
        total_structural_units += len(traversal)
        for unit_id in traversal:
            structural_tasks.append(
                _process_structural_unit(doc, unit_id, nodes, doc_name, global_config, summary_cache)
            )

    _dual_log("Step 3", f"Total structural units to process: {total_structural_units}", "96")

    processed_structural_units = 0
    for future in asyncio.as_completed(structural_tasks):
        result = await future
        if result is not None:
            parsed_nodes, parsed_edges, doc_id, unit_id = result
            unit_level = int(doc_unit_level_map.get(doc_id, {}).get(unit_id, 1) or 1)
            for entity_name, records in parsed_nodes.items():
                for record in records:
                    record["level"] = unit_level
                structural_nodes[(entity_name, unit_level)].extend(records)
            for edge_key, records in parsed_edges.items():
                src_name, tgt_name = tuple(sorted((edge_key[0], edge_key[1])))
                for record in records:
                    record["level"] = unit_level
                structural_edges[(src_name, tgt_name, unit_level)].extend(records)

        processed_structural_units += 1
        now_ticks = PROMPTS["process_tickers"][
            processed_structural_units % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} [Structural] Processed {processed_structural_units}({processed_structural_units*100//total_structural_units}%) units",
            end="\r",
            flush=True,
        )
    print()  # clear the progress bar

    _dual_log(
        "Step 3",
        f"Structural aggregation done: entity_keys={len(structural_nodes)}, relation_keys={len(structural_edges)}",
        "92",
    )

    if structural_nodes:
        _dual_log("Step 4", f"Upsert structural entities: {len(structural_nodes)}", "94")
        await asyncio.gather(
            *[
                _merge_nodes_then_upsert(entity_name, records, knwoledge_graph_inst, global_config)
                for (entity_name, _level), records in structural_nodes.items()
            ]
        )
    if structural_edges:
        _dual_log("Step 4", f"Upsert structural relations: {len(structural_edges)}", "94")
        await asyncio.gather(
            *[
                _merge_edges_then_upsert(src_id, tgt_id, records, knwoledge_graph_inst, global_config)
                for (src_id, tgt_id, _level), records in structural_edges.items()
            ]
        )


    _dual_log("Step 5", "Build content-level entities and relations from chunks", "96")
    # --- Skipped doc/chunk counting logic ---
    # Group chunks by doc_id
    doc_chunk_map = defaultdict(list)
    for chunk_key, chunk_dp in ordered_chunks:
        doc_id = str(chunk_dp.get("full_doc_id", "")).strip()
        doc_chunk_map[doc_id].append((chunk_key, chunk_dp))

    skipped_doc_count = 0
    skipped_chunk_count = 0
    total_doc_count = len(doc_chunk_map)
    total_chunk_count = len(ordered_chunks)
    chunks_to_process = []
    for doc_id, chunk_list in doc_chunk_map.items():
        all_dual_processed = all(chunk_dp.get("dual_processed", False) for _, chunk_dp in chunk_list)
        if all_dual_processed:
            skipped_doc_count += 1
            skipped_chunk_count += len(chunk_list)
        else:
            for chunk_key, chunk_dp in chunk_list:
                if not chunk_dp.get("dual_processed", False):
                    chunks_to_process.append((chunk_key, chunk_dp))
                else:
                    skipped_chunk_count += 1

    _dual_log("Step 5", f"Total docs: {total_doc_count}, total chunks: {total_chunk_count}", "95")
    _dual_log("Step 5", f"Skipped docs (all chunks done): {skipped_doc_count}", "93")
    _dual_log("Step 5", f"Skipped chunks (already dual_processed): {skipped_chunk_count}", "93")
    _dual_log("Step 5", f"Pending chunks for dual content pass: {len(chunks_to_process)}", "95")


    async def _process_content_chunk(chunk_key, chunk_dp):
        unit_id = str(chunk_dp.get("structure_leaf_node_id", "")).strip()
        doc_id = str(chunk_dp.get("full_doc_id", "")).strip()
        if not unit_id or not doc_id:
            logger.debug("[DualGraph][Step 5] Skip chunk missing unit/doc mapping: chunk=%s", chunk_key)
            await _mark_dual_chunk_processed(chunk_key, chunk_dp)
            return None

        unit = unit_meta.get((doc_id, unit_id), None)
        if unit is None:
            logger.debug("[DualGraph][Step 5] Skip chunk without unit meta: chunk=%s, doc=%s, unit=%s", chunk_key, doc_id, unit_id)
            await _mark_dual_chunk_processed(chunk_key, chunk_dp)
            return None

        entity_prompt = PROMPTS["entity_extraction"].format(
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
            doc_name=unit["doc_name"],
            doc_description=unit["unit_summary"],
            input_text=chunk_dp["content"],
        )
        parsed_nodes, _ = await _llm_extract_with_retry(
            use_llm_func,
            entity_prompt,
            tuple_delimiter,
            record_delimiter,
            completion_delimiter,
            chunk_key=chunk_key,
            section_node_id=unit_id,
            doc_id=doc_id,
            global_config=global_config,
            extraction_stage="content_level_entity_extraction",
        )

        relation_prompt = PROMPTS["relation_extraction"].format(
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
            entities=json.dumps(list(parsed_nodes.keys()), ensure_ascii=False),
            input_text=chunk_dp["content"],
        )
        _, parsed_edges = await _llm_extract_with_retry(
            use_llm_func,
            relation_prompt,
            tuple_delimiter,
            record_delimiter,
            completion_delimiter,
            chunk_key=chunk_key,
            section_node_id=unit_id,
            doc_id=doc_id,
            global_config=global_config,
            extraction_stage="content_level_relation_extraction",
        )

        for entity_name, records in parsed_nodes.items():
            for record in records:
                record["level"] = 0
            content_nodes[(entity_name, 0)].extend(records)

        for edge_key, records in parsed_edges.items():
            src_name, tgt_name = tuple(sorted((edge_key[0], edge_key[1])))
            for record in records:
                record["level"] = 0
            content_edges[(src_name, tgt_name, 0)].extend(records)

        await _mark_dual_chunk_processed(chunk_key, chunk_dp)
        return True

    # Step 5: gather all chunk tasks
    content_tasks = [_process_content_chunk(chunk_key, chunk_dp) for chunk_key, chunk_dp in chunks_to_process]

    processed_content_chunks = 0
    for future in asyncio.as_completed(content_tasks):
        await future
        processed_content_chunks += 1
        now_ticks = PROMPTS["process_tickers"][
            processed_content_chunks % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} [Content] Processed {processed_content_chunks}({processed_content_chunks*100//len(chunks_to_process)}%) chunks",
            end="\r",
            flush=True,
        )

    print()  # clear the progress bar
    _dual_log("Step 5", f"Processed {processed_content_chunks}/{len(chunks_to_process)} content chunks", "92")

    _dual_log(
        "Step 5",
        f"Content aggregation done: entity_keys={len(content_nodes)}, relation_keys={len(content_edges)}",
        "92",
    )

    content_merged_entities = []
    if content_nodes:
        _dual_log("Step 6", f"Upsert content entities: {len(content_nodes)}", "94")
        content_merged_entities = await asyncio.gather(
            *[
                _merge_nodes_then_upsert(entity_name, records, knwoledge_graph_inst, global_config)
                for (entity_name, _level), records in content_nodes.items()
            ]
        )
    if content_edges:
        _dual_log("Step 6", f"Upsert content relations: {len(content_edges)}", "94")
        await asyncio.gather(
            *[
                _merge_edges_then_upsert(src_id, tgt_id, records, knwoledge_graph_inst, global_config)
                for (src_id, tgt_id, _level), records in content_edges.items()
            ]
        )

    await _repair_graph_integrity_before_vectorization("Step 7")

    _dual_log("Step 7", "Build dual-level export payloads", "96")
    level_graph_exports = await _load_dual_exports_from_existing_graph()
    structural_graph_entities = level_graph_exports["structural_graph_entities"]
    structural_graph_relations = level_graph_exports["structural_graph_relations"]
    content_graph_entities = level_graph_exports["content_graph_entities"]
    content_graph_relations = level_graph_exports["content_graph_relations"]

    _dual_log(
        "Step 7",
        "Export payload sizes: structural_entities={}, structural_relations={}, content_entities={}, content_relations={}".format(
            len(structural_graph_entities),
            len(structural_graph_relations),
            len(content_graph_entities),
            len(content_graph_relations),
        ),
        "92",
    )

    await _upsert_dual_exports_to_vector(
        structural_graph_entities,
        structural_graph_relations,
        content_graph_entities,
        content_graph_relations,
    )

    _dual_log("Step 9", "Dual graph build completed", "92")
    return {
        "structural_graph_entities": structural_graph_entities,
        "structural_graph_relations": structural_graph_relations,
        "content_graph_entities": content_graph_entities,
        "content_graph_relations": content_graph_relations,
    }

async def extract_hierarchical_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
)-> Union[BaseGraphStorage, None]:
    """Extract entities and relations from text chunks

    Args:
        chunks: text chunks
        knowledge_graph_inst: knowledge graph instance
        entity_vdb: entity vector database
        global_config: global configuration

    Returns:
        Union[BaseGraphStorage, None]: knowledge graph instance
    """
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    is_chinese = _is_chinese_language(global_config)
    entity_extract_prompt = PROMPTS[
        "entity_extraction_zh" if is_chinese else "entity_extraction"
    ]
    relation_extract_prompt = PROMPTS["hi_relation_extraction"]
    entity_types = (
        PROMPTS["DEFAULT_ENTITY_TYPES_ZH"]
        if is_chinese
        else PROMPTS["DEFAULT_ENTITY_TYPES"]
    )

    context_base_entity = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        entity_types_zh=",".join(entity_types),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]       # judge if there are still entities still need to be extracted

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content_entity(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        section_node_id = chunk_dp.get("structure_leaf_node_id")
        doc_id = str(chunk_dp.get("full_doc_id", "")).strip()
        doc_name = str(chunk_dp.get("doc_name", chunk_dp.get("full_doc_id", ""))).strip()
        doc_description = str(chunk_dp.get("doc_description", "")).strip()
        hint_prompt = entity_extract_prompt.format(
            **context_base_entity,
            input_text=content,
            doc_name=doc_name,
            doc_description=doc_description,
        )
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        # check if need gleaning
        if entity_extract_max_gleaning > 0:
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result == "yes":
                logger.info(f"\033[91m[Found Missed Entities, Gleaning {entity_extract_max_gleaning} times]\033[0m")
                for now_glean_index in range(entity_extract_max_gleaning):
                    glean_result = await use_llm_func(continue_prompt, history_messages=history)
                    history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
                    final_result += glean_result
                    if now_glean_index == entity_extract_max_gleaning - 1:
                        break
                    if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                        if_loop_prompt, history_messages=history
                    )
                    if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                    if if_loop_result != "yes":
                        break
            else:
                logger.info(f"\033[92m[No Missed Entities]\033[0m")
        else:
            logger.info("\033[93m[Skip Gleaning Check] entity_extract_max_gleaning=0\033[0m")

        maybe_nodes, maybe_edges = await _llm_extract_with_retry(
            use_llm_func,
            hint_prompt,
            context_base_entity["tuple_delimiter"],
            context_base_entity["record_delimiter"],
            context_base_entity["completion_delimiter"],
            chunk_key,
            section_node_id,
            doc_id,
            initial_response=final_result,
            global_config=global_config,
            extraction_stage="separate_entity_extraction",
        )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)
    
    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    entity_results = await asyncio.gather(
        *[_process_single_content_entity(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar

    context_entities = {
        key[0]: list(x[0].keys()) for key, x in zip(ordered_chunks, entity_results)
    }
    
    already_processed = 0
    async def _process_single_content_relation(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        section_node_id = chunk_dp.get("structure_leaf_node_id")
        doc_id = str(chunk_dp.get("full_doc_id", "")).strip()

        entities = context_entities[chunk_key]
        context_base_relation = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entities=",".join(entities)
            )
        hint_prompt = relation_extract_prompt.format(**context_base_relation, input_text=content)      # fill in the parameter
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        # check if need gleaning
        if entity_extract_max_gleaning > 0:
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
            if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result == "yes":
                logger.info(f"\033[91m[Found Missed Relations, Gleaning {entity_extract_max_gleaning} times]\033[0m")
                for now_glean_index in range(entity_extract_max_gleaning):
                    glean_result = await use_llm_func(continue_prompt, history_messages=history)

                    history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
                    final_result += glean_result
                    if now_glean_index == entity_extract_max_gleaning - 1:
                        break

                    if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                        if_loop_prompt, history_messages=history
                    )
                    if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                    if if_loop_result != "yes":
                        break
            else:
                logger.info(f"\033[92m[No Missed Relations]\033[0m")
        else:
            logger.info("\033[93m[Skip Gleaning Check] entity_extract_max_gleaning=0\033[0m")

        maybe_nodes, maybe_edges = await _llm_extract_with_retry(
            use_llm_func,
            hint_prompt,
            context_base_relation["tuple_delimiter"],
            context_base_relation["record_delimiter"],
            context_base_relation["completion_delimiter"],
            chunk_key,
            section_node_id,
            doc_id,
            initial_response=final_result,
            global_config=global_config,
            extraction_stage="separate_relation_extraction",
        )
        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    relation_results = []
    all_relations = {}
    if not is_chinese:
        # Chinese prompt already extracts both entities and relationships.
        relation_results = await asyncio.gather(
            *[_process_single_content_relation(c) for c in ordered_chunks]
        )
        print()

        # fetch all relations from results
        for item in relation_results:
            for k, v in item[1].items():
                all_relations[k] = v
    
    maybe_nodes = defaultdict(list)                     # for all chunks
    maybe_edges = defaultdict(list)
    # extracted entities and relations
    for m_nodes, m_edges in entity_results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    for _, m_edges in relation_results:
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    # store the nodes
    all_entities_data = await asyncio.gather(           
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    # store the edges
    await asyncio.gather(                               
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None and not is_chinese:
        data_for_vdb = {                                # key is the md5 hash of the entity name string
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],   # entity name and description construct the content
                "entity_name": dp["entity_name"],
                "entity_type": dp.get("entity_type", "unknown"),
                "level": str(dp.get("level", 0)),
                "kind": "entity",
            }
            for dp in all_entities_data
        }
        relation_data_for_vdb = {
            compute_mdhash_id(f"{src}|{tgt}", prefix="rel-"): {
                "content": f"{src}\n{tgt}\n{desc}",
                "source": src,
                "target": tgt,
                "level": "0",
                "kind": "relation",
            }
            for (src, tgt), relations in maybe_edges.items()
            for desc in [GRAPH_FIELD_SEP.join(sorted({str(r.get('description', '')).strip() for r in relations if str(r.get('description', '')).strip()}))]
            if desc
        }
        data_for_vdb.update(relation_data_for_vdb)
        await entity_vdb.upsert(data_for_vdb)
        await _mark_graph_records_vectorized(
            knowledge_graph_inst,
            entity_names=[str(dp.get("entity_name", "")).strip() for dp in all_entities_data],
            edge_pairs=[(str(src), str(tgt)) for (src, tgt) in maybe_edges.keys()],
            level=0,
        )
    elif is_chinese:
        logger.info("\033[93m[Skip Entity Embedding Storage]\033[0m")
    return knowledge_graph_inst

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    text_chunks_storage: Union[BaseKVStorage, None] = None,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = [
        (chunk_id, chunk_data)
        for chunk_id, chunk_data in chunks.items()
        if isinstance(chunk_data, dict) and not bool(chunk_data.get("processed", False))
    ]
    if not ordered_chunks:
        logger.info("No unprocessed chunks found for entity extraction")
        return knwoledge_graph_inst

    is_chinese = _is_chinese_language(global_config)
    entity_extract_prompt = PROMPTS[
        "entity_extraction_zh" if is_chinese else "entity_extraction"
    ]
    entity_types = (
        PROMPTS["DEFAULT_ENTITY_TYPES_ZH"]
        if is_chinese
        else PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        entity_types_zh=",".join(entity_types),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]     # means low quality in the last extraction
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]       # judge if there are still entities still need to be extracted

    already_processed = 0
    already_entities = 0
    already_relations = 0
    processed_chunk_count = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):           # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations, processed_chunk_count
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        section_node_id = chunk_dp.get("structure_leaf_node_id")
        doc_id = str(chunk_dp.get("full_doc_id", "")).strip()
        doc_name = str(chunk_dp.get("doc_name", chunk_dp.get("full_doc_id", ""))).strip()
        doc_description = str(chunk_dp.get("doc_description", "")).strip()
        hint_prompt = entity_extract_prompt.format(
            **context_base,
            input_text=content,
            doc_name=doc_name,
            doc_description=doc_description,
        )
        final_result = await use_llm_func(hint_prompt)                                      # feed into LLM with the prompt

        # check if need gleaning
        if entity_extract_max_gleaning > 0:
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)               # set as history
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result == "yes":
                logger.info(f"\033[91m[Found Missed Entities, Gleaning {entity_extract_max_gleaning} times]\033[0m")
                for now_glean_index in range(entity_extract_max_gleaning):
                    glean_result = await use_llm_func(continue_prompt, history_messages=history)
                    history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)      # add to history
                    final_result += glean_result
                    if now_glean_index == entity_extract_max_gleaning - 1:
                        break
                    if_loop_result: str = await use_llm_func(                                       # judge if we still need the next iteration
                        if_loop_prompt, history_messages=history
                    )
                    if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                    if if_loop_result != "yes":
                        break
            else:
                logger.info(f"\033[92m[No Missed Entities]\033[0m")
        else:
            logger.info("\033[93m[Skip Gleaning Check] entity_extract_max_gleaning=0\033[0m")

        maybe_nodes, maybe_edges = await _llm_extract_with_retry(
            use_llm_func,
            hint_prompt,
            context_base["tuple_delimiter"],
            context_base["record_delimiter"],
            context_base["completion_delimiter"],
            chunk_key,
            section_node_id,
            doc_id,
            initial_response=final_result,
            global_config=global_config,
            extraction_stage="entity_relation_extraction",
        )

        # Merge and write graph data immediately for each chunk, enabling resume from failures.
        all_entities_data = []
        if maybe_nodes:
            all_entities_data = await asyncio.gather(
                *[
                    _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
                    for k, v in maybe_nodes.items()
                ]
            )
        if maybe_edges:
            await asyncio.gather(
                *[
                    _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
                    for k, v in maybe_edges.items()
                ]
            )

        if entity_vdb is not None and not is_chinese:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                    "entity_type": dp.get("entity_type", "unknown"),
                    "level": str(dp.get("level", 0)),
                    "kind": "entity",
                }
                for dp in all_entities_data
            }
            relation_data_for_vdb = {
                compute_mdhash_id(f"{src}|{tgt}", prefix="rel-"): {
                    "content": f"{src}\n{tgt}\n{desc}",
                    "source": src,
                    "target": tgt,
                    "level": "0",
                    "kind": "relation",
                }
                for (src, tgt), relations in maybe_edges.items()
                for desc in [
                    GRAPH_FIELD_SEP.join(
                        sorted(
                            {
                                str(r.get("description", "")).strip()
                                for r in relations
                                if str(r.get("description", "")).strip()
                            }
                        )
                    )
                ]
                if desc
            }
            if relation_data_for_vdb:
                data_for_vdb.update(relation_data_for_vdb)
            if data_for_vdb:
                await entity_vdb.upsert(data_for_vdb)
                await _mark_graph_records_vectorized(
                    knwoledge_graph_inst,
                    entity_names=[str(dp.get("entity_name", "")).strip() for dp in all_entities_data],
                    edge_pairs=[(str(src), str(tgt)) for (src, tgt) in maybe_edges.keys()],
                    level=0,
                )

        if text_chunks_storage is not None:
            next_chunk_data = {**chunk_dp, "processed": True}
            await text_chunks_storage.upsert({chunk_key: next_chunk_data})

        already_processed += 1                                      # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        processed_chunk_count += 1
        now_ticks = PROMPTS["process_tickers"][                     # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return

    # Process chunks one-by-one so completed chunks can be resumed safely after errors.
    for chunk_item in ordered_chunks:
        await _process_single_content(chunk_item)

    print()  # clear the progress bar
    if processed_chunk_count == 0:
        logger.warning("Didn't process any chunks, maybe all chunks are already marked as processed")
        return knwoledge_graph_inst
    if is_chinese:
        logger.info("\033[93m[Skip Entity Embedding Storage]\033[0m")
    return knwoledge_graph_inst


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [      # the entities related to the retrieved entities
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(   # get relations related to the retrieved entities
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )                               # where the source entities are the retrieved entities
    all_one_hop_nodes = set()       # find the one hop neighbors
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(  # get node information from storage
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {               # find the text chunks of the 1-hop neighbors entities
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,     # count of relations related to the chunk
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(        # sort by relation counts
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    return all_edges_data


async def _find_most_related_edges_from_paths(
    path_datas: list[dict],
    path: list[str],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    # all_related_edges = await asyncio.gather(
    #     *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    # )
    # all_reasoning_path = await asyncio.gather(
    #                         *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in knowledge_graph_inst._graph.subgraph(path).edges()]
    #                     )
    all_reasoning_path = knowledge_graph_inst._graph.subgraph(path).edges()
    all_edges = set()
    all_edges.update([tuple(sorted(e)) for e in all_reasoning_path])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_bridge_knowledge,
    )
    return all_edges_data


# context functions
async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = _filter_vdb_hits_by_kind(await entities_vdb.query(query, top_k=query_param.top_k), "entity")          # find the top-k(20) related entities
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _build_hierarchical_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = _filter_vdb_hits_by_kind(await entities_vdb.query(query, top_k=query_param.top_k * 10), "entity")          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    def find_path_with_required_nodes(graph, source, target, required_nodes):
        # inital final path
        final_path = []
        # 起点设置为当前节点
        current_node = source

        # 遍历必经节点
        for next_node in required_nodes:
            # 找到从当前节点到下一个必经节点的最短路径
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            # 合并路径（避免重复添加当前节点）
            if final_path:
                final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
            else:
                final_path.extend(sub_path)
            
            # 更新当前节点为下一个必经节点
            current_node = next_node

        # 最后，从最后一个必经节点到目标节点的路径
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    key_entities = []
    max_entity_num = query_param.top_m
    key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = find_path_with_required_nodes(knowledge_graph_inst._graph, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        # use_reasoning_path = await _find_most_related_edges_from_entities(
        #                     path_datas, query_param, knowledge_graph_inst
        #                 )
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)
    
    # # fetch the relations of the reasoning paths
    # reasoning_path = []
    # for i in range(len(path) - 1):
    #     src = path[i]
    #     tgt = path[i + 1]
    #     cur_relation = (await knowledge_graph_inst.get_edge(src, tgt))['description']
    #     reasoning_path.append(cur_relation)
    # reasoning_path = list(set(reasoning_path))

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # display reference info
    entities = [n["entity_name"] for n in node_datas]
    chunks = [(t["full_doc_id"], t["chunk_order_index"]) for t in use_text_units]

    references_context = (
        f"Entities ({len(entities)}): {entities}\n\n"
        f"Chunks (doc_id, chunk_index) ({len(chunks)}): {chunks}\n"
    )

    logging.info(f"====== References ======:\n{references_context}")
    return f"""
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Detail Entity Information-----
```csv
{entities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_hibridge_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = _filter_vdb_hits_by_kind(await entities_vdb.query(query, top_k=query_param.top_k * 10), "entity")          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # use_relations = await _find_most_related_edges_from_entities(
    #     node_datas, query_param, knowledge_graph_inst
    # )

    def find_path_with_required_nodes(graph, source, target, required_nodes):
        # inital final path
        final_path = []
        # 起点设置为当前节点
        current_node = source

        # 遍历必经节点
        for next_node in required_nodes:
            # 找到从当前节点到下一个必经节点的最短路径
            try:
                sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
            except nx.NetworkXNoPath:
                # raise ValueError(f"No path between {current_node} and {next_node}.")
                final_path.extend([next_node])
                current_node = next_node
                continue
            
            # 合并路径（避免重复添加当前节点）
            if final_path:
                final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
            else:
                final_path.extend(sub_path)
            
            # 更新当前节点为下一个必经节点
            current_node = next_node

        # 最后，从最后一个必经节点到目标节点的路径
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=target)
            final_path.extend(sub_path[1:])  # 从第二个节点开始添加，避免重复
        except nx.NetworkXNoPath:
            # raise ValueError(f"No path between {current_node} and {target}.")
            final_path.extend([target])

        return final_path

    key_entities = []
    max_entity_num = query_param.top_m
    key_entities = [overall_node_datas[:max_entity_num]]
    # unique key entities
    key_entities = [[e['entity_name'] for e in k] for k in key_entities]
    key_entities = list(set([k for kk in key_entities for k in kk]))
    # find the shortest path between the key entities
    try:
        path = find_path_with_required_nodes(knowledge_graph_inst._graph, key_entities[0], key_entities[-1], key_entities[1:-1])
        # path = list(set(path))
        path_datas = await asyncio.gather(      # get full information of retrieved entities
            *[knowledge_graph_inst.get_node(r) for r in path]
        )
        path_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r) for r in path]
        )
        path_datas = [                          # add rank, which is the degree
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(path, path_datas, path_degrees)
            if n is not None
        ]
        use_reasoning_path = await _find_most_related_edges_from_paths(
                                path_datas, path, query_param, knowledge_graph_inst
                            )
    except ValueError as e:
        print(e)

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_reasoning_path)} reasoning path items, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    reasoning_path_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_reasoning_path):
        reasoning_path_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    reasoning_path_context = list_of_list_to_csv(reasoning_path_section_list)
    
    # reasoning_path_context = list_of_list_to_csv([["id", "content"]] + [[i, p] for i, p in enumerate(reasoning_path)])
    
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Reasoning Path-----
```csv
{reasoning_path_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_higlobal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = _filter_vdb_hits_by_kind(await entities_vdb.query(query, top_k=query_param.top_k * 10), "entity")          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    overall_node_datas = node_datas
    node_datas = node_datas[:query_param.top_k]

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )


    logger.info(
        f"Using {len(use_text_units)} text units"
    )

    entites_section_list = [["id", "entity", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Source Documents-----
```csv
{text_units_context}
```
"""


async def _build_hilocal_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = _filter_vdb_hits_by_kind(await entities_vdb.query(query, top_k=query_param.top_k), "entity")          # find the top-k(20) related entities

    if not len(results):    # results just with entity name
        return None
    node_datas = await asyncio.gather(      # get full information of retrieved entities
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):    # for robustness
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [                          # add rank, which is the degree
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    logger.info(
        f"Using {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    
    relation_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relation_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relation_context = list_of_list_to_csv(relation_section_list)
    
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relations-----
```csv
{relation_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


# query functions


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    with timer():
        results = await chunks_vdb.query(query, top_k=query_param.top_k)
        if not len(results):
            return PROMPTS["fail_response"]
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        )
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
        if query_param.only_need_context:
            return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response


async def dual_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    full_docs_db: BaseKVStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["best_model_func"]
    with timer():
        context = await _build_dual_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            full_docs_db,
            text_chunks_db,
            query_param,
            global_config,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt = PROMPTS["dual_rag_response"].format(
        context_data=context,
        response_type=query_param.response_type,
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response
