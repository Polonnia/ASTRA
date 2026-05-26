import asyncio
import json
import logging
import os
import re

try:
    from .utils import (
        ChatGPT_API_async,
        ChatGPT_API_with_finish_reason,
        count_tokens,
        create_clean_structure_for_description,
        extract_json,
        format_structure,
        generate_doc_description,
        generate_node_summary,
        structure_to_list,
        write_node_id,
    )
except Exception:
    from utils import (
        ChatGPT_API_async,
        ChatGPT_API_with_finish_reason,
        count_tokens,
        create_clean_structure_for_description,
        extract_json,
        format_structure,
        generate_doc_description,
        generate_node_summary,
        structure_to_list,
        write_node_id,
    )


logger = logging.getLogger(__name__)


def split_text_by_chars(text, chunk_chars=12000):
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    return [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def _normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", "", s)
    return s


def _structure_sort_key(structure):
    if not isinstance(structure, str):
        return (10**9,)
    parts = []
    for p in structure.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(10**9)
    return tuple(parts)


def _clean_toc_items(items):
    if not isinstance(items, list):
        return []

    cleaned = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        structure = str(item.get("structure", "")).strip()
        title = str(item.get("title", "")).strip()
        if not structure or not title:
            continue
        key = (structure, title)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"structure": structure, "title": title})

    cleaned.sort(key=lambda x: _structure_sort_key(x["structure"]))
    return cleaned


def _log_toc_items(stage, toc_items, max_items=30):
    if not toc_items:
        logger.info("%s: no TOC items found", stage)
        return

    logger.info("%s: found %d TOC items", stage, len(toc_items))
    for item in toc_items[:max_items]:
        logger.info("%s: %s -> %s", stage, item.get("structure"), item.get("title"))
    if len(toc_items) > max_items:
        logger.info("%s: ... %d more items omitted", stage, len(toc_items) - max_items)


def _ask_json_with_finish_reason(prompt, model=None):
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=prompt)
    logger.info("LLM JSON check finished with reason: %s", finish_reason)
    if finish_reason not in ("finished", "max_output_reached"):
        return {}
    parsed = extract_json(response)
    return parsed if isinstance(parsed, dict) else {}


def _chunk_has_toc(chunk_text, model=None):
    prompt = f"""
You are given a text chunk from a document.

Task: decide whether this chunk contains table-of-contents style entries (section titles list),
not just normal body paragraphs.

Return JSON only:
{{
  "has_toc": "yes" or "no",
  "reason": "short reason"
}}

Text chunk:
{chunk_text}
"""
    result = _ask_json_with_finish_reason(prompt, model=model)
    return str(result.get("has_toc", "no")).strip().lower() == "yes"


def _is_toc_complete(chunk_text, toc_items, model=None):
    if not toc_items:
        return False

    prompt = f"""
You are given:
1) a text chunk, and
2) an extracted TOC list.

Task: decide whether the extracted TOC is complete for all TOC content that appears in the given text chunk.

Return JSON only:
{{
  "toc_complete": "yes" or "no",
  "reason": "short reason"
}}

Extracted TOC:
{json.dumps(toc_items, ensure_ascii=False, indent=2)}

Text chunk:
{chunk_text}
"""
    result = _ask_json_with_finish_reason(prompt, model=model)
    return str(result.get("toc_complete", "no")).strip().lower() == "yes"


def _extract_toc_until_complete(chunks, model=None):
    if not chunks:
        return [], None

    chunk_idx = 0
    while chunk_idx < len(chunks):
        logger.info("TOC scan check on chunk %d/%d", chunk_idx + 1, len(chunks))
        has_toc = _chunk_has_toc(chunks[chunk_idx], model=model)
        logger.info("TOC scan result for chunk %d: has_toc=%s", chunk_idx + 1, has_toc)

        if not has_toc:
            chunk_idx += 1
            continue

        merged_text = chunks[chunk_idx]
        merged_count = 1
        while True:
            logger.info(
                "TOC extraction attempt on merged chunk window starting at %d with %d chunk(s)",
                chunk_idx + 1,
                merged_count,
            )
            toc_items = _clean_toc_items(_extract_toc_init_txt(merged_text, model=model))
            _log_toc_items("Merged window TOC", toc_items)

            is_complete = _is_toc_complete(merged_text, toc_items, model=model)
            logger.info(
                "TOC completeness check for window starting at %d: %s",
                chunk_idx + 1,
                is_complete,
            )

            if is_complete:
                logger.info(
                    "TOC confirmed complete. Stop scanning remaining chunks after %d/%d.",
                    chunk_idx + merged_count,
                    len(chunks),
                )
                return toc_items, (chunk_idx + merged_count - 1)

            next_chunk_idx = chunk_idx + merged_count
            if next_chunk_idx >= len(chunks):
                logger.info("Reached last chunk while expanding TOC window; using current TOC result")
                return toc_items, (chunk_idx + merged_count - 1)

            logger.info(
                "TOC incomplete, merging next chunk %d/%d and retrying",
                next_chunk_idx + 1,
                len(chunks),
            )
            merged_text += "\n" + chunks[next_chunk_idx]
            merged_count += 1

    logger.info("No TOC-like chunk detected in the full document")
    return [], None


def _extract_toc_init_txt(part, model=None):
    prompt = """
    You are an expert in extracting a hierarchical table of contents.

    Your task is to extract section titles and output them in a strict tree-index sequence.

    Rules:
    1. structure uses numeric hierarchy such as 1, 1.1, 1.1.1.
    2. title should keep the original title wording from text (only normalize spacing if needed).
    3. Return JSON list only.

    Output format:
    [
      {
        "structure": "1",
        "title": "Introduction"
      }
    ]

    Directly return the final JSON structure. Do not output anything else.
    """
    response, finish_reason = ChatGPT_API_with_finish_reason(
        model=model,
        prompt=prompt + "\nGiven text:\n" + part,
    )
    logger.info("Initial TOC extraction finished with reason: %s", finish_reason)
    if finish_reason != "finished":
        raise Exception(f"finish reason: {finish_reason}")
    return extract_json(response)


def _extract_toc_continue_txt(previous_toc, part, model=None):
    prompt = """
    You are an expert in extracting a hierarchical table of contents from plain text.

    You will be given:
    1) previous extracted TOC items
    2) a new text chunk

    Your task is to continue the TOC extraction and return ONLY NEW items not already in previous TOC.

    Rules:
    1. structure uses numeric hierarchy such as 1, 1.1, 1.1.1.
    2. title should keep original title wording from text (only normalize spacing if needed).
    3. If there are no new TOC items in this chunk, return []

    Output format:
    [
      {
        "structure": "2.3",
        "title": "Experiments"
      }
    ]

    Directly return JSON list only. Do not output anything else.
    """
    payload = (
        prompt
        + "\nPrevious TOC:\n"
        + json.dumps(previous_toc, ensure_ascii=False, indent=2)
        + "\n\nCurrent text chunk:\n"
        + part
    )
    response, finish_reason = ChatGPT_API_with_finish_reason(model=model, prompt=payload)
    logger.info("TOC continuation extraction finished with reason: %s", finish_reason)
    if finish_reason != "finished":
        raise Exception(f"finish reason: {finish_reason}")
    return extract_json(response)


def _find_title_positions(lines, offsets, toc_items):
    line_cursor = 0
    result = []

    for item in toc_items:
        title = item["title"]
        normalized_title = _normalize_text(title)
        matched_pos = None

        for i in range(line_cursor, len(lines)):
            line_norm = _normalize_text(lines[i])
            if not line_norm:
                continue
            if normalized_title and (
                normalized_title in line_norm
                or line_norm in normalized_title
            ):
                matched_pos = offsets[i]
                line_cursor = i + 1
                break

        result.append({
            "structure": item["structure"],
            "title": item["title"],
            "level": len(item["structure"].split(".")),
            "start_pos": matched_pos,
            "text": "",
        })

    return result


def _collect_title_candidates(lines, toc_items, max_candidates_per_title=6):
    candidates = []
    for item in toc_items:
        title = item["title"]
        normalized_title = _normalize_text(title)
        title_candidates = []
        if not normalized_title:
            candidates.append(title_candidates)
            continue

        for i, line in enumerate(lines):
            line_norm = _normalize_text(line)
            if not line_norm:
                continue
            if normalized_title in line_norm or line_norm in normalized_title:
                title_candidates.append(i)
                if len(title_candidates) >= max_candidates_per_title:
                    break

        candidates.append(title_candidates)
    return candidates


def _build_line_context(lines, line_index, window=2):
    start = max(0, line_index - window)
    end = min(len(lines), line_index + window + 1)
    context_lines = []
    for i in range(start, end):
        prefix = ">> " if i == line_index else "   "
        context_lines.append(f"{prefix}L{i + 1}: {lines[i].rstrip()}")
    return "\n".join(context_lines)


async def _verify_heading_candidate(title, context, model=None, semaphore=None):
    prompt = f"""
    You are given a section title and a short context around one matched line in a plain-text document.

    Task: decide whether the matched line is truly a chapter/section boundary heading,
    rather than regular body text that happens to contain the same title words.

    Section title: {title}

    Context:
    {context}

    Return JSON only:
    {{
      "is_section_boundary": "yes" or "no",
      "confidence": "high" or "medium" or "low"
    }}
    """

    try:
        if semaphore is not None:
            async with semaphore:
                response = await ChatGPT_API_async(model=model, prompt=prompt)
        else:
            response = await ChatGPT_API_async(model=model, prompt=prompt)
        json_result = extract_json(response)
        return str(json_result.get("is_section_boundary", "no")).strip().lower() == "yes"
    except Exception:
        return False


async def _find_title_positions_with_llm_verification(
    lines,
    offsets,
    toc_items,
    min_start_pos=0,
    model=None,
    llm_max_concurrency=5,
    max_candidates_per_title=6,
    context_window=2,
):
    candidate_lines = _collect_title_candidates(
        lines,
        toc_items,
        max_candidates_per_title=max_candidates_per_title,
    )

    # Restrict heading matching to content after TOC area.
    if min_start_pos and min_start_pos > 0:
        filtered_candidates = []
        for line_indices in candidate_lines:
            filtered_candidates.append(
                [idx for idx in line_indices if offsets[idx] >= min_start_pos]
            )
        candidate_lines = filtered_candidates
        logger.info(
            "Heading search range restricted to offsets >= %d (after TOC)",
            min_start_pos,
        )

    semaphore = asyncio.Semaphore(max(1, int(llm_max_concurrency or 5)))
    verify_tasks = []
    task_keys = []

    for title_idx, line_indices in enumerate(candidate_lines):
        title = toc_items[title_idx]["title"]
        logger.info(
            "Heading verification candidates for TOC item %d ('%s'): %d",
            title_idx + 1,
            title,
            len(line_indices),
        )
        for line_idx in line_indices:
            context = _build_line_context(lines, line_idx, window=context_window)
            verify_tasks.append(
                _verify_heading_candidate(
                    title=title,
                    context=context,
                    model=model,
                    semaphore=semaphore,
                )
            )
            task_keys.append((title_idx, line_idx))

    verified = {}
    if verify_tasks:
        results = await asyncio.gather(*verify_tasks, return_exceptions=True)
        for (title_idx, line_idx), result in zip(task_keys, results):
            verified[(title_idx, line_idx)] = (result is True)

    line_cursor = 0
    result = []
    for idx, item in enumerate(toc_items):
        selected_line_idx = None

        # Prefer verified heading candidates that keep ordering.
        for line_idx in candidate_lines[idx]:
            if line_idx < line_cursor:
                continue
            if verified.get((idx, line_idx), False):
                selected_line_idx = line_idx
                break

        # Fallback to first matched candidate if verification found nothing.
        if selected_line_idx is None:
            for line_idx in candidate_lines[idx]:
                if line_idx >= line_cursor:
                    selected_line_idx = line_idx
                    break

        start_pos = offsets[selected_line_idx] if selected_line_idx is not None else None
        if selected_line_idx is not None:
            line_cursor = selected_line_idx + 1
            logger.info(
                "Selected heading for '%s' at line %d",
                item["title"],
                selected_line_idx + 1,
            )
        else:
            logger.info("No heading match selected for '%s'", item["title"])

        result.append({
            "structure": item["structure"],
            "title": item["title"],
            "level": len(item["structure"].split(".")),
            "start_pos": start_pos,
            "text": "",
        })

    return result


def _fill_text_for_nodes(flat_nodes, full_text):
    n = len(flat_nodes)
    for i in range(n):
        start_pos = flat_nodes[i].get("start_pos")
        if start_pos is None:
            flat_nodes[i]["text"] = ""
            continue

        level_i = flat_nodes[i]["level"]
        end_pos = len(full_text)
        for j in range(i + 1, n):
            next_start = flat_nodes[j].get("start_pos")
            if next_start is None:
                continue
            if flat_nodes[j]["level"] <= level_i:
                end_pos = next_start
                break

        flat_nodes[i]["text"] = full_text[start_pos:end_pos].strip()


def _flat_to_tree_with_text(flat_nodes):
    if not flat_nodes:
        return []

    root_nodes = []
    stack = []

    for node in flat_nodes:
        tree_node = {
            "title": node["title"],
            "text": node.get("text", ""),
            "nodes": [],
        }
        current_level = node["level"]

        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, _ = stack[-1]
            parent_node["nodes"].append(tree_node)

        stack.append((tree_node, current_level))

    return root_nodes


def _fallback_chunk_title(text, max_len=80):
    first_line = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else "Section Chunk"
    first_line = re.sub(r"^[\d\W_]+", "", first_line).strip() or "Section Chunk"
    return first_line[:max_len]


def _split_text_for_summary_chunks(text, max_tokens, model=None):
    token_model = model or "deepseek-chat"
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, model=token_model)

        if para_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_tokens = 0

            approx_target_chars = max(800, int(len(para) * (max_tokens / max(para_tokens, 1))))
            sub_parts = split_text_by_chars(para, approx_target_chars)
            for sub in sub_parts:
                sub = sub.strip()
                if sub:
                    chunks.append(sub)
            continue

        if current and (current_tokens + para_tokens > max_tokens):
            chunks.append("\n\n".join(current).strip())
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current).strip())

    return [c for c in chunks if c]


async def _generate_chunk_title_and_summary(chunk_text, model=None, semaphore=None):
    prompt = f"""
You are given a chapter text chunk.

Return JSON only with:
{{
  "title": "a concise descriptive title for this chunk",
  "summary": "a concise summary for this chunk"
}}

Chunk text:
{chunk_text}
"""
    try:
        if semaphore is not None:
            async with semaphore:
                response = await ChatGPT_API_async(model=model, prompt=prompt)
        else:
            response = await ChatGPT_API_async(model=model, prompt=prompt)
        parsed = extract_json(response)
        title = str(parsed.get("title", "")).strip() if isinstance(parsed, dict) else ""
        summary = str(parsed.get("summary", "")).strip() if isinstance(parsed, dict) else ""
        if not title:
            title = _fallback_chunk_title(chunk_text)
        if not summary:
            summary = chunk_text.strip()
        return {"title": title, "summary": summary}
    except Exception:
        return {
            "title": _fallback_chunk_title(chunk_text),
            "summary": chunk_text.strip(),
        }


async def _generate_aggregate_summary_from_chunks(node_title, chunk_summaries, model=None, semaphore=None):
    prompt = f"""
You are given summaries from multiple chunks of one chapter titled: {node_title}

Generate one concise overall chapter summary.
Return plain text only.

Chunk summaries:
{json.dumps(chunk_summaries, ensure_ascii=False, indent=2)}
"""
    try:
        if semaphore is not None:
            async with semaphore:
                response = await ChatGPT_API_async(model=model, prompt=prompt)
        else:
            response = await ChatGPT_API_async(model=model, prompt=prompt)
        return str(response).strip()
    except Exception:
        return "\n".join(chunk_summaries).strip()


async def _generate_summaries_for_structure_txt(structure, summary_token_threshold, model=None, llm_max_concurrency=5):
    token_model = model or "deepseek-chat"
    semaphore = asyncio.Semaphore(max(1, int(llm_max_concurrency or 5)))

    async def process_node(node):
        if not isinstance(node, dict):
            return

        if node.get("_is_chunk_child"):
            return

        node_text = (node.get("text") or "").strip()
        if not node_text:
            if node.get("nodes"):
                for child in node.get("nodes", []):
                    await process_node(child)
            return

        token_count = count_tokens(node_text, model=token_model)

        if token_count <= summary_token_threshold:
            if node.get("nodes"):
                node["prefix_summary"] = node_text
            else:
                node["summary"] = node_text
            for child in node.get("nodes", []):
                await process_node(child)
            return

        chunk_texts = _split_text_for_summary_chunks(
            node_text,
            max_tokens=summary_token_threshold,
            model=token_model,
        )

        if len(chunk_texts) <= 1:
            summary = await generate_node_summary(node, model=model)
            if node.get("nodes"):
                node["prefix_summary"] = summary
            else:
                node["summary"] = summary
            for child in node.get("nodes", []):
                await process_node(child)
            return

        logger.info(
            "Node '%s' exceeds summary threshold (%d tokens). Split into %d chunk summaries.",
            node.get("title", "Untitled"),
            token_count,
            len(chunk_texts),
        )

        chunk_tasks = [
            _generate_chunk_title_and_summary(chunk_text, model=model, semaphore=semaphore)
            for chunk_text in chunk_texts
        ]
        chunk_results = await asyncio.gather(*chunk_tasks)

        chunk_children = []
        chunk_summaries = []
        for chunk_text, chunk_result in zip(chunk_texts, chunk_results):
            chunk_title = str(chunk_result.get("title", "")).strip() or _fallback_chunk_title(chunk_text)
            chunk_summary = str(chunk_result.get("summary", "")).strip() or chunk_text.strip()
            chunk_summaries.append(chunk_summary)
            chunk_children.append(
                {
                    "title": chunk_title,
                    "text": chunk_text,
                    "summary": chunk_summary,
                    "_is_chunk_child": True,
                }
            )

        node.setdefault("nodes", [])
        node["nodes"].extend(chunk_children)

        aggregate_summary = await _generate_aggregate_summary_from_chunks(
            node.get("title", "Untitled"),
            chunk_summaries,
            model=model,
            semaphore=semaphore,
        )
        node["prefix_summary"] = aggregate_summary

    if isinstance(structure, list):
        for root in structure:
            await process_node(root)
    elif isinstance(structure, dict):
        await process_node(structure)

    return structure


def _cleanup_chunk_child_flags(data):
    if isinstance(data, dict):
        data.pop("_is_chunk_child", None)
        if "nodes" in data and isinstance(data["nodes"], list):
            for child in data["nodes"]:
                _cleanup_chunk_child_flags(child)
    elif isinstance(data, list):
        for item in data:
            _cleanup_chunk_child_flags(item)


async def txt_to_tree(
    txt_path,
    chunk_chars=12000,
    llm_max_concurrency=5,
    if_add_node_summary='no',
    summary_token_threshold=200,
    model=None,
    if_add_doc_description='no',
    if_add_node_text='yes',
    if_add_node_id='yes',
):
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_content = f.read()

    logger.info("Starting TXT processing for %s", txt_path)

    chunks = split_text_by_chars(txt_content, chunk_chars=chunk_chars)
    if not chunks:
        chunks = [txt_content]

    logger.info(
        "Split TXT into %d chunk(s) with chunk_chars=%d",
        len(chunks),
        chunk_chars,
    )

    toc_items, toc_end_chunk_idx = _extract_toc_until_complete(chunks, model=model)

    content_start_pos = 0
    if toc_end_chunk_idx is not None:
        content_start_pos = sum(len(c) for c in chunks[: toc_end_chunk_idx + 1])
        logger.info(
            "TOC detected up to chunk %d/%d; heading search starts from char offset %d",
            toc_end_chunk_idx + 1,
            len(chunks),
            content_start_pos,
        )

    # Fallback to previous full-scan behavior if no TOC is detected.
    if not toc_items:
        logger.info("Fallback: running full continuation-based TOC extraction")
        logger.info("TOC extraction round 1/%d (initial chunk)", len(chunks))
        toc_items = _extract_toc_init_txt(chunks[0], model=model)
        toc_items = _clean_toc_items(toc_items)
        _log_toc_items("TOC round 1 cleaned", toc_items)

        for round_index, chunk in enumerate(chunks[1:], start=2):
            logger.info("TOC extraction round %d/%d (continuation)", round_index, len(chunks))
            new_items = _extract_toc_continue_txt(toc_items, chunk, model=model)
            if isinstance(new_items, list):
                _log_toc_items(f"TOC round {round_index} raw", new_items)
            toc_items = _clean_toc_items(toc_items + (new_items if isinstance(new_items, list) else []))
            _log_toc_items(f"TOC round {round_index} cleaned", toc_items)
        # Full-scan fallback uses the full document as heading search range.
        content_start_pos = 0

    logger.info("Final TXT TOC extraction complete")
    _log_toc_items("Final TOC", toc_items)

    lines = txt_content.splitlines(keepends=True)
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    flat_nodes = await _find_title_positions_with_llm_verification(
        lines,
        offsets,
        toc_items,
        min_start_pos=content_start_pos,
        model=model,
        llm_max_concurrency=llm_max_concurrency,
        max_candidates_per_title=6,
        context_window=2,
    )
    _fill_text_for_nodes(flat_nodes, txt_content)
    tree_structure = _flat_to_tree_with_text(flat_nodes)

    if if_add_node_summary == 'yes':
        tree_structure = format_structure(
            tree_structure,
            order=['title', 'node_id', 'summary', 'prefix_summary', 'text', 'nodes'],
        )
        tree_structure = await _generate_summaries_for_structure_txt(
            tree_structure,
            summary_token_threshold=summary_token_threshold,
            model=model,
            llm_max_concurrency=llm_max_concurrency,
        )

        _cleanup_chunk_child_flags(tree_structure)

        if if_add_node_id == 'yes':
            write_node_id(tree_structure)

        if if_add_node_text == 'no':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'summary', 'prefix_summary', 'nodes'],
            )

        if if_add_doc_description == 'yes':
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
                'doc_description': doc_description,
                'structure': tree_structure,
            }
    else:
        _cleanup_chunk_child_flags(tree_structure)
        if if_add_node_id == 'yes':
            write_node_id(tree_structure)
        if if_add_node_text == 'yes':
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'summary', 'prefix_summary', 'text', 'nodes'],
            )
        else:
            tree_structure = format_structure(
                tree_structure,
                order=['title', 'node_id', 'summary', 'prefix_summary', 'nodes'],
            )

    return {
        'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
        'structure': tree_structure,
    }
