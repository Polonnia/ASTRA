# Dual-Level Graph Build and Retrieval Flow

## Scope

This document explains:

1. How dual-level graph data is built from input documents.
2. How GDB and VDB are updated.
3. How retrieval uses those data structures in each query mode.
4. A completeness check of the current implementation.
5. How to use the feature in practice.

## End-to-End Ingestion Pipeline

## 1) Insert entrypoint

`ASTRA.ainsert()` performs the following high-level stages:

1. Build `new_docs` and de-duplicate by doc hash id.
2. Chunk documents into `inserting_chunks`.
3. Run base entity extraction (`entity_extraction_func`, default `extract_entities`) and upsert to graph/VDB.
4. Optionally run dual-level extraction (`build_dual_level_graphs`) when:
   - `enable_structural_graph == True`
   - `language == "english"`
5. Persist `full_docs` and `text_chunks` KV stores.

Relevant code:

- `astra/astra.py` in `ASTRA.ainsert()`
- `astra/_op.py` in `extract_entities()` and `build_dual_level_graphs()`

## 2) Base extraction (content chunks)

`extract_entities()` and `extract_hierarchical_entities()` do:

1. LLM extraction from chunk text.
2. Merge entities and relations into GDB:
   - entities: `_merge_nodes_then_upsert()`
   - relations: `_merge_edges_then_upsert()`
3. Upsert entity and relation vectors into entities VDB.

Current relation VDB key (after latest changes):

- base path: `rel-{md5(source|target)}`

Relation description is used in `content` for embedding, but not in the key.

## 3) Dual-level extraction (structural + content)

`build_dual_level_graphs()` does:

1. Parse structured JSON from `new_docs` (`structure` tree).
2. Structural pass (level=1):
   - For each structure unit summary, extract entities/relations.
   - Extract cross-unit relations between parent and child units.
   - Merge into GDB.
3. Content pass (level=0):
   - For each leaf chunk aligned to structure unit, extract entities/relations with unit context.
   - Merge into GDB.
4. Export level-specific collections:
   - `structural_graph_entities`, `structural_graph_relations`
   - `content_graph_entities`, `content_graph_relations`
5. Upsert mixed vectors into entities VDB.

Current dual-level VDB keys:

- entity: `ent-{md5(entity|graph_level|entity_name)}`
- relation: `rel-{md5(relation|graph_level|source|target)}`

This means:

- Same entity name in same level overwrites/merges one VDB record.
- Same entity name across different levels keeps separate VDB records.
- Same relation endpoints in same level overwrite one VDB record.
- Same relation endpoints across different levels keep separate VDB records.

## Merge Semantics

## GDB entity merge

Entity merge key is `entity_name` only.

Merged fields include:

1. `description` (set union and optional summary normalization)
2. `source_id` (set union)
3. `section` ids (set union)
4. `entity_type` (set union)
5. `level` (max level)

Implication:

- Same name across levels becomes one GDB node.
- Resulting node level tends to be the max observed level.

## GDB relation merge

Relation merge key is undirected `(sorted(src, tgt))`.

Merged fields include:

1. `weight` (sum)
2. `description` (set union and optional summary normalization)
3. `source_id` (set union)
4. `order` (min)

Implication:

- Structural/content relations with same endpoints merge into one GDB edge.

## Retrieval Pipeline by Query Mode

## naive

1. Query `chunks_vdb` by user query.
2. Fetch chunk text from `text_chunks` by returned ids.
3. Build final answer from chunk content.

## hi / hi_local / hi_global / hi_bridge / hi_nobridge

1. Query `entities_vdb` (entity hits).
2. Resolve entities from GDB (`get_node`, `node_degree`).
3. Expand to related chunks via entity `source_id` and graph neighborhood.
4. Expand to related edges from GDB.
5. Build context and call answer model.

Important:

- These modes do not strictly filter by `graph_level` today.
- Because GDB merges cross-level same-name nodes/edges, the retrieval view is mostly merged graph semantics.

## tree

1. Decompose query into:
   - `detail_subquestions`
   - `macro_subquestions`
   - `keywords`
2. Detail retrieval:
   - VDB query for detail subquestions only.
   - Entity hits are resolved to GDB node descriptions.
   - Relation hits use VDB row fields directly.
3. Keyword retrieval:
   - Direct graph entity-name matching (no VDB).
   - Merged with detail entity evidence by entity name.
4. Macro retrieval:
   - LLM-based document selection from `doc_description`.
   - LLM-guided structure-tree traversal per selected document.
   - Returns organized macro evidence (`useful_information`).
5. Final answer model uses the assembled context sections.

## Completeness Check (Current State)

## Implemented and complete

1. Dual-level build gate by config and language.
2. Structural + content extraction passes.
3. GDB merge logic for entities and relations.
4. VDB upsert logic for mixed dual-level records.
5. Tree retrieval with macro/doc-selection traversal.
6. Keyword path moved to direct entity-name graph matching.
7. Detail+keyword entity dedup merge in tree context output.

## Notable behavior to be aware of

1. GDB does not preserve separate structural/content nodes for same entity name.
   - It merges by entity name.
2. GDB relation level is not stored as an explicit field.
   - Endpoint-matched edges are merged.
3. Hi* retrieval modes do not enforce per-level filtering.
   - They operate on merged graph plus mixed VDB entity recall.
4. Tree detail relation evidence is taken from VDB relation hit rows, not re-resolved from GDB edges.

These are design choices in current code, not runtime faults.

## Usage Guide

## A) Minimum config for dual-level build

Set at initialization:

- `enable_structural_graph=True`
- `language="english"`
- `enable_local=True` (entities VDB required)

Example:

```python
from astra.astra import ASTRA

rag = ASTRA(
    working_dir="./astra_cache",
    enable_local=True,
    enable_hierachical_mode=True,
    enable_structural_graph=True,
    language="english",
)
```

## B) Input requirements for structural pass

For structural extraction from `full_docs`, each document content should be valid JSON with:

- `doc_name`
- `doc_description` (recommended)
- `structure` (list)
- per-node fields such as:
  - `node_id` (recommended)
  - `title`
  - `summary` and/or `text`
  - child list in `nodes` (or `children`)

## C) Insert data

```python
with open("path/to/your_structure.json", "r", encoding="utf-8") as f:
    content = f.read()

rag.insert(content)
```

## D) Query modes

```python
from astra.base import QueryParam

# Tree mode
answer = rag.query("your question", QueryParam(mode="tree"))

# Hierarchical modes
answer_hi = rag.query("your question", QueryParam(mode="hi"))
answer_local = rag.query("your question", QueryParam(mode="hi_local"))
answer_bridge = rag.query("your question", QueryParam(mode="hi_bridge"))
```

## E) Output artifacts to inspect

Under `working_dir` you should see:

- `kv_store_full_docs.json`
- `kv_store_text_chunks.json`
- `vdb_entities.json`
- graph storage artifacts (backend-dependent)

If export is enabled (`export_workbench_json=True`), additional workbench JSON files are generated.

## Quick Validation Checklist

1. Insert logs show both:
   - `[Entity Extraction]`
   - `[Dual-Level Graph Extraction]`
2. `vdb_entities.json` contains entity/relation records with `graph_level` metadata.
3. Tree mode context includes:
   - `Merged Entity Evidence`
   - `Detail Relation Evidence`
   - `Macro Structure Evidence`
4. Re-inserting updated relation descriptions overwrites same relation key in VDB (same level+endpoints).

