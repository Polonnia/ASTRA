<div align="center">

# ASTRA: Adaptive Structure and Topic-aware Retrieval-Augmented Generation



## Install

```bash
cd ASTRA
pip install -e .
```

## Quick Start

You can just utilize the following code to perform a query with ASTRA.

```python
from astra import ASTRA, QueryParam

graph_func = ASTRA(
    working_dir="./your_work_dir",
    enable_llm_cache=True,
    enable_hierachical_mode=True, 
    embedding_batch_num=6,
    embedding_func_max_async=8,
    )
# indexing
with open("path_to_your_context.txt", "r") as f:
    graph_func.insert(f.read())
# retrieval & generation
print("Perform dual search:")
print(graph_func.query("The question you want to ask?", param=QueryParam(mode="dual")))
```

Or if you want to employ ASTRA with DeepSeek, ChatGLM, or other third-party retrieval api, here are the examples in `./astra_search_deepseek.py`, `./astra_search_glm.py`, and `./astra_search_openai.py`. The API keys and the LLM configurations can be set at `./config.yaml`.

The entry scripts read documents from the repository `./files/` directory. Pass only the file name when running them, for example:

```shell
python astra_search_openai.py your_context.txt
python astra_search_glm.py your_context.txt
python astra_search_deepseek.py your_context.txt
```


## Configuration

You can configure model endpoints and runtime options in `./config.yaml`:

- LLM provider settings (`openai`, `deepseek`, `glm`)
- Embedding model settings
- Retrieval and token-budget controls
- Working directory behavior

For third-party APIs, set API key, model name, and base URL in the corresponding section.

## Core Usage

ASTRA supports indexing and question answering in one workflow:

```python
from astra import ASTRA, QueryParam

graph_func = ASTRA(
    working_dir="./your_work_dir",
    enable_llm_cache=True,
    embedding_batch_num=6,
    embedding_func_max_async=8
)

# 1) Index
with open("path_to_your_context.txt", "r", encoding="utf-8") as f:
    graph_func.insert(f.read())

# 2) Query
answer = graph_func.query(
    "The question you want to ask?",
    param=QueryParam(mode="dual"),
)
print(answer)
```

## Query Modes

Common query modes include:

- `dual`: ASTRA dual retrieval mode.
- `naive`: naive retrieval baseline mode.

Default mode is `dual`.

## Running Entry Scripts

Example scripts are provided for different LLM providers:

- `astra_search_openai.py`
- `astra_search_glm.py`
- `astra_search_deepseek.py`

Each script reads source files from `./files/` by file name:

```shell
python astra_search_openai.py your_context.txt
python astra_search_glm.py your_context.txt
python astra_search_deepseek.py your_context.txt
```

## Project Layout

- `astra/`: core library implementation.
- `pageindex/`: page-index processing utilities.
- `files/`: example input documents of the computer science category from the Ultradomain dataset.