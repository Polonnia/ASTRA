import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from astra import ASTRA, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from astra.base import BaseKVStorage
from astra._utils import compute_args_hash
from tqdm import tqdm

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASTRA search with a file from ./files")
    parser.add_argument(
        "file_name",
        help="File name inside the repository files/ directory, for example demo.txt",
    )
    return parser.parse_args()


def resolve_input_file(file_name: str) -> Path:
    input_path = Path(__file__).resolve().parent / "files" / file_name
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return input_path

# Extract configurations
GLM_API_KEY = config['glm']['api_key']
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']
GLM_URL = config['glm']['base_url']


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['glm_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = "embedding-3"
    client = AsyncOpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_URL
    ) 
    embedding = await client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


graph_func = ASTRA(
    working_dir=config['astra']['working_dir'],
    enable_llm_cache=config['astra']['enable_llm_cache'],
    embedding_func=GLM_embedding,
    best_model_func=deepseepk_model_if_cache,
    cheap_model_func=deepseepk_model_if_cache,
    embedding_batch_num=config['astra']['embedding_batch_num'],
    embedding_func_max_async=config['astra']['embedding_func_max_async'],
    enable_naive_rag=config['astra']['enable_naive_rag'],
    chunk_token_size=config['astra']['chunk_token_size'],
    entity_extract_max_gleaning=config['astra']['entity_extract_max_gleaning'],
    enable_tree_traversal=config['astra'].get('enable_tree_traversal', False),
    detail_top_k_entity=config['astra'].get('detail_top_k_entity', 10),
    detail_top_k_relation=config['astra'].get('detail_top_k_relation', 10),
    macro_top_k_entity=config['astra'].get('macro_top_k_entity', 10),
    macro_top_k_relation=config['astra'].get('macro_top_k_relation', 10),
    keyword_top_k_entity=config['astra'].get('keyword_top_k_entity', 10),
    keyword_top_k_relation=config['astra'].get('keyword_top_k_relation', 10),
    max_token_for_text_unit=config['astra'].get('max_token_for_text_unit', 20000),
    max_token_for_graph=config['astra'].get('max_token_for_graph', 20000),
    max_token_for_summary=config['astra'].get('max_token_for_summary', 20000),
    )

args = parse_args()
input_path = resolve_input_file(args.file_name)

# comment this if the working directory has already been indexed
with open(input_path, "r", encoding="utf-8") as f:
    graph_func.insert(f.read())

print("Perform dual search:")
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="dual")))
