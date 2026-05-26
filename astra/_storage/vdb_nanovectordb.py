import asyncio
import os
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )
        self.hybrid_cosine_weight = float(
            self.global_config.get("hybrid_cosine_weight", 0.7)
        )
        self.hybrid_bm25_weight = float(
            self.global_config.get("hybrid_bm25_weight", 0.3)
        )
        total_weight = self.hybrid_cosine_weight + self.hybrid_bm25_weight
        if total_weight <= 0:
            self.hybrid_cosine_weight = 0.7
            self.hybrid_bm25_weight = 0.3
        elif abs(total_weight - 1.0) > 1e-9:
            self.hybrid_cosine_weight /= total_weight
            self.hybrid_bm25_weight /= total_weight

        self._text_store_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}_text.json"
        )
        self._text_store: dict[str, str] = {}
        if os.path.exists(self._text_store_file_name):
            try:
                with open(self._text_store_file_name, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._text_store = {
                        str(k): str(v) for k, v in loaded.items() if str(v).strip()
                    }
            except Exception as e:
                logger.warning(
                    "Failed to load text store for hybrid retrieval (%s): %s",
                    self.namespace,
                    e,
                )

    @staticmethod
    def _tokenize_for_bm25(text: str) -> list[str]:
        # Keep English words/numbers and CJK characters to support multilingual retrieval.
        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower())

    def _bm25_scores(self, query: str) -> dict[str, float]:
        if not self._text_store:
            return {}

        tokenized_docs: dict[str, list[str]] = {}
        doc_lengths: dict[str, int] = {}
        df: Counter = Counter()

        for doc_id, text in self._text_store.items():
            tokens = self._tokenize_for_bm25(text)
            if not tokens:
                continue
            tokenized_docs[doc_id] = tokens
            doc_lengths[doc_id] = len(tokens)
            df.update(set(tokens))

        if not tokenized_docs:
            return {}

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return {}

        k1 = 1.5
        b = 0.75
        n_docs = len(tokenized_docs)
        avgdl = sum(doc_lengths.values()) / max(n_docs, 1)
        query_tf = Counter(query_tokens)

        scores: dict[str, float] = {}
        for doc_id, tokens in tokenized_docs.items():
            tf = Counter(tokens)
            dl = doc_lengths[doc_id]
            score = 0.0
            for term, qf in query_tf.items():
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                n_t = df.get(term, 0)
                idf = math.log(1 + (n_docs - n_t + 0.5) / (n_t + 0.5))
                denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
                score += idf * ((f * (k1 + 1)) / max(denom, 1e-9)) * qf
            if score > 0:
                scores[doc_id] = score
        return scores

    def _bm25_scores_from_payload(self, query: str, payloads: list[dict]) -> dict[str, float]:
        temp_text_store: dict[str, str] = {}
        for item in payloads:
            doc_id = str(item.get("__id__", "")).strip()
            if not doc_id:
                continue
            text_parts = [
                str(item.get("content", "")).strip(),
                str(item.get("entity_name", "")).strip(),
                str(item.get("source", "")).strip(),
                str(item.get("target", "")).strip(),
                str(item.get("description", "")).strip(),
            ]
            text = "\n".join([part for part in text_parts if part])
            if text:
                temp_text_store[doc_id] = text

        if not temp_text_store:
            return {}

        original_store = self._text_store
        try:
            self._text_store = temp_text_store
            return self._bm25_scores(query)
        finally:
            self._text_store = original_store

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_v = min(values)
        max_v = max(values)
        if abs(max_v - min_v) < 1e-12:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        tickers = ["-", "\\", "|", "/"]
        total_records = len(list_data)
        written_records = 0
        all_results = []

        for batch_idx, embedding_batch in enumerate(embeddings_list):
            start = batch_idx * self._max_batch_size
            end = min(start + len(embedding_batch), total_records)
            data_batch = list_data[start:end]
            for i, d in enumerate(data_batch):
                d["__vector__"] = embedding_batch[i]

            batch_result = self._client.upsert(datas=data_batch)
            if isinstance(batch_result, list):
                all_results.extend(batch_result)
            else:
                all_results.append(batch_result)

            written_records += len(data_batch)
            ticker = tickers[batch_idx % len(tickers)]
            print(
                f"{ticker} [VectorDB:{self.namespace}] Inserted {written_records}({written_records * 100 // total_records}%) vectors\r",
                end="",
                flush=True,
            )

        for k, v in data.items():
            content = str(v.get("content", "")).strip()
            if content:
                self._text_store[str(k)] = content

        print()  # clear the progress bar
        return all_results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        vector_results = self._client.query(
            query=embedding,
            top_k=max(top_k * 4, top_k),
            better_than_threshold=self.cosine_better_than_threshold,
        )
        vector_scores = {
            str(dp["__id__"]): float(dp.get("__metrics__", 0.0))
            for dp in vector_results
            if "__id__" in dp
        }
        bm25_scores = self._bm25_scores(query)
        if not bm25_scores and vector_results:
            # Fallback for old indexes that do not yet have persisted text store.
            bm25_scores = self._bm25_scores_from_payload(query, vector_results)

        vector_norm = self._normalize_scores(vector_scores)
        bm25_norm = self._normalize_scores(bm25_scores)

        candidate_ids = set(vector_scores.keys())
        if bm25_scores:
            top_bm25_ids = sorted(bm25_scores, key=bm25_scores.get, reverse=True)[: max(top_k * 4, top_k)]
            candidate_ids.update(top_bm25_ids)

        if not candidate_ids:
            return []

        vector_payload = {str(dp["__id__"]): dp for dp in vector_results if "__id__" in dp}
        merged = []
        for doc_id in candidate_ids:
            cosine_score = vector_norm.get(doc_id, 0.0)
            bm25_score = bm25_norm.get(doc_id, 0.0)
            fused_score = (
                self.hybrid_cosine_weight * cosine_score
                + self.hybrid_bm25_weight * bm25_score
            )
            payload = vector_payload.get(doc_id, {"__id__": doc_id})
            merged.append(
                {
                    **payload,
                    "id": doc_id,
                    "distance": fused_score,
                    "cosine_score": vector_scores.get(doc_id, 0.0),
                    "bm25_score": bm25_scores.get(doc_id, 0.0),
                }
            )

        merged.sort(key=lambda x: x["distance"], reverse=True)
        return merged[:top_k]

    async def index_done_callback(self):
        self._client.save()
        try:
            with open(self._text_store_file_name, "w", encoding="utf-8") as f:
                json.dump(self._text_store, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(
                "Failed to save text store for hybrid retrieval (%s): %s",
                self.namespace,
                e,
            )
