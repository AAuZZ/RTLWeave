from __future__ import annotations

import numpy as np
import hashlib
import os
import sqlite3
import torch
from transformers import AutoModel, AutoTokenizer
try:
    from term import warn as twarn
except Exception:  # pragma: no cover - package import path
    from .term import warn as twarn


class Embedding:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        if not cls._initialized:
            try:
                cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_name = "jinaai/jina-embeddings-v2-base-code"
                cls._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
                cls._model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True).to(cls._device)
                cls._model.eval()
                cls._initialized = True
            except Exception as e:
                twarn(f"Embedding init failed: {e}")
                raise
        return cls._instance

    def __init__(self):
        cache_path = os.getenv("KGVERI_EMBEDDING_CACHE")
        if cache_path and not hasattr(self, "_cache"):
            self._cache = sqlite3.connect(cache_path, timeout=60, check_same_thread=False)
            self._cache.execute("PRAGMA journal_mode=WAL")
            self._cache.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
            self._cache.commit()
        return

    def get_embedding(self, text):
        try:
            if text is None:
                return None
            if not isinstance(text, str):
                text = str(text)
            if not text.strip():
                return None
            cache = getattr(self, "_cache", None)
            key = hashlib.sha256(("jina-v2-base-code:512:mean:fp32:v1\n" + text).encode()).hexdigest()
            if cache is not None:
                row = cache.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
                if row:
                    return np.frombuffer(row[0], dtype=np.float32).tolist()
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Embedding model is not initialized")

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                summed = torch.sum(hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                embedding = summed / counts
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            vector = embedding[0].detach().cpu().numpy().astype(np.float32)
            if cache is not None:
                cache.execute("INSERT OR IGNORE INTO vectors VALUES (?, ?)", (key, vector.tobytes()))
                cache.commit()
            return vector.tolist()
        except Exception as e:
            twarn(f"Error getting embedding: {e}")
            return None

    def _cos_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def text_similarity(self, text1, text2):
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        return self._cos_similarity(vec1, vec2)
