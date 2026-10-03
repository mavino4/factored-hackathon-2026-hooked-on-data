"""Intent from sentence embeddings: the text becomes a vector that abstracts the wording and
the language, and a classifier on those vectors decides.

Embedders (same interface: ``name``, ``embed``, ``clear``):
- ``OllamaEmbedder``: bge-m3 (1024 dimensions, multilingual) served by local Ollama
  (``ollama pull bge-m3``).
- ``OpenAIEmbedder``: OpenAI text-embedding-3-small (1536 dimensions, multilingual), with
  OPENAI_API_KEY from ``.env``.

Both cache every vector on disk (``CachedEmbedder``): a message is embedded once, the same
vector serves every model on top, and repeated runs neither pay nor wait.

Classifiers on the vectors (``EmbeddingClassifier(method=...)``): logistic regression
(``logreg``), nearest-neighbour vote (``knn``), random forest (``rf``) and gradient
boosting (``boost``). Needs scikit-learn (dependency group ``classifiers``)."""

import json
import time
from pathlib import Path

import httpx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from aiplatform.agent.classifiers import Prediction, normalize

EMBED_MODEL = "bge-m3"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_EMBED_USD_PER_MTOK = 0.02


def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


class CachedEmbedder:
    """Vectors cached in memory and, with ``cache_dir``, on disk: a text is embedded once,
    whatever classifier uses it and however often the bench runs. ``timed_embed`` embeds
    one text per call and remembers each call's latency and tokens, so a real per-message
    cost is measured once and then read back. Subclasses fetch a batch of vectors."""

    name = "cached"
    file_stem = "cached"

    def __init__(self, cache_dir: Path | None = None, batch: int = 64):
        self._batch = batch
        self._cache: dict[str, np.ndarray] = {}
        self._meta: dict[str, dict] = {}  # text -> {"ms": latency, "tokens": billed}
        self._path = cache_dir / f"{self.file_stem}.npz" if cache_dir else None
        self.tokens = 0  # billed during this run
        if self._path and self._path.exists():
            try:
                data = np.load(self._path)
                self._cache = dict(zip(data["texts"].tolist(), data["vectors"]))
                meta = self._path.with_suffix(".json")
                if meta.exists():
                    self._meta = json.loads(meta.read_text())
            except (OSError, ValueError, KeyError) as exc:  # e.g. cut mid-write
                print(f"ignoring a damaged embeddings cache {self._path}: {exc}")
                self._cache, self._meta = {}, {}

    def _fetch(self, texts: list[str]) -> tuple[list, int]:
        """Raw vectors for ``texts`` and the tokens billed."""
        raise NotImplementedError

    def clear(self) -> None:
        self._cache.clear()

    def _save(self) -> None:
        if not self._path or not self._cache:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        texts = list(self._cache)
        tmp = self._path.with_name(self._path.stem + ".tmp.npz")
        np.savez(tmp, texts=np.array(texts), vectors=np.stack([self._cache[t] for t in texts]))
        tmp.replace(self._path)  # atomic: a killed run never leaves half a file
        meta, tmp_meta = self._path.with_suffix(".json"), self._path.with_suffix(".tmp.json")
        tmp_meta.write_text(json.dumps(self._meta, ensure_ascii=False))
        tmp_meta.replace(meta)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit vectors, one row per text; only missing texts are fetched, in batches."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for i in range(0, len(missing), self._batch):
            chunk = missing[i:i + self._batch]
            vectors, tokens = self._fetch(chunk)
            self.tokens += tokens
            for text, vector in zip(chunk, vectors):
                self._cache[text] = _unit(vector)
        if missing:
            self._save()
        return np.stack([self._cache[t] for t in texts])

    def timed_embed(self, texts: list[str]) -> dict[str, dict]:
        """Embed each text in its own call, as a live message would be, and return its
        latency (ms) and tokens. Measured once per text; later runs read them back."""
        todo = [t for t in dict.fromkeys(texts) if "ms" not in self._meta.get(t, {})]
        for text in todo:
            start = time.perf_counter()
            vectors, tokens = self._fetch([text])
            self._meta[text] = {"ms": (time.perf_counter() - start) * 1000, "tokens": tokens}
            self._cache[text] = _unit(vectors[0])
            self.tokens += tokens
        if todo:
            self._save()
        return {t: self._meta[t] for t in texts}


class OllamaEmbedder(CachedEmbedder):
    """bge-m3 on local Ollama (free)."""

    name = "bge-m3"
    file_stem = "bge-m3"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = EMBED_MODEL,
                 cache_dir: Path | None = None, batch: int = 64):
        super().__init__(cache_dir, batch)
        self._client = httpx.Client(base_url=base_url, timeout=600)
        self._model = model

    def _fetch(self, texts: list[str]) -> tuple[list, int]:
        for attempt in range(3):  # Ollama drops the connection while swapping models
            try:
                response = self._client.post("/api/embed", json={
                    "model": self._model, "input": texts, "keep_alive": "30m"})
                response.raise_for_status()
                body = response.json()
                return body["embeddings"], body.get("prompt_eval_count", 0)
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")


class OpenAIEmbedder(CachedEmbedder):
    """OpenAI text-embedding-3-small through the official SDK (billed per token)."""

    name = "openai-3s"
    file_stem = OPENAI_EMBED_MODEL

    def __init__(self, api_key: str, *, model: str = OPENAI_EMBED_MODEL,
                 cache_dir: Path | None = None, batch: int = 256, http_client=None):
        from openai import OpenAI

        super().__init__(cache_dir, batch)
        self._client = OpenAI(api_key=api_key, http_client=http_client)
        self._model = model

    def _fetch(self, texts: list[str]) -> tuple[list, int]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        return vectors, response.usage.prompt_tokens


# Small grids, chosen by 5-fold cross-validation on the training set only (3 for boosting,
# which is the slowest to fit on 1-1.5k dimensions).
MODELS = {
    "logreg": (LogisticRegression(max_iter=4000, class_weight="balanced"),
               {"C": [0.5, 2.0, 8.0, 32.0]}, 5),
    "rf": (RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=0),
           {"n_estimators": [300, 600], "max_features": ["sqrt", 0.1]}, 5),
    "boost": (HistGradientBoostingClassifier(early_stopping=True, random_state=0,
                                             class_weight="balanced"),
              {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}, 3),
}


class EmbeddingClassifier:
    def __init__(self, embedder, method: str = "logreg", k: int = 7):
        self.name = f"{embedder.name}-{method}"
        self._embedder, self._method, self._k = embedder, method, k
        self._model = None
        self._vectors: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        # Same cleaning as the other classifiers (homoglyphs, zero-width).
        return self._embedder.embed([normalize(t) for t in texts])

    def fit(self, texts: list[str], labels: list[str]) -> "EmbeddingClassifier":
        vectors = self._embed(texts)
        if self._method == "knn":
            self._vectors, self._labels = vectors, np.asarray(labels)
        else:
            estimator, grid, folds = MODELS[self._method]
            search = GridSearchCV(estimator, grid, cv=folds, scoring="f1_macro", n_jobs=-1)
            self._model = search.fit(vectors, labels).best_estimator_
        return self

    def predict_text(self, text: str) -> Prediction:
        vector = self._embed([text])
        if self._method == "knn":
            similarity = self._vectors @ vector[0]
            top = np.argsort(similarity)[-self._k:]
            votes: dict[str, float] = {}
            for i in top:
                votes[self._labels[i]] = votes.get(self._labels[i], 0.0) + float(similarity[i])
            intent = max(votes, key=votes.get)
            confidence = votes[intent] / sum(votes.values())
            return Prediction(intent, confidence,
                              f"knn {self._k}, nearest {float(similarity[top[-1]]):.2f}")
        probs = self._model.predict_proba(vector)[0]
        best = int(probs.argmax())
        return Prediction(self._model.classes_[best], float(probs[best]),
                          f"{self._method} p={probs[best]:.2f}")
