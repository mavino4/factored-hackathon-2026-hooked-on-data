"""Intent from sentence embeddings: the text becomes a vector that abstracts the wording and
the language, and a classifier on those vectors decides.

Embedders (same interface: ``name``, ``embed``, ``clear``):
- ``OllamaEmbedder``: bge-m3 (1024 dimensions, multilingual) served by local Ollama
  (``ollama pull bge-m3``).
- ``OpenAIEmbedder``: OpenAI text-embedding-3-small (1536 dimensions, multilingual), with
  OPENAI_API_KEY from ``.env``. Vectors are also cached on disk, so repeated runs neither
  pay nor wait.

Classifiers on the vectors (``EmbeddingClassifier(method=...)``): logistic regression
(``logreg``), nearest-neighbour vote (``knn``), random forest (``rf``) and gradient
boosting (``boost``). Needs scikit-learn (dependency group ``classifiers``)."""

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


class OllamaEmbedder:
    name = "bge-m3"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = EMBED_MODEL,
                 batch: int = 64):
        self._client = httpx.Client(base_url=base_url, timeout=600)
        self._model, self._batch = model, batch
        self._cache: dict[str, np.ndarray] = {}

    def clear(self) -> None:
        self._cache.clear()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit vectors, one row per text."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for i in range(0, len(missing), self._batch):
            chunk = missing[i:i + self._batch]
            for attempt in range(3):  # Ollama drops the connection while swapping models
                try:
                    response = self._client.post("/api/embed", json={
                        "model": self._model, "input": chunk, "keep_alive": "30m"})
                    response.raise_for_status()
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            for text, vector in zip(chunk, response.json()["embeddings"]):
                self._cache[text] = _unit(vector)
        return np.stack([self._cache[t] for t in texts])


class OpenAIEmbedder:
    """text-embedding-3-small through the official SDK. ``tokens`` counts what was billed;
    ``read_disk = False`` makes it call the API even for texts cached on disk (to time
    real calls)."""

    name = "openai-3s"

    def __init__(self, api_key: str, *, model: str = OPENAI_EMBED_MODEL,
                 cache_dir: Path | None = None, batch: int = 256, http_client=None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, http_client=http_client)
        self._model, self._batch = model, batch
        self._cache: dict[str, np.ndarray] = {}
        self._path = cache_dir / f"{model}.npz" if cache_dir else None
        self._disk: dict[str, np.ndarray] = {}
        if self._path and self._path.exists():
            data = np.load(self._path)
            self._disk = dict(zip(data["texts"].tolist(), data["vectors"]))
        self.read_disk = True
        self.tokens = 0

    def clear(self) -> None:
        self._cache.clear()

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            texts = list(self._disk)
            np.savez(self._path, texts=np.array(texts),
                     vectors=np.stack([self._disk[t] for t in texts]))

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.read_disk:
            for t in texts:
                if t not in self._cache and t in self._disk:
                    self._cache[t] = self._disk[t]
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for i in range(0, len(missing), self._batch):
            chunk = missing[i:i + self._batch]
            response = self._client.embeddings.create(model=self._model, input=chunk)
            self.tokens += response.usage.prompt_tokens
            for item in sorted(response.data, key=lambda d: d.index):
                self._cache[chunk[item.index]] = self._disk[chunk[item.index]] = (
                    _unit(item.embedding))
        if missing:
            self._save()
        return np.stack([self._cache[t] for t in texts])


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
