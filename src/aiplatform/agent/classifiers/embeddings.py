"""Intent from bge-m3 sentence embeddings (multilingual, served by Ollama): the text is
turned into a 1024-dimension vector that abstracts the wording and the language, and a
logistic regression (or a nearest-neighbour vote, ``knn``) on those vectors decides.

Needs Ollama with ``bge-m3`` (``ollama pull bge-m3``) and scikit-learn (dependency group
``classifiers``). Vectors are cached per text, so training and repeated runs embed each
message once."""

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from aiplatform.agent.classifiers import Prediction, normalize

EMBED_MODEL = "bge-m3"


class OllamaEmbedder:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = EMBED_MODEL,
                 batch: int = 64):
        self._client = httpx.Client(base_url=base_url, timeout=600)
        self._model, self._batch = model, batch
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit vectors, one row per text."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for i in range(0, len(missing), self._batch):
            chunk = missing[i:i + self._batch]
            response = self._client.post("/api/embed", json={
                "model": self._model, "input": chunk, "keep_alive": "30m"})
            response.raise_for_status()
            for text, vector in zip(chunk, response.json()["embeddings"]):
                v = np.asarray(vector, dtype=np.float32)
                self._cache[text] = v / np.linalg.norm(v)
        return np.stack([self._cache[t] for t in texts])


class EmbeddingClassifier:
    def __init__(self, embedder: OllamaEmbedder, method: str = "logreg", k: int = 7):
        self.name = f"embeddings-{method}"
        self._embedder, self._method, self._k = embedder, method, k
        self._model: LogisticRegression | None = None
        self._vectors: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        # Same cleaning as the other classifiers (homoglyphs, zero-width), case kept.
        return self._embedder.embed([normalize(t) for t in texts])

    def fit(self, texts: list[str], labels: list[str]) -> "EmbeddingClassifier":
        vectors = self._embed(texts)
        if self._method == "knn":
            self._vectors, self._labels = vectors, np.asarray(labels)
        else:
            search = GridSearchCV(LogisticRegression(max_iter=4000, class_weight="balanced"),
                                  {"C": [0.5, 2.0, 8.0, 32.0]}, cv=5, scoring="f1_macro",
                                  n_jobs=-1)
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
                          f"embeddings p={probs[best]:.2f}")
