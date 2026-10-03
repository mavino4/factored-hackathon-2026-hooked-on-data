"""Embedding classifiers (agent/classifiers/embeddings.py) without Ollama or OpenAI: a fake
embedder for the models, a mocked HTTP transport for the OpenAI embedder. Skipped unless
the ``classifiers`` dependency group is installed."""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from aiplatform.agent.classifiers.embeddings import EmbeddingClassifier

WORDS = {"saldo": 0, "debo": 0, "asesor": 1, "persona": 1, "hola": 2, "gracias": 2}


class FakeEmbedder:
    """A bag of known words, one dimension per intent, plus a little noise."""
    name = "fake"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def clear(self):
        pass

    def embed(self, texts):
        rows = []
        for text in texts:
            v = self.rng.normal(0, 0.05, 8).astype(np.float32)
            for word, dim in WORDS.items():
                if word in text:
                    v[dim] += 1.0
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


TRAIN = ([f"cual es mi saldo {i}" for i in range(12)] + [f"cuanto debo {i}" for i in range(12)]
         + [f"quiero un asesor {i}" for i in range(12)] + [f"una persona {i}" for i in range(12)]
         + [f"hola {i}" for i in range(12)] + [f"gracias {i}" for i in range(12)])
LABELS = ["account"] * 24 + ["human"] * 24 + ["general"] * 24


@pytest.mark.parametrize("method", ["logreg", "knn", "rf", "boost"])
def test_each_model_learns_the_vectors(method):
    classifier = EmbeddingClassifier(FakeEmbedder(), method=method).fit(TRAIN, LABELS)
    assert classifier.name == f"fake-{method}"
    for text, intent in (("mi saldo por favor", "account"), ("pasame un asesor", "human"),
                         ("hola!", "general")):
        prediction = classifier.predict_text(text)
        assert prediction.intent == intent
        assert 0.0 < prediction.confidence <= 1.0


def test_openai_embedder_batches_normalizes_and_caches(tmp_path):
    httpx2 = pytest.importorskip("httpx2")
    pytest.importorskip("openai")
    from aiplatform.agent.classifiers.embeddings import OpenAIEmbedder

    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        data = [{"object": "embedding", "index": i, "embedding": [3.0, 4.0] if "a" in t
                 else [0.0, 2.0]} for i, t in enumerate(body["input"])]
        return httpx2.Response(200, json={
            "object": "list", "data": data, "model": body["model"],
            "usage": {"prompt_tokens": 7, "total_tokens": 7}})

    def embedder():
        return OpenAIEmbedder("sk-test", cache_dir=tmp_path,
                              http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))

    first = embedder()
    vectors = first.embed(["a", "b", "a"])
    assert [c["input"] for c in calls] == [["a", "b"]]  # one request, no duplicates
    assert calls[0]["model"] == "text-embedding-3-small"
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0], [0.6, 0.8]], rtol=1e-6)
    assert first.tokens == 7

    second = embedder()  # a new run reads the disk cache
    second.embed(["a", "b"])
    assert len(calls) == 1
    second.clear()
    second.read_disk = False  # timing real calls: ask the API again
    second.embed(["b"])
    assert [c["input"] for c in calls] == [["a", "b"], ["b"]]
