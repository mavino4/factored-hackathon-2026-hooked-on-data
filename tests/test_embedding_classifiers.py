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


def test_openai_embedder_embeds_each_text_once_and_measures_it_once(tmp_path):
    httpx2 = pytest.importorskip("httpx2")
    pytest.importorskip("openai")
    from aiplatform.agent.classifiers.embeddings import OpenAIEmbedder

    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["input"])
        data = [{"object": "embedding", "index": i, "embedding": [3.0, 4.0] if "a" in t
                 else [0.0, 2.0]} for i, t in enumerate(body["input"])]
        return httpx2.Response(200, json={
            "object": "list", "data": data, "model": body["model"],
            "usage": {"prompt_tokens": 7, "total_tokens": 7}})

    def embedder():
        return OpenAIEmbedder("sk-test", cache_dir=tmp_path,
                              http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))

    first = embedder()
    vectors = first.embed(["a", "b", "a"])  # training: one batch, no duplicates
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0], [0.6, 0.8]], rtol=1e-6)
    measured = first.timed_embed(["c", "d"])  # test: one call per message, timed
    assert calls == [["a", "b"], ["c"], ["d"]]
    assert measured["c"]["tokens"] == 7 and measured["c"]["ms"] >= 0
    assert first.tokens == 21

    second = embedder()  # a later run: vectors, latency and tokens come from disk
    second.embed(["a", "b"])
    assert second.timed_embed(["c"]) == {"c": measured["c"]}
    assert len(calls) == 3 and second.tokens == 0
