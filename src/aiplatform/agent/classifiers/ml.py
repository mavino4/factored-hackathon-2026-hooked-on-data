"""Classic ML: TF-IDF of character n-grams (robust to typos, accents and es/pt/en mixed)
plus word n-grams, into a logistic regression. Needs scikit-learn (dependency group
``classifiers``). Trained by evals/classify_bench.py on evals/intents/train.jsonl."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

from aiplatform.agent.classifiers import Prediction, normalize


class TfidfClassifier:
    name = "ml"

    def __init__(self, c: float | None = None):
        self._c = c  # None: chosen by cross-validation on the training set
        self._model: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfClassifier":
        features = FeatureUnion([
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                      sublinear_tf=True)),
            ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                      sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ])
        pipeline = Pipeline([("features", features), ("model", LogisticRegression(
            max_iter=4000, class_weight="balanced"))])
        normalized = [normalize(t) for t in texts]
        if self._c is None:
            search = GridSearchCV(pipeline, {"model__C": [1.0, 5.0, 20.0, 50.0]}, cv=5,
                                  scoring="f1_macro", n_jobs=-1)
            search.fit(normalized, labels)
            self._model, self._c = search.best_estimator_, search.best_params_["model__C"]
        else:
            pipeline.set_params(model__C=self._c)
            self._model = pipeline.fit(normalized, labels)
        return self

    def predict_text(self, text: str) -> Prediction:
        assert self._model is not None, "fit first"
        probs = self._model.predict_proba([normalize(text)])[0]
        best = int(probs.argmax())
        intent = self._model.classes_[best]
        return Prediction(intent, float(probs[best]), f"tf-idf p={probs[best]:.2f}")
