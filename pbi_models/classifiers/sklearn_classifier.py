from abc import ABC
from sklearn.utils import all_estimators
import importlib


class SklearnClassifier(ABC):
    """
    Base class for classifiers based on Scikit-learn plus LightGBM and XGBoost. All (scikit-learn) classifiers should inherit from this class and implement the fit and predict methods.
    """

    def _get_sklearn_classifier(self, model_name: str) -> type:
        # Special case for LGBM & XGB
        if model_name == "LGBMClassifier":
            from lightgbm import LGBMClassifier

            return LGBMClassifier
        elif model_name == "XGBClassifier":
            from xgboost import XGBClassifier

            return XGBClassifier

        # Generic sklearn models
        estimators = all_estimators(type_filter="classifier")
        for name, class_ in estimators:
            if name == model_name:
                module_name = str(class_).split("'")[1].split(".")[1]
                class_name = class_.__name__
                return getattr(
                    importlib.import_module(f"sklearn.{module_name}"), class_name
                )

        raise ValueError(f"Incorrect model name ({model_name}) for SklearnClassifier.")

    def __init__(
        self,
        bacterium_embed_dim: int,
        phage_embed_dim: int,
        sklearn_model_name: str,
        sklearn_model_params: dict,
        kmer_dim: int = 0,
    ) -> None:
        self.sklearn_model_name = sklearn_model_name
        self.sklearn_model_params = sklearn_model_params
        self.kmer_dim = kmer_dim

        self.sklearn_model = self._get_sklearn_classifier(sklearn_model_name)(
            **sklearn_model_params
        )

    def fit(self, *args, **kwargs):
        return self.sklearn_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.sklearn_model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.sklearn_model.predict_proba(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return getattr(self.sklearn_model, name)
        except AttributeError:
            raise AttributeError(f"'SklearnClassifier' object and its underlying '{self.sklearn_model_name}' have no attribute '{name}'")

    def reset_model(self, *args, **kwargs):
        self.sklearn_model = self._get_sklearn_classifier(self.sklearn_model_name)(
            **self.sklearn_model_params
        )

    def name(self) -> str:
        return f"SklearnClassifier-{self.sklearn_model_name}"

    def __repr__(self):
        return f"SklearnClassifier(sklearn_model={self.sklearn_model_name}({self.sklearn_model_params}))"
