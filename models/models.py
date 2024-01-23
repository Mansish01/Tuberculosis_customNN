from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

MODELS = {
    "kneighbors": KNeighborsClassifier.__name__,
    "logistic": LogisticRegression.__name__,   
}


class ModelZoo:
    def __init__(self, **model_config):
        model_name = model_config["name"]
        assert model_name in    MODELS.keys()
        model_classname =   MODELS[model_name]
        model_args = model_config["args"] if model_config["args"] else {}
        self.model = globals()[model_classname](**model_args)

    def get_model(self):
        return self.model

