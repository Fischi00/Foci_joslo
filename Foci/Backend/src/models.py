from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_model_zoo():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "Logistic": LogisticRegression(max_iter=3000),
        #"Svm": SVC(probability=True),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(50,50), early_stopping=True, max_iter=3000)
    }