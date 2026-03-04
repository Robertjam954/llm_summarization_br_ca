import os
import sys

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_and_prepare(data_path="processed.cleveland.data"):
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(data_path, header=None)
    df.columns = [
        "age",
        "sex",
        "cp",
        "restbp",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "hd",
    ]

    df.loc[df["ca"] == "?", "ca"] = 0
    df.loc[df["thal"] == "?", "thal"] = 0

    X = df.drop("hd", axis=1).copy()
    y = df["hd"].copy()

    X_encoded = pd.get_dummies(X, columns=["cp", "restecg", "slope", "thal"])
    X_encoded["ca"] = pd.to_numeric(X_encoded["ca"])

    return X_encoded, y


def plot_and_save_confusion(estimator, X_test, y_test, filename):
    ConfusionMatrixDisplay.from_estimator(
        estimator, X_test, y_test, display_labels=["Does not have HD", "Has HD"]
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf_xgb = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    clf_xgb.fit(X_train, y_train)
    plot_and_save_confusion(clf_xgb, X_test, y_test, "confusion_baseline.png")

    # tuned classifier
    clf_tuned = xgb.XGBClassifier(
        random_state=42,
        objective="binary:logistic",
        gamma=1,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        reg_lambda=10,
    )
    clf_tuned.fit(X_train, y_train)
    plot_and_save_confusion(clf_tuned, X_test, y_test, "confusion_tuned.png")

    # single-tree inspection
    clf_tree = xgb.XGBClassifier(
        random_state=42,
        objective="binary:logistic",
        gamma=1,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=1,
        reg_lambda=10,
    )
    clf_tree.fit(X_train, y_train)

    bst = clf_tree.get_booster()
    for importance_type in ("weight", "gain", "cover", "total_gain", "total_cover"):
        print(f"{importance_type}:", bst.get_score(importance_type=importance_type))

    try:
        graph = xgb.to_graphviz(clf_tree, num_trees=0, size="10,10")
        graph.render(filename="xgboost_tree", cleanup=True)
    except Exception as e:
        print("Could not render tree with graphviz:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
