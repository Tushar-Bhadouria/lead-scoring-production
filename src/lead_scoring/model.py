from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)

from xgboost import XGBClassifier


CLASS_WEIGHT = {
    0: 0.704,
    1: 1.724,
}


def build_decision_tree():
    return DecisionTreeClassifier(
        criterion="log_loss",
        max_depth=6,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        splitter="best",
        class_weight=CLASS_WEIGHT,
    )


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=3,
        min_weight_fraction_leaf=0.0,
        class_weight=CLASS_WEIGHT,
        random_state=42,
    )


def build_gradient_boost():
    return HistGradientBoostingClassifier(
        learning_rate=0.01,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=3,
        loss="log_loss",
        random_state=42,
    )


def build_xgboost(device="cpu"):
    return XGBClassifier(
        learning_rate=0.01,
        n_estimators=100,
        max_depth=9,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.45,
        tree_method="hist",
        device=device,
        random_state=42,
    )


def build_stacking_model():
    decision_tree = build_decision_tree()
    random_forest = build_random_forest()
    xgboost = build_xgboost(device="cpu")
    gradient_boost = build_gradient_boost()

    estimators = [
        ("dt", decision_tree),
        ("rf", random_forest),
        ("xgb", xgboost),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=gradient_boost,
        cv=5,
        n_jobs=-1,
    )