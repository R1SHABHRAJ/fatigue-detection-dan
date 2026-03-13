import shap
import lime
import lime.lime_tabular
import numpy as np


# ============================================================
# SHAP EXPLANATION
# ============================================================

def run_shap(model, X_train, feature_names):

    """
    Generate SHAP explanations for trained model
    """

    explainer = shap.Explainer(model.predict, X_train)

    shap_values = explainer(X_train)

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_names
    )

    return shap_values


# ============================================================
# LIME EXPLANATION
# ============================================================

def run_lime(model, X_train, feature_names):

    """
    Generate LIME explanation for one sample
    """

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="classification"
    )

    explanation = explainer.explain_instance(
        X_train[0],
        model.predict_proba,
        num_features=10
    )

    explanation.show_in_notebook()

    return explanation