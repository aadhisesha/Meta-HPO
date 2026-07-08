from pyexpat import features

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

from train_meta import (
    train_meta_model,
    sample_config,
    config_to_vector,
    evaluate,
    get_meta_features
)



def preprocess_dataframe(df):
    """
    Automatically preprocess any uploaded dataset.

    - One-hot encodes categorical features
    - Encodes target labels if needed
    - Fills missing values
    - Returns float32 features
    """

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    preprocessing_info = []

    # ----------------------------
    # Find categorical columns
    # ----------------------------
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if len(categorical_cols) > 0:

        preprocessing_info.append(
            f"One-Hot Encoded: {', '.join(categorical_cols)}"
        )

        X = pd.get_dummies(
            X,
            columns=categorical_cols,
            drop_first=False
        )

    # ----------------------------
    # Convert everything to numeric
    # ----------------------------
    X = X.apply(pd.to_numeric, errors="coerce")

    # ----------------------------
    # Fill missing values
    # ----------------------------
    imputer = SimpleImputer(strategy="mean")

    X = imputer.fit_transform(X)

    X = np.asarray(X, dtype=np.float32)

    # ----------------------------
    # Encode target if categorical
    # ----------------------------
    if not pd.api.types.is_numeric_dtype(y):

        le = LabelEncoder()

        y = le.fit_transform(y.astype(str))

        preprocessing_info.append(
            "Target labels encoded"
        )

    y = np.asarray(y)

    return X, y, preprocessing_info


def rank_candidates(meta_model, meta_features, n_candidates=100):

    candidates = []

    for _ in range(n_candidates):

        hp = sample_config()

        hp_vec = config_to_vector(hp)

        features = np.concatenate([meta_features, hp_vec])
        print("Feature vector length:", len(features))
        print(features)

        inp = torch.tensor(
            features,
            dtype=torch.float32
        )

        with torch.no_grad():

            predicted = meta_model(inp).item()

        candidates.append((predicted, hp))

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates

def run_hpo(meta_model, X, y):

    meta = get_meta_features(X, y)

    ranked = rank_candidates(
        meta_model,
        meta,
        n_candidates=100
    )

    best_score = -1

    best_hp = None

    scores = []

    best_curve = []

    rows = []

    # Evaluate only top 10

    for i, (predicted, hp) in enumerate(ranked[:10]):

        actual = evaluate(X, y, hp)

        scores.append(actual)

        if actual > best_score:

            best_score = actual

            best_hp = hp

        best_curve.append(best_score)

        rows.append({

            "Trial": i + 1,

            "Model": hp["type"],

            "Predicted": round(predicted, 3),

            "Actual": round(actual, 3)

        })

        st.write(

            f"Trial {i+1}"

            f" | {hp['type']}"

            f" | Pred={predicted:.3f}"

            f" | Actual={actual:.3f}"

        )

    return (

        best_hp,

        best_score,

        scores,

        best_curve,

        pd.DataFrame(rows)

    )


st.title("Meta-Learning Hyperparameter Optimization")

file = st.file_uploader("Upload CSV (last column = target)")

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())


    X, y, preprocessing_info = preprocess_dataframe(df)

    st.subheader("Preprocessing Summary")

    for item in preprocessing_info:
        st.write("✅", item)

    st.write("Dataset Shape:", df.shape)
    st.write("Processed Feature Shape:", X.shape)
    st.write("Target Shape:", y.shape)
    st.write("X dtype:", X.dtype)
    st.write("y dtype:", y.dtype)

    # -----------------------------
    # Run Meta-HPO
    # -----------------------------
    
    if st.button("Run Meta-HPO"):

        import os

        st.write("Loading trained surrogate...")

        if not os.path.exists("surrogate.pt"):
            st.error("surrogate.pt not found. Run train_meta.py first.")
            st.stop()

        meta_model = nn.Sequential(
        nn.Linear(18, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
        )

        meta_model.load_state_dict(torch.load("surrogate.pt"))
        meta_model.eval()

        st.write("Running optimization...")

        best_hp, best_score, scores, best_curve, trial_df = run_hpo(
        meta_model,
        X,
        y
        )

        st.success("Done!")

        st.subheader("Best Configuration")
        st.write(best_hp)

        st.subheader("Best Accuracy")
        st.write(f"{best_score:.4f}")

        st.subheader("Trial Results")
        st.dataframe(trial_df)

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(
            trial_df["Trial"],
            trial_df["Actual"],
            marker="o",
            linewidth=2,
            label="Actual Accuracy"
        )

        best_so_far = trial_df["Actual"].cummax()

        ax.plot(
            trial_df["Trial"],
            best_so_far,
            linewidth=3,
            label="Best So Far"
        )

        ax.set_xlabel("Trial")
        ax.set_ylabel("Accuracy")
        ax.set_title("Meta-Learning Guided Search")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        st.subheader("Prediction vs Actual")

        comparison = trial_df.copy()

        comparison["Difference"] = (
            comparison["Predicted"] -
            comparison["Actual"]
        ).abs()

        st.dataframe(comparison)