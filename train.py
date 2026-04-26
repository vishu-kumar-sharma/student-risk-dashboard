"""
Train and compare student dropout-risk classifiers.

This script:
1. Loads the raw student lifestyle CSV.
2. Engineers a three-class Risk_Level target from CGPA and stress.
3. Builds preprocessing pipelines that handle missing values, encode categories,
   and scale numeric features with StandardScaler.
4. Trains Random Forest, SVM, and Logistic Regression models.
5. Evaluates each model with accuracy and weighted F1-score.
6. Exports all trained pipelines plus metadata for the Streamlit app.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# The app and training script both use this location by default. You can override
# it with STUDENT_DATA_PATH if you move the CSV elsewhere.
DEFAULT_DATA_PATH = Path(
    r"C:\Users\Aman Vyas\Desktop\student_lifestyle_performance_dataset.csv"
)
DATA_PATH = Path(os.getenv("STUDENT_DATA_PATH", DEFAULT_DATA_PATH))

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_student_lifestyle_performance_dataset.csv"
METADATA_JSON_PATH = MODEL_DIR / "training_metadata.json"
METADATA_JOBLIB_PATH = MODEL_DIR / "training_metadata.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Risk_Level"
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV and normalize column names lightly."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {csv_path}. "
            "Set STUDENT_DATA_PATH or update DEFAULT_DATA_PATH in train.py."
        )

    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]
    return df


def engineer_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a three-class student risk label.

    The rules intentionally combine academic performance and stress:
    - High Risk: very low CGPA, very high stress, or weak CGPA plus high stress.
    - Medium Risk: moderate CGPA/stress warning signs.
    - Low Risk: students not matching either warning pattern.
    """
    required_columns = {"CGPA", "Stress_Level_1_to_10"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    output = df.copy()
    cgpa = pd.to_numeric(output["CGPA"], errors="coerce")
    stress = pd.to_numeric(output["Stress_Level_1_to_10"], errors="coerce")

    high_risk = (cgpa < 5.5) | (stress >= 8.0) | ((cgpa < 6.5) & (stress >= 7.0))
    medium_risk = (cgpa < 7.0) | (stress >= 6.0) | ((cgpa < 7.5) & (stress >= 5.5))

    output[TARGET_COLUMN] = np.select(
        [high_risk, medium_risk],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    return output


def create_directory_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a cleaned copy for the dashboard directory.

    Training pipelines still handle missing values internally, but the directory
    view should not show raw blanks when a reasonable imputed value is available.
    """
    cleaned = df.copy()

    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = cleaned.select_dtypes(exclude=["number"]).columns.tolist()

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in categorical_columns:
        if column == TARGET_COLUMN:
            cleaned[column] = cleaned[column].fillna("Unknown Risk")
        else:
            mode = cleaned[column].mode(dropna=True)
            fallback = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[column] = cleaned[column].fillna(fallback)

    return cleaned


def split_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
    """Return feature matrix, target vector, and feature type groupings."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected engineered target column '{TARGET_COLUMN}'.")

    feature_df = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    feature_columns = feature_df.columns.tolist()

    return feature_df, target, feature_columns, numeric_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a version-compatible OneHotEncoder.

    scikit-learn renamed the sparse argument to sparse_output, so this helper
    keeps the script usable across common installed versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical inputs."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_models(
    numeric_features: List[str], categorical_features: List[str]
) -> Dict[str, Pipeline]:
    """Create the three requested model pipelines."""
    model_specs = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    pipelines: Dict[str, Pipeline] = {}
    for model_name, estimator in model_specs.items():
        pipelines[model_name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ]
        )

    return pipelines


def safe_artifact_name(model_name: str) -> str:
    """Convert a display model name into a stable file name."""
    return model_name.lower().replace(" ", "_") + ".joblib"


def train_and_evaluate() -> dict:
    """Train all models, export artifacts, and return training metadata."""
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    raw_df = load_dataset(DATA_PATH)
    labeled_df = engineer_risk_level(raw_df)
    cleaned_df = create_directory_dataset(labeled_df)
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    X, y, feature_columns, numeric_features, categorical_features = split_features_and_target(
        labeled_df
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_models(numeric_features, categorical_features)
    results = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        weighted_f1 = f1_score(y_test, predictions, average="weighted")
        report = classification_report(y_test, predictions, output_dict=True)

        artifact_path = MODEL_DIR / safe_artifact_name(model_name)
        joblib.dump(pipeline, artifact_path)

        results[model_name] = {
            "artifact_path": str(artifact_path),
            "accuracy": float(accuracy),
            "f1_score_weighted": float(weighted_f1),
            "classification_report": report,
            "is_best_model": False,
        }

    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    results[best_model_name]["is_best_model"] = True

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATA_PATH),
        "cleaned_dataset_path": str(CLEANED_DATA_PATH),
        "target_column": TARGET_COLUMN,
        "target_labels": RISK_LABELS,
        "target_engineering": {
            "High Risk": "CGPA < 5.5 OR Stress >= 8.0 OR (CGPA < 6.5 AND Stress >= 7.0)",
            "Medium Risk": "CGPA < 7.0 OR Stress >= 6.0 OR (CGPA < 7.5 AND Stress >= 5.5)",
            "Low Risk": "All remaining students",
        },
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "best_model_name": best_model_name,
        "models": results,
    }

    with METADATA_JSON_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    joblib.dump(metadata, METADATA_JOBLIB_PATH)
    return metadata


def print_summary(metadata: dict) -> None:
    """Print a compact console summary after training."""
    print("\nTraining complete.")
    print(f"Best model: {metadata['best_model_name']}")
    print("\nModel comparison:")

    for model_name, model_info in metadata["models"].items():
        recommended = " <- recommended" if model_info["is_best_model"] else ""
        print(
            f"- {model_name}: "
            f"accuracy={model_info['accuracy']:.4f}, "
            f"weighted_f1={model_info['f1_score_weighted']:.4f}"
            f"{recommended}"
        )

    print(f"\nArtifacts saved to: {MODEL_DIR}")
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    training_metadata = train_and_evaluate()
    print_summary(training_metadata)
