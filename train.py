import argparse
import os
import pandas as pd
import joblib
import tempfile

from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def train(data_path: str, model_dir: str):
    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(data_path)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------
    numeric_features = X.columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ]
    )

    # -----------------------------
    # Model
    # -----------------------------
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # -----------------------------
    # Train
    # -----------------------------
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {auc}")

    # -----------------------------
    # Save model locally (OS-agnostic)
    # -----------------------------
    local_dir = tempfile.mkdtemp()
    local_model_path = os.path.join(local_dir, "diabetes_model.joblib")

    joblib.dump(pipeline, local_model_path)
    print(f"Local model saved at {local_model_path}")

    # -----------------------------
    # Upload to GCS (FAIL if not uploaded)
    # -----------------------------
    if not model_dir.startswith("gs://"):
        raise RuntimeError("model_dir must be a GCS path (gs://...) for Vertex AI")

    gcs_path = model_dir.replace("gs://", "")
    bucket_name, *prefix = gcs_path.split("/")
    blob_path = "/".join(prefix + ["diabetes_model.joblib"])

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(local_model_path)

    # HARD verification (no lies)
    if not blob.exists():
        raise RuntimeError("❌ MODEL UPLOAD FAILED: Blob does not exist in GCS")

    print(f"✅ CONFIRMED upload: gs://{bucket_name}/{blob_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input diabetes CSV file (GCS path)"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="GCS directory to save the trained model"
    )

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        model_dir=args.model_dir
    )
