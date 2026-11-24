import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_text(text):
    # Cleaned project text contains only alphabetic words and spaces.
    return clean_text(text).split()

def document_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

def main(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place ecommerceDataset.csv in the project folder "
            "or pass its path with --data."
        )

    df = pd.read_csv(data_path, header=None, names=["category", "description"])
    df = df.dropna(subset=["description", "category"]).drop_duplicates().reset_index(drop=True)
    df["clean_text"] = df["description"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize_text)

    print("Dataset rows after cleaning:", len(df))
    print("Categories:", sorted(df["category"].astype(str).unique()))

    # Train Word2Vec using the same key settings as the supplied notebook.
    embedding_model = Word2Vec(
        sentences=df["tokens"].tolist(),
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=10,
    )

    X = np.array([document_vector(tokens, embedding_model) for tokens in df["tokens"]])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["category"].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    ann_model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.30),
        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(len(encoder.classes_), activation="softmax"),
    ])
    ann_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ann_model.fit(
        X_train, y_train, validation_split=0.20, epochs=10, batch_size=32, verbose=1
    )
    loss, accuracy = ann_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    ann_model.save(output_dir / "ann_model.keras")
    embedding_model.save(str(output_dir / "word2vec.model"))
    joblib.dump(encoder, output_dir / "label_encoder.joblib")
    print(f"Saved model artifacts to: {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ecommerceDataset.csv", help="Path to the CSV dataset")
    args = parser.parse_args()
    main(args.data)
