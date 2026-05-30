import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv(r"C:\Users\chitt\complaint-auto-routing-system\data\complaints.csv")

print(f"Dataset Shape: {df.shape}")

print("Loading embedding model...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Generating embeddings...")

embeddings = model.encode(
    df["complaint_text"].tolist(),
    show_progress_bar=True
)

print("Embedding Shape:", embeddings.shape)

np.save(
    "data/complaint_embeddings.npy",
    embeddings
)

print("Embeddings saved successfully")