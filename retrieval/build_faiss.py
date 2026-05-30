import faiss
import numpy as np

embeddings = np.load(
    "data/complaint_embeddings.npy"
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    "retrieval/faiss.index"
)

print("FAISS index created")