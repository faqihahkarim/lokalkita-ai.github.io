import numpy as np

path = r"C:\laragon\www\web\lokalkita\model\metadata_model\metadata_sbert_embeddings.npy"

emb = np.load(path)

print("Shape:", emb.shape)
print("First row:", emb[0][:10])

import numpy as np

npy_path = r"C:\laragon\www\web\lokalkita\model\metadata_model\metadata_sbert_embeddings.npy"
txt_path = r"C:\laragon\www\web\lokalkita\model\metadata_model\embeddings.txt"

emb = np.load(npy_path)

with open(txt_path, "w", encoding="utf-8") as f:
    for row in emb:
        f.write(str(row.tolist()) + "\n")

print("DONE! Saved to:", txt_path)
