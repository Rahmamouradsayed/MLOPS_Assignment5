import os
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
    n_redundant=3, n_classes=3, n_clusters_per_class=1, flip_y=0.02, random_state=42)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
df["target"] = y
os.makedirs("data", exist_ok=True)
df.to_csv("data/dataset.csv", index=False)
print(f"Dataset created: {len(df)} rows")
