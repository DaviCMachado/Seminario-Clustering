"""Calcula métricas internas sem consultar rótulos conhecidos."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

IDENTIFIERS = {"filepath", "relative_path", "dataset", "group_id", "duration_seconds"}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("clusters_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/internal_metrics.csv"))
    args = parser.parse_args()
    features = pd.read_csv(args.features)
    numeric = [c for c in features.select_dtypes("number").columns if c not in IDENTIFIERS]
    matrix = StandardScaler().fit_transform(features[numeric].fillna(features[numeric].median()))
    rows = []
    for path in sorted(args.clusters_dir.glob("clusters_*.csv")):
        assignments = pd.read_csv(path)
        merged = features[["filepath"]].merge(assignments[["filepath", "cluster"]], on="filepath", how="inner")
        labels = merged["cluster"].to_numpy()
        if not 1 < len(set(labels)) < len(labels): continue
        positions = features.index[features["filepath"].isin(merged["filepath"])].to_numpy()
        rows.append({"run": path.stem, "samples": len(labels), "clusters": len(set(labels)),
                     "silhouette": silhouette_score(matrix[positions], labels),
                     "calinski_harabasz": calinski_harabasz_score(matrix[positions], labels),
                     "davies_bouldin": davies_bouldin_score(matrix[positions], labels)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Métricas internas salvas em: {args.output}")
if __name__ == "__main__": main()
