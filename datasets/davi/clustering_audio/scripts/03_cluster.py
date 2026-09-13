"""Executa clustering apenas sobre as colunas acústicas."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from src.configuration import load_config
from src.paths import OUTPUTS_DIR
IDENTIFIERS = {"filepath", "relative_path", "dataset", "group_id", "duration_seconds"}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--config", type=Path, default=PIPELINE_ROOT / "config" / "baseline.yaml")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "models")
    args = parser.parse_args()
    config, frame = load_config(args.config), pd.read_csv(args.features)
    columns = [c for c in frame.select_dtypes("number").columns if c not in IDENTIFIERS]
    matrix = StandardScaler().fit_transform(frame[columns].fillna(frame[columns].median()).to_numpy())
    if config["clustering"].get("pca_components"):
        matrix = PCA(n_components=config["clustering"]["pca_components"], random_state=config["seed"]).fit_transform(matrix)
    method = config["clustering"]["method"].lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for k in config["clustering"]["k_values"]:
        if not 2 <= int(k) < len(frame): continue
        if method == "kmeans": model = KMeans(n_clusters=k, random_state=config["seed"], n_init="auto")
        elif method == "gmm": model = GaussianMixture(n_components=k, random_state=config["seed"])
        elif method in {"hierarchical", "agglomerative"}: model = AgglomerativeClustering(n_clusters=k)
        elif method == "hdbscan":
            try:
                import hdbscan
            except ImportError as exc:
                raise ImportError("Instale hdbscan (requirements.txt) para usar este método") from exc
            model = hdbscan.HDBSCAN(min_cluster_size=k)
        else: raise ValueError("Método suportado: kmeans, gmm, hierarchical ou hdbscan")
        result = frame[[c for c in IDENTIFIERS if c in frame]].copy()
        result["cluster"] = model.fit_predict(matrix)
        output = args.output_dir / f"clusters_{method}_k{k}.csv"
        result.to_csv(output, index=False)
        output.with_suffix(".json").write_text(json.dumps({"method": method, "k": k, "features": columns, "config": config}, indent=2), encoding="utf-8")
        print(f"Clusters salvos: {output}")

if __name__ == "__main__": main()
