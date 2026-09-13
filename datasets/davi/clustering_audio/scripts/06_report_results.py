"""Consolida métricas e interpreta clusters após o ajuste não supervisionado."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics(metrics_dir: Path, output_dir: Path, pattern: str) -> None:
    files = list(metrics_dir.glob(pattern))
    tables = [pd.read_csv(path).assign(metrics_file=path.stem) for path in files]
    tables = [table for table in tables if {"run", "silhouette", "davies_bouldin"}.issubset(table.columns)]
    if not tables:
        raise ValueError("Nenhum CSV de métricas internas encontrado")
    summary = pd.concat(tables, ignore_index=True).sort_values("silhouette", ascending=False)
    summary.to_csv(output_dir / "metrics_ranking.csv", index=False)
    top = summary.head(min(20, len(summary))).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(top) * .32)))
    axes[0].barh(top["run"], top["silhouette"], color="#3874a8"); axes[0].set_title("Silhouette — maior é melhor")
    axes[1].barh(top["run"], top["davies_bouldin"], color="#b45f3b"); axes[1].set_title("Davies–Bouldin — menor é melhor")
    for axis in axes: axis.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(output_dir / "metrics_ranking.png", dpi=180); plt.close(fig)
    print(f"Ranking salvo em: {output_dir / 'metrics_ranking.csv'}")


def interpret_clusters(clusters: Path, manifest: Path, output_dir: Path) -> None:
    assigned, source = pd.read_csv(clusters), pd.read_csv(manifest)
    if "label_for_description" not in source:
        raise ValueError("O manifesto não tem label_for_description")
    merged = assigned.merge(source[["filepath", "label_for_description"]], on="filepath", how="left")
    composition = pd.crosstab(merged["cluster"], merged["label_for_description"], normalize="index")
    composition.to_csv(output_dir / "cluster_composition_posthoc.csv")
    plt.figure(figsize=(max(7, len(composition.columns) * .7), max(4, len(composition) * .5)))
    sns.heatmap(composition, cmap="Blues", annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.title("Composição pós-hoc dos clusters por rótulo conhecido")
    plt.xlabel("Rótulo usado apenas para interpretação"); plt.ylabel("Cluster")
    plt.tight_layout(); plt.savefig(output_dir / "cluster_composition_posthoc.png", dpi=180); plt.close()
    print(f"Composição pós-hoc salva em: {output_dir / 'cluster_composition_posthoc.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--pattern", default="*.csv", help="filtro de arquivos dentro de --metrics-dir")
    parser.add_argument("--clusters", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/report"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.metrics_dir: plot_metrics(args.metrics_dir, args.output_dir, args.pattern)
    if args.clusters or args.manifest:
        if not (args.clusters and args.manifest): parser.error("--clusters e --manifest devem ser usados juntos")
        interpret_clusters(args.clusters, args.manifest, args.output_dir)
    if not args.metrics_dir and not args.clusters: parser.error("Informe --metrics-dir ou --clusters/--manifest")


if __name__ == "__main__": main()
