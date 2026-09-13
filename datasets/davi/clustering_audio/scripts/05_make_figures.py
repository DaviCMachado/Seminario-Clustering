"""Gera espectrogramas comparativos e projeções PCA coloridas por cluster."""
from __future__ import annotations
import argparse
from pathlib import Path
import librosa, librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IDENTIFIERS = {"filepath", "relative_path", "dataset", "group_id", "duration_seconds"}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, help="WAV para comparar janelas")
    parser.add_argument("--features", type=Path, help="CSV de atributos")
    parser.add_argument("--clusters", type=Path, help="CSV com clusters")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.audio:
        y, sr = librosa.load(args.audio, sr=16000, mono=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
        for axis, window in zip(axes.ravel(), ["boxcar", "hann", "hamming", "blackman"]):
            spec = librosa.amplitude_to_db(abs(librosa.stft(y, n_fft=640, hop_length=320, window=window)), ref=max)
            librosa.display.specshow(spec, sr=sr, hop_length=320, x_axis="time", y_axis="hz", ax=axis)
            axis.set_title(window)
        fig.tight_layout(); fig.savefig(args.output_dir / "window_comparison.png", dpi=180); plt.close(fig)
    if args.features and args.clusters:
        frame, labels = pd.read_csv(args.features), pd.read_csv(args.clusters)
        merged = frame.merge(labels[["filepath", "cluster"]], on="filepath")
        numeric = [c for c in merged.select_dtypes("number").columns if c not in {"cluster", "duration_seconds"}]
        xy = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(merged[numeric].fillna(merged[numeric].median())))
        plt.figure(figsize=(8, 6)); scatter = plt.scatter(xy[:, 0], xy[:, 1], c=merged["cluster"], s=8, cmap="tab20")
        plt.colorbar(scatter, label="cluster"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
        plt.savefig(args.output_dir / "clusters_pca.png", dpi=180); plt.close()
if __name__ == "__main__": main()
