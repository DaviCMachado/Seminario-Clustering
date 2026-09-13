"""Mede uma amostra e projeta duração da extração para um manifesto inteiro.

O resultado é uma estimativa local: inclui disco, CPU e versões das bibliotecas
da máquina que de fato executará o experimento.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from src.audio_features import extract_features
from src.configuration import load_config

def format_seconds(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=PIPELINE_ROOT / "config" / "baseline.yaml")
    parser.add_argument("--samples", type=int, default=10, help="arquivos para medir após o aquecimento")
    args = parser.parse_args()
    config, manifest = load_config(args.config), pd.read_csv(args.manifest)
    sample = manifest.sample(n=min(args.samples + 1, len(manifest)), random_state=config["seed"])
    warmup, measured = sample.iloc[0], sample.iloc[1:]
    print(f"Aquecendo bibliotecas com: {warmup.relative_path}")
    start = time.perf_counter()
    extract_features(warmup.filepath, config["audio"], config["features"], config.get("representation"))
    warmup_seconds = time.perf_counter() - start
    durations = []
    for item in measured.itertuples(index=False):
        start = time.perf_counter()
        extract_features(item.filepath, config["audio"], config["features"], config.get("representation"))
        durations.append(time.perf_counter() - start)
    average = sum(durations) / len(durations)
    estimated = average * len(manifest)
    print(f"Aquecimento: {format_seconds(warmup_seconds)}")
    print(f"Média recorrente: {average:.3f} s/arquivo ({len(durations)} amostras)")
    print(f"Extração estimada: {format_seconds(estimated)} para {len(manifest):,} arquivos")
    grid = config.get("grid") or config.get("dwt_grid")
    if grid:
        if "frame_length_ms" in grid:
            combinations = len(grid["frame_length_ms"]) * len(grid["hop_length_ms"]) * len(grid["window"])
        else:
            combinations = len(grid["wavelet"]) * len(grid["level"])
        print(f"Grade de janelas: {combinations} execuções → {format_seconds(estimated * combinations)} serialmente")
    print("Clustering e métricas tendem a acrescentar poucos minutos para amostras até dezenas de milhares de vetores.")

if __name__ == "__main__": main()
