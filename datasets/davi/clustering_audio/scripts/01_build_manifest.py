"""Cria um inventário de arquivos de áudio sem modificar os datasets brutos.

O manifesto é a fronteira entre dados brutos e experimentos: caminhos e grupos
de origem ficam explícitos antes de qualquer extração ou clustering.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.paths import MANIFESTS_DIR, RAW_DATASETS


def infer_record(path: Path, dataset: str) -> dict[str, str]:
    relative = path.relative_to(RAW_DATASETS[dataset])
    parts = relative.parts
    record = {"dataset": dataset, "relative_path": str(relative), "label_for_description": "", "group_id": path.stem}

    if dataset == "al_emadi":
        record["label_for_description"] = parts[1] if len(parts) > 1 else ""
        record["group_id"] = path.stem.rsplit("_", 1)[0]
    elif dataset == "drone_audio":
        record["label_for_description"] = parts[1] if len(parts) > 1 else ""
        record["group_id"] = path.stem.rsplit("_", 1)[0]
    elif dataset == "dregon":
        record["label_for_description"] = parts[0]
        record["group_id"] = path.stem.split("_seg", 1)[0]
    elif dataset == "yi_2023":
        record["label_for_description"] = parts[0]
        tokens = path.stem.split("_")
        record["group_id"] = "_".join(tokens[:4]) if len(tokens) >= 4 else path.stem
    return record


def parse_label_counts(value: str) -> dict[str, int]:
    """Converte `label=quantidade,...` em limites de amostragem reproduzíveis."""
    try:
        return {part.split("=", 1)[0]: int(part.split("=", 1)[1]) for part in value.split(",")}
    except (IndexError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use, por exemplo: bebop_1=250,membo_1=250,unknown=500") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*RAW_DATASETS, "all"], default="all")
    parser.add_argument("--output", type=Path, default=MANIFESTS_DIR / "manifest.csv")
    parser.add_argument("--include-processed", action="store_true", help="inclui segmentos derivados, apenas para DREGON")
    parser.add_argument("--per-label", type=parse_label_counts, help="amostra estratificada: label=quantidade,...")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    names = RAW_DATASETS if args.dataset == "all" else {args.dataset: RAW_DATASETS[args.dataset]}
    rows = []
    for name, root in names.items():
        for wav in root.rglob("*.wav"):
            relative = wav.relative_to(root)
            # Evita duplicação do Al Emadi (pastas binary/multiclass contêm os
            # mesmos exemplos) e a fila de revisão do Drone Audio, que não é
            # parte do corpus anotado principal.
            if name == "al_emadi" and relative.parts[0] != "Multiclass_Drone_Audio":
                continue
            if name == "drone_audio" and relative.parts[0] != "audio":
                continue
            # Segmentos derivados são criados sob demanda; não entram no
            # inventário padrão para não duplicar a evidência de uma gravação.
            if "processed" not in wav.parts or args.include_processed:
                rows.append({"filepath": str(wav.resolve()), **infer_record(wav, name)})
    frame = pd.DataFrame(rows)
    if args.per_label:
        selected = []
        for label, count in args.per_label.items():
            candidates = frame[frame["label_for_description"] == label]
            if len(candidates) < count:
                raise ValueError(f"Rótulo {label!r} tem {len(candidates)} itens; solicitados: {count}")
            selected.append(candidates.sample(n=count, random_state=args.seed))
        frame = pd.concat(selected, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Manifesto criado: {args.output} ({len(frame)} arquivos)")


if __name__ == "__main__":
    main()
