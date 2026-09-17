"""Extrai atributos por frame e os agrega em um vetor por áudio."""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from src.audio_features import extract_features
from src.configuration import load_config
from src.paths import FEATURES_DIR


def process_record(record, run: dict) -> tuple[dict | None, dict | None]:
    """Extrai uma linha; falhas individuais não interrompem o experimento."""
    try:
        values = extract_features(record.filepath, run["audio"], run["features"], run.get("representation"))
        return {"filepath": record.filepath, "relative_path": record.relative_path,
                "dataset": record.dataset, "group_id": record.group_id, **values}, None
    except Exception as exc:
        return None, {"filepath": record.filepath, "error": str(exc)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=PIPELINE_ROOT / "config" / "baseline.yaml")
    parser.add_argument("--output", type=Path, default=FEATURES_DIR / "features.csv")
    parser.add_argument("--limit", type=int, help="limita arquivos; útil no piloto e na ablação")
    parser.add_argument("--workers", type=int, default=1, help="threads para extração; use 4–8 em datasets maiores")
    args = parser.parse_args()
    config, manifest = load_config(args.config), pd.read_csv(args.manifest)
    limit = args.limit or config.get("dataset", {}).get("max_samples")
    if limit:
        manifest = manifest.sample(n=min(int(limit), len(manifest)), random_state=config["seed"])
    grid = config.get("grid")
    dwt_grid = config.get("dwt_grid")
    combinations = [config]
    if grid:
        combinations = []
        for frame, hop, window in product(grid["frame_length_ms"], grid["hop_length_ms"], grid["window"]):
            run = deepcopy(config); run["audio"].update({"frame_length_ms": frame, "hop_length_ms": hop, "window": window})
            run["experiment_name"] = f"{config['experiment_name']}__frame{frame}ms_hop{hop}ms_{window}"
            combinations.append(run)
    elif dwt_grid:
        combinations = []
        for wavelet, level in product(dwt_grid["wavelet"], dwt_grid["level"]):
            run = deepcopy(config); run["representation"].update({"wavelet": wavelet, "level": level})
            run["experiment_name"] = f"{config['experiment_name']}__{wavelet}_L{level}"
            combinations.append(run)
    for run in combinations:
        output = args.output if len(combinations) == 1 else args.output.with_name(f"{args.output.stem}__{run['experiment_name'].split('__')[-1]}{args.output.suffix}")
        rows, errors = [], []
        print(f"Configuração: {run['experiment_name']}")
        records = list(manifest.itertuples(index=False))
        if args.workers == 1:
            results = map(lambda record: process_record(record, run), records)
        else:
            executor = ThreadPoolExecutor(max_workers=args.workers)
            results = executor.map(lambda record: process_record(record, run), records)
        for number, (row, error) in enumerate(results, start=1):
            if row is not None: rows.append(row)
            if error is not None: errors.append(error)
            if number % 100 == 0 or number == len(manifest): print(f"Processados {number}/{len(manifest)}; sucesso: {len(rows)}; falhas: {len(errors)}")
        if args.workers != 1:
            executor.shutdown(wait=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        output.with_suffix(".metadata.json").write_text(json.dumps({"config": run, "input_manifest": str(args.manifest), "rows": len(rows), "errors": len(errors)}, indent=2), encoding="utf-8")
        if errors: pd.DataFrame(errors).to_csv(output.with_suffix(".errors.csv"), index=False)
        print(f"Atributos salvos em: {output}")

if __name__ == "__main__": main()
