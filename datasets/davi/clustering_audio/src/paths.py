"""Caminhos centralizados para que scripts nunca alterem os áudios brutos."""
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DAVI_ROOT = PIPELINE_ROOT.parent
RAW_DATASETS = {
    "al_emadi": DAVI_ROOT / "Al Emadi (2019)",
    "dregon": DAVI_ROOT / "dregon dataset",
    "drone_audio": DAVI_ROOT / "Drone_Audio_Dataset",
    "yi_2023": DAVI_ROOT / "Yi et al. (2023)",
}
MANIFESTS_DIR = PIPELINE_ROOT / "data" / "manifests"
FEATURES_DIR = PIPELINE_ROOT / "data" / "features"
OUTPUTS_DIR = PIPELINE_ROOT / "outputs"
