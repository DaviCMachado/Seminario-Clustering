# Scripts

| Script | Estado | Responsabilidade |
|---|---|---|
| `00_estimate_runtime.py` | implementado | mede amostra e prevê duração local |
| `01_build_manifest.py` | implementado | indexa os WAVs e registra grupos de origem |
| `02_extract_features.py` | implementado | atributos com STFT/MFCC e janelas |
| `03_cluster.py` | implementado | K-Means, GMM, hierárquico ou HDBSCAN |
| `04_evaluate_internal.py` | implementado | métricas internas |
| `05_make_figures.py` | implementado | figuras de sinal e clusters |
| `06_report_results.py` | implementado | ranking de métricas e interpretação pós-hoc |

Execute o manifesto a partir de `clustering_audio`:

```powershell
python scripts/01_build_manifest.py --dataset drone_audio --output data/manifests/drone_audio.csv
python scripts/00_estimate_runtime.py data/manifests/drone_audio.csv --samples 10
```

Exemplo completo (a execução integral do dataset pode ser demorada):

```powershell
python scripts/02_extract_features.py data/manifests/drone_audio.csv --output data/features/drone_audio.csv
python scripts/03_cluster.py data/features/drone_audio.csv
python scripts/04_evaluate_internal.py data/features/drone_audio.csv outputs/models
```

Passe `--config config/window_ablation.yaml` para expandir automaticamente as
36 combinações de janelas. Use `--limit 500` no piloto para limitar a amostra.

Passe `--config config/dwt_ablation.yaml` para executar as seis combinações DWT
(três wavelets e dois níveis).
