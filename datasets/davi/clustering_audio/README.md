# Clustering acústico de drones

Estrutura de experimentos não supervisionados para investigar como segmentação
temporal e funções de janela afetam agrupamentos de áudio de drones.

Os áudios originais permanecem nas pastas vizinhas. Os scripts devem gerar
manifestos em `data/manifests`, atributos em `data/features` e resultados em
`outputs`; nenhum deles altera os datasets brutos.

## Fluxo experimental

1. `01_build_manifest.py`: indexar áudios e seus grupos de origem.
2. `02_extract_features.py`: aplicar a configuração temporal e criar vetores
   acústicos por arquivo.
3. `03_cluster.py`: executar um método de clustering e salvar atribuições.
4. `04_evaluate_internal.py`: calcular somente métricas internas e estabilidade.
5. `05_make_figures.py`: produzir espectrogramas, projeções e gráficos.

As configurações ficam em `config/`. Cada execução deve registrar o arquivo de
configuração usado, a semente e a versão do manifesto nos resultados.

## Convenções

- Rótulos conhecidos não entram em `data/features` nem no ajuste do clustering.
- A coluna `group_id` identifica a gravação/vídeo de origem e serve para evitar
  que segmentos correlacionados sejam tratados como amostras independentes.
- `outputs/` e os artefatos grandes de `data/` são ignorados pelo Git.

Consulte `docs/protocolo_experimental.md` antes de executar os experimentos.
O diagrama completo está em `docs/arquitetura_experimental.md`.
