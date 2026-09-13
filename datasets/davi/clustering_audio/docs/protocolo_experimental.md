# Protocolo experimental

## Objetivo

Medir como a escala temporal (tamanho/salto de frame) e a função de janela
afetam a estrutura encontrada por métodos de clustering em áudio de drones.

## Ordem recomendada

1. Al Emadi: piloto pequeno e visual.
2. Drone_Audio_Dataset: experimento principal, inicialmente balanceado.
3. DREGON: análise controlada de ruído e segmentação.
4. Yi et al.: robustez entre ambiente, cenário e microfone.

## Desenho

- Variar `frame_length_ms`, `hop_length_ms` e `window` de acordo com
  `config/window_ablation.yaml`.
- Ajustar clustering somente aos vetores acústicos; não fornecer rótulos.
- Para K-Means/GMM, selecionar `k` pelas métricas internas. Para DBSCAN/HDBSCAN,
  registrar número de clusters e proporção de ruído.
- Repetir cada configuração com sementes/reamostragens e reportar média e
  dispersão das métricas internas.

## Controles contra dependência

Partes oriundas do mesmo vídeo, da mesma gravação longa ou de um par de
microfones devem compartilhar um `group_id`. Amostragens e análises de
estabilidade são feitas por grupo, nunca separando esses itens entre conjuntos.

## Uso interpretativo de rótulos

Rótulos disponíveis podem ser usados apenas depois do clustering para descrever
a composição dos grupos; eles não devem integrar a seleção de atributos, o
ajuste de modelos ou a escolha do número de clusters no experimento principal.
