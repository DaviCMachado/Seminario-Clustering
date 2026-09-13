# Arquitetura experimental

```mermaid
flowchart TD
    A[Datasets brutos WAV<br/>Al Emadi / Drone Audio / DREGON] --> B[Manifesto<br/>caminho + group_id]
    B --> C[Seleção de amostra<br/>sem usar rótulos]
    C --> D{Representação acústica}

    D --> E[STFT<br/>frame + hop + janela]
    E --> F[MFCC + descritores espectrais<br/>estatísticas temporais]

    D --> G[DWT<br/>wavelet + nível]
    G --> H[Energia relativa + entropia<br/>estatísticas dos coeficientes]

    F --> I[Matriz de atributos<br/>um vetor por áudio]
    H --> I
    I --> J[Padronização e PCA opcional]
    J --> K{Clustering}
    K --> L[K-Means]
    K --> M[GMM]
    K --> N[Hierárquico]
    K --> O[HDBSCAN]
    L --> P[Métricas internas]
    M --> P
    N --> P
    O --> P
    P --> Q[Silhouette<br/>Calinski-Harabasz<br/>Davies-Bouldin]
    Q --> R[Comparar configurações<br/>e selecionar candidatas]
    R --> S[Figuras e interpretação<br/>rótulos apenas após a seleção]
```

Cada caminho entre `D` e `I` é um experimento de representação. Rótulos
conhecidos não são usados para construir atributos, formar clusters ou escolher
a configuração; podem apenas contextualizar os resultados ao final.
