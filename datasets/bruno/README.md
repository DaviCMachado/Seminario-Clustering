# Descrição dos datasets utilizados
A pasta apresenta dois arquivos `.hdf5`, que são conjuntos de dados com *features* já extraidas. O conjunto de dados base é um arranjo planar de 900 respostas impulsivas, medidas em sala de controle vazia [1].

- `PlanarArray_Eer.hdf5`: Envelope de reflexões iniciais, replicando procedimento adotado por Zhang, Zhu e Shen [2].
- `PlanarArray_Eds.hdf5`: Envelope de reflexões iniciais e som direto. Adapta procedimento [2] realizado em `PlanarArray_Eer.hdf5` para incluir som direto da fonte.

## Referências
[1] X. Karakonstantis and E. Fernandez Grande, “Planar Room Impulse Response Dataset - ACT, DTU Electro (b. 355 r. 008).” Technical University of Denmark, 2024. doi: 10.11583/DTU.21740453.

[2] Zhang, Z., Zhu, G., & Shen, Y. (2018). Data clustering analysis of early reflections in small room. *The Journal of the Acoustical Society of America, 144*(4), EL328–EL332. [https://doi.org/10.1121/1.5065073](https://doi.org/10.1121/1.5065073)
