# GMM
Nota: Tem algumas coisas erradas aqui. Especialmente no passo de maximization
## Um quase abstract
O GMM, apesar de apresentado como uma técnica de clustering, pode ser entendido como uma tentativa de ajustar um RNG a um conjunto de dados. O RNG que queremos modelar é um conjunto de $K$ gaussianas e possui um seguinte comportamento:
- Ao chamar o RNG, é sorteado uma das $K$ gaussianas para gerar um dado
- A $k$-ésima gaussiana sorteia um dado.

Note que: embora o processo de sortear um dado apresente trivialidade(se já conhecido um conjunto de parâmetros que se ajuste bem ao seu conjunto de dados), estimar parâmetros apresenta dificuldade, visto que este admite múltiplas soluções.

Para estimar esses parâmetros, podemos fazer o seguinte:
- **Inverter o problema para algo mensurável:** este é feito por meio do Teorema de Bayes
- **Tentar maximizar a probabilidade do meu modelo gerar meu conjunto:** Conseguimos resolver isso iterativamente por meio do algoritmo *Expectation Maximization*.

O pertencimento a um grupo, nesse caso, é a probabilidade da $k$-ésima gaussiana ter gerado tal observação. 

## Modelando o "problema direto": O comportamento do RNG
Considere um conjunto de dados $\mathbf{X}\in \{\mathbf{x}_1,\dots,\mathbf{x}_N\}$ que consiste em observações de uma variável aleatória $\mathbf{x}$ com $D$ dimensões.

Quero criar um RNG que modele esses dados. Decido então aproximar o problema da seguinte forma:
Tenho $K$ gaussianas que são responsáveis por gerar um valor, cada um com um conjunto de parâmetros dado por $\theta_k = \{\mu_k,\Sigma_k\}$. a probabilidade da $k$-ésima gaussiana gerar $x_n$ é dada por
$$
p(x_n|\theta_k) = \mathcal{N}(x_n|\theta_k)
$$

Além disso, tenho uma variável aleatória categórica $z_n$ que sorteia uma das $K$ gaussianas e manda ela gerar um valor. Nesse caso, a probabilidade de $z_n$ sortear a $k$-ésima gaussiana é dada por:
$$
p(z_n = k) = \pi_k,
$$

Qualquer uma das gaussianas podem gerar $x_n$, e temos que levar em conta que nem todas as $K$ gaussianas tem chances iguais de serem escolhidas. Nesse caso, para modelarmos a probabilidade de $x_n$ ser sorteado, utilizamos a formula da probabilidade condicional
$$
\begin{align*}
p(x_n) &= \Sigma_{z_n} p(z_n)p(x_n|z_n)\\    
p(x_n) &= \Sigma_{k=1}^K \pi_k \mathcal{N}(x_n|\theta_k),
\end{align*}
$$
nesse contexto, chamamos $\pi_k$ de coeficientes da mistura, que quando somados dão 1.

> Basicamente, o meu RNG:
> - sorteia uma das $K$ gaussianas com $z_n$; e
> - manda essa gaussiana sortear um número, com probabilidade $\mathcal{N}(x_n|\theta_k)$

## Invertendo o problema: Ajustando meu RNG a um conjunto de dados
### O problema inverso
Como descrito, o GMM basicamente tenta ajustar um gerador de dados sintéticos a um conjunto de observações. Na seção anterior, vimos o processo de gerar os dados, no entanto, eu ainda não tenho uma boa estimativa do meu conjunto de parâmetros(vulgo, não sei $p(x_n)$). É possível ajustar o meu modelo a um conjunto de observações?

A formula que estabelece a relação entre o problema direto e o problema inverso é o Teorema de Bayes
$$
\begin{align*}
P(A|B) &= \frac{P(B|A)P(A)}{P(B)}\\
p(z_n=k|x_n) &= \frac{p(x_n|z_n=k)p(z_n=k)}{p(x_n)}\\
p(z_n=k|x_n) &= \frac{\mathcal{N}(x_n|\theta_k)\pi_k}{\Sigma_{k=1}^K \pi_k \mathcal{N}(x_n|\theta_k)},
\end{align*}
$$
em que $p(z_n=k|x_n)$ representa a probabilidade de $x_n$ ter sido gerado pelo $k$-ésimo cluster(gaussiana). Visto que $p(z_n=k|x_n)$ aparece recorrentemente na formulação, introduzimos o simbolo $\gamma_{nk} = p(z_n=k|x_n)$.

### Inferindo parâmetros para meu problema
Para encontrarmos uma solução sensata para meu problema, devemos levar em conta a probabilidade de meu modelo, dado pelo conjunto de parâmetros $\theta = \{\theta_1,\theta_2,\dots,\theta_K\}$, gerar todo o meu conjunto de dados $\mathbf{X}$, essa probabilidade pode ser escrita como
$$
\begin{align*}
p(\mathbf{X}|\theta) &= \prod_{n=1}^{N}\sum_{k=1}^{K}\pi_k p(x_n|\theta_k)\\
\ell(\theta)=p(\mathbf{X}|\theta) &= \prod_{n=1}^{N}\sum_{k=1}^{K}\pi_k\mathcal{N}(x_n|\theta_k).
\end{align*}
$$

Por conveniência matemática, observamos em forma logaritmica
$$
\log \ell(\theta)= \sum_{n=1}^{N}\log\sum_{k=1}^{K}\pi_k\mathcal{N}(x_n|\theta_k).
$$
> Por que não operar sobre $\ell(\theta)$
> - Derivada é mais complicada
> - Estabilidade numérica: valores muito pequenos podem resultar em underflow
> - No caso do GMM, ao aplicar $\log$, garantimos uma função monotônicamente crescente

Nós queremos maximizar a probabilidade do meu RNG gerar o meu conjunto de dados. O conjunto de parâmetros $\hat{\theta}$ que realiza esse feito é o resultado do seguinte problema de otimização
$$
\hat{\theta} = \arg \max_{\theta}\left\{\log \ell(\theta)\right\}.
$$

É importante notar que não é possível determinar um $\hat{\theta}$ global, no entanto, conseguimos maximizar $\ell(\theta)$ para um conjunto de parâmetros $\theta$ fixos. 
<!-- Nesse caso, o máximo ocorre ao encontrar a região em que $\ell(\theta)$ muda de sinal,  -->

>> ISSO ESTÀ ERRADO!!!

isto acontece quando
$$
\frac{\mathrm{d}\ell(\theta)}{\mathrm{d}\theta}  = 0,
$$
o algoritmo *Expectation Maximization* se aproveita dessa definição para aproximar $\hat{\theta}$.
>**Nota:** relendo o Aggarwal percebo que a questão da derivada so existe nas direções da média e covariância. O peso envolve outras coisas...
### Expectation Maximization
#### Passo 0: Inicialização
Dar um chute para $\theta_1^{(\tau=0)},\dots,\theta_K^{(\tau=0)}$

#### Passo 1: Expectation
Para cada observação $x_n$ e grupo $k$, avaliar 
$$
\gamma_{nk}^{(\tau)} = \frac{\mathcal{N}(x_n|\theta_k^{(\tau)})\pi_k^{(\tau)}}{\Sigma_{k=1}^K \pi_k^{(\tau)} \mathcal{N}(x_n|\theta_k^{(\tau)})}
$$
#### Passo 2: Maximization
Conforme comentado anteriormente, o máximo da função para um $\theta$ fixo ocorre quando
$$
\frac{\mathrm{d}\ell(\theta)}{\mathrm{d}\theta}  = 0,
$$
visto que há solução analítica para $\frac{\mathrm{d}\ell(\theta)}{\mathrm{d}\theta}$ (isto é, a solução é uma equação conhecida). Conseguimos maximizar localmente.
> Ver as equações do slide, não quero redigitar

#### Passo 3: Repetir
Visto que estimamos um novo conjunto $\theta^{(\tau +1)}$, podemos repetir os passos 2 e 3 usando $\theta^{(\tau +1)}$ como dado de entrada, maximizando $\ell(\theta)$ mais ainda. Dessa forma, com o passar das iterações, o modelo começa a se ajustar cada vez melhor ao conjunto de dados.

<!-- obter um novo conjunto de parâmetros $\theta^{(\tau+1)}$ -->
