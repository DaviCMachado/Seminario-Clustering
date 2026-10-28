import numpy as np
from sklearn import cluster

_float_eps = np.finfo(float).eps*20

class GMM:
    def __init__(self, K=3, max_iter=100, tol=1e-4, reg=1E-6, random_state=None):
        """Classe para análise de clusters

        Args:
            n_components (int, optional): Número de clusters do modelo. Defaults to 3.
            max_iter (int, optional): Número máximo de iterações do GMM. Defaults to 100.
            tol (float, optional): Limiar de convergência. Defaults to 1e-4.
            reg (float, optional): Parâmetro de regularização da matriz, adicionado à
                                    diagonal principal da matriz de covariância, 
                                    evitando 'overfitting'. Defaults to 1E-6.
            random_state (int, optional): seed. Defaults to None.
        """
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reg = reg


    def _initialize(self, X,):
        """Utiliza k-means para chutar um conjunto de parâmetros iniciais
        para o modelo. Conforme discutido em apresentação, os centróides
        se transformam em médias, os pesos são o tamanho relativo de cada
        conjunto e a covariância é a matriz de covariância de cada cluster.

        Note que: apenas implementei a matriz de covariância sem imposição
        de formato (`cov_type='full'`).

        Args:
            X (ndarray): Dados de entrada
        """
        n_samples, n_dims = X.shape
        km_mu,km_lbl,_=cluster.k_means(X = X,
                                        n_clusters = self.K,
                                        n_init=1,
                                        init='k-means++',
                                        random_state=self.random_state
                                        )
        self.means_ = km_mu #a média é o centróide

        self.pi_ = np.zeros([self.K, 1])
        self.covariances_ = np.zeros([self.K, n_dims, n_dims])
        for k in range(self.K):
            # Os pesos são o quão representativo cada cluster
            # é para meu conjunto
            self.pi_[k] = km_lbl[ km_lbl == k ].shape[0]/n_samples
            # X^T @ X + alpha @ I
            self.covariances_[k,:] = np.cov(X[km_lbl == k].T)
            self.covariances_[k,:] +=  self.reg * np.eye(n_dims)


    def _gaussian(self, X, mean, cov):
        """Avalia a densidade de probabilidade de uma gaussiana 
        multivariada de média `mean` e covariância `cov` nos 
        pontos `X`.

        Args:
            X (ndarray): Os pontos onde avaliar (observações x dimensões)
            mean (ndarray): valor médio (n_dimensões,)
            cov (_type_): Matriz de covariância (n_dimensões,n_dimensões)

        Returns:
            ndarray: Densidade de probabilidade avaliada nos pontos. (n_observações,) 
        """
        n_dims = X.shape[1]
        diff = X - mean
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        norm = 1.0 / np.sqrt(((2 * np.pi) ** n_dims) * det)
        exp = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return norm * exp


    def _expectation(self, X):
        """O E Step avalia a responsabilidade do
        meu conjunto de gaussianas pelo meu conjunto
        de observações.

        Args:
            X (ndarray): Conjunto de observações

        Returns:
            ndarray: gamma_nk (n_signals, n_dim)
        """
        n_samples = X.shape[0]
        # responsabilidade (gamma_nk)
        gamma = np.zeros((n_samples, self.K))

        for k in range(self.K):
            gamma[:, k] = self.pi_[k] * self._gaussian(
                X, self.means_[k], self.covariances_[k]
            )

        # Normaliza (responsabilidades)
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1e-10
        gamma /= gamma_sum
        return gamma


    def _maximization(self, X, gamma):
        """Após calcular a responsabilidade do meu modelo
        sobre cada observação. Utilizamos solução analítica
        para maximizar l(theta), com theta fixo.

        Args:
            X (ndarray): Conjunto de observações
            resp (ndarray): Responsabilidade de cada cluster
        """
        n_samples, n_dims = X.shape
        gamma_k = gamma.sum(axis=0) + 1e-10

        # Atualiza pesos
        self.pi_ = gamma_k / n_samples

        # Atualiza médias
        self.means_ = (gamma.T @ X) / gamma_k[:, np.newaxis]

        # Atualiza covariâncias
        for k in range(self.K):
            deviation = X - self.means_[k] # Desvio (da média)
            self.covariances_[k] = ((gamma[:, k] * deviation.T) @ deviation) / gamma_k[k]
            self.covariances_[k] += self.reg * np.eye(n_dims)



    def _log_l(self, X):
        """Avalia a probabilidade do modelo de ter gerado
        todo o meu cnjunto de dados

        Args:
            X (ndarray): o dataset

        Returns:
            float: o "log-likelihood" para meu conjunto theta
        """
        n_samples = X.shape[0]
        prob = np.zeros(n_samples)
        for k in range(self.K):
            prob += self.pi_[k] * self._gaussian(
                X, self.means_[k], self.covariances_[k]
            )
        # Para evitar ln(0) -> -infty
        return np.sum(np.log(prob + _float_eps))


    def fit(self, X):
        """Repete expectation e maximization até 
        a diferença entre log-likelihoods entre
        iterações ser inferior a tol

        Args:
            X (ndarray): o conjunto de dados

        Returns:
            self: o proprio objeto
        """
        X = np.asarray(X)
        self._initialize(X)
        prev_ll = -np.inf

        for i in range(self.max_iter):
            resp = self._expectation(X)
            self._maximization(X, resp)
            ll = self._log_l(X)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        print(f"Convergiu {i+1}")
        self.n_iter_ = i + 1
        self.lower_bound_ = ll
        return self


    def predict_proba(self, X):
        """Avalia gamma_nk: retorna o
        pertencimento à cada cluster

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self._expectation(np.asarray(X))

    def predict(self, X):
        """Avalia gamma_nk e retorna o cluster mais 
        provável de ter gerado um determinado dado. 
        O modelo se torna, então, um problema de
        hard clustering.

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.argmax(self.predict_proba(X), axis=1)

