import numpy as np
from scipy.spatial.distance import pdist

def xie_beni(X,centroids,categories):
    k_vec = np.arange(centroids.shape[0]) # cluster indexes
    N = len(categories) # Number of observations
    num = 0
    for j in k_vec:
        X_j = X[(categories==j), :] # Observations in category
        m_j = centroids[j,:]        # Centroid of category
        num += np.linalg.norm(X_j - m_j)**2

    den = np.min(pdist(centroids, metric='euclidean'))**2
    den *= N
    return num/den
