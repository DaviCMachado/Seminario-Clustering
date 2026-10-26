#%%
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt
from ykk import GMM


#%%
X, _ = make_blobs(n_samples=[200,200,200,50,50],
                  n_features=2, 
                  centers=[[-25,15],
                           [15,-15],
                           [-5,0],
                           [-25,15],
                           [15,-15],], cluster_std=[5,5,12,15,15], random_state=42)
plt.scatter(X[:,0],X[:,1])
#%%
gmm = GMM(n_components=2, max_iter=100, random_state=42,reg=1E-6)
gmm._initialize(X=X,)
#%%

frames = []
ll_old = -np.inf
for i in range(100):
    resp = gmm._expectation(X)
    gmm._maximization(X, resp)
    ll = gmm._log_l(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    
    frames.append({
        'X': X.copy(),
        'probs0': probs[:, 0].copy(),
        'means': gmm.means_.copy(),
        'll': ll
    })
    if abs(ll-ll_old) <= 1E-4:
        break
    

#%%
fig, ax = plt.subplots(figsize=(8, 6))
def update(frame_idx):
    ax.clear()
    f = frames[frame_idx]
    ax.scatter(f['X'][:, 0], f['X'][:, 1], c=f['probs0'],
               cmap='viridis', s=30, alpha=0.7)
    ax.scatter(f['means'][:, 0], f['means'][:, 1],
               c='red', marker='X', s=200, edgecolors='black')
    ax.set_title(f"Iteração {frame_idx + 1} — $\\log \\ell(\\theta)$ = {f['ll']:.3f}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

anim = FuncAnimation(fig, update, frames=len(frames), interval=(1/15)*1E3, repeat=True)
plt.close(fig)
HTML(anim.to_jshtml())

