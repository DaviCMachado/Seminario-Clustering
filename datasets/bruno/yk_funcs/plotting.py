import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import locale
import warnings
import numpy as np
import matplotlib.pyplot as plt


def plot_2d_plane():
    pass

def plot_wavetable(signals,categories,indexes,fs=44100,ax=None):
    mics = np.arange(signals.shape[0])
    ax = _parse_ax(ax, projection='3d')
    for n in indexes:
        z = np.ones(len(mics))*n*1E3/fs
        ax.scatter(mics,z,signals[:,n],c=categories,s=5)

def _parse_ax(ax,**kwargs):
    if ax is None:
        return plt.axes(**kwargs)
    else:
        return ax



def enable_latex():
    """Muda o sistema global de fontes do matplotlib para
    LaTeX + Times New Roman
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
def set_ptbr():
    locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
    plt.rcParams['axes.formatter.use_locale'] = True
