"""Um pedaço de ykk_utils.dsp_funcs
"""
import numpy as np

def envelope(signal, axis=-1):
    """Determines envelope of an signal using Hilbert Transform.
    Used for analysis of room reflections, in this context, the
    envelope of a signal is called 'reflectogram' or 'echogram'.

    References:
        [1] Kuttruff. Room Acoustics (4nd Ed.). Section 8.4

    Args:
        signal (NDArray): Array of signals to calculate
        axis (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    signal = np.atleast_2d(signal)
    signal_jw = np.fft.rfft(signal, axis=axis)
    signal_hilbert = np.fft.irfft(-1j*signal_jw, axis=axis)
    return np.sqrt(signal**2 + signal_hilbert**2).squeeze()

def modified_kurtosis(signal, block_sizes=[4,64],axis=-1):
    """
    Calculates time dependent Kurtosis, used for onset time estimation 
    on impulse responses [1]. Kurtosis is a metric for estimation of the
    "degree of normality" of a distribution, where `k = 3` denotes a normal
    distribution, therefore, at reflection onset instant `k` is expected 
    to be greater than 3.

    Validity:
        It is expected reliable results for mode densities inferior to 
        1 mode/ms, which can be estimated from:

        `mode_density = 4*pi * (c0**3) * (t**2) / V`,
        
        where `c0` represents the sound speed in air, `V` is the room volume, and
        `t` is the time instant after onset (`t-t0`) [1].


    The treatment of discontinuity at edges are done by wrapping the signal, this behaviour is
    based on `ita_kurtosis()` from ITA-Toolbox [2].
    

    References:
        [1]: Usher. An improved method to determine the onset timings of 
        reflections in an acoustic impulse response. J. Acoust. Soc. Am. 1 April 2010; 127 (4). 
        https://doi.org/10.1121/1.3361042
        [2]: Berzborn, et al. The ITA-Toolbox: An Open Source MATLAB Toolbox for Acoustic 
        Measurements and Signal Processing. 


    Args:
        signal (NDArray): Impulse responses you want to analyse.
        block_size (NDArray): Analysis window length in samples, must be an array
        of two elements(smaller and larger window). Defaults to [4, 64], the values
        used in the original paper.
        axis (int, optional): Axis along which the operations are computed. Defaults to -1.

    Returns:
        NDArray: Time dependent modified kurtosis estimation. 
    """

    signal = np.atleast_2d(signal)

    if (len(block_sizes)!= 2):
        raise ValueError('`block_sizes` parameter must be an array of with two integers.') 

    block_sizes = np.sort(block_sizes)

    half_bksize = ((block_sizes-1)/2).astype(int)
    block_sizes = (half_bksize*2) + 1

    #suffix _l denotes large (time window), likewise _s denotes small
    pad_indexer_l = [slice(None)]*signal.ndim
    chk_indexer_l = [slice(None)]*signal.ndim
    chk_indexer_l[axis] = half_bksize[1]


    pad_indexer_s = [slice(None)]*signal.ndim
    chk_indexer_s = [slice(None)]*signal.ndim
    chk_indexer_s[axis] = half_bksize[0]

    out_indexer = [slice(None)]*signal.ndim
    output = np.zeros(signal.shape)

    # Wrap padding
    pad_width = np.zeros([signal.ndim,2],dtype=int)
    pad_width[axis,[0,1]] = [half_bksize[1]]*2
    signal_pad = np.pad(signal,pad_width=pad_width,mode='wrap')

    pad_width = np.zeros([signal.ndim,2],dtype=int)
    pad_width[axis,[0,1]] = [half_bksize[0]]*2
    signal_pad2 = np.pad(signal,pad_width=pad_width,mode='wrap')

    # Por enquanto sample a sample, depois vejo como vetorizar
    for idx in range(signal.shape[axis]):
        pad_indexer_l[axis] = slice(idx, idx+block_sizes[1])
        pad_indexer_s[axis] = slice(idx, idx+block_sizes[0])

        out_indexer[axis] = idx

        sig_chk_l = signal_pad[tuple(pad_indexer_l)] #(n,chk)
        sig_chk_s = signal_pad2[tuple(pad_indexer_s)] #(n,chk)

        chk_mean_l = np.nanmean(sig_chk_l, axis=axis, keepdims=True) #(n,)
        chk_mean_s = np.nanmean(sig_chk_s, axis=axis, keepdims=True) ##(n,)

        var_l = np.nanmean((sig_chk_l-chk_mean_l)**2, axis=axis, keepdims=True) #n,

        output[tuple(out_indexer)] = ((
            (chk_mean_s - chk_mean_l)**4 / var_l
            )**2).squeeze()

    output = np.nan_to_num(output,nan=0)
    return output.squeeze()

def tvec(data,fs):
    """Gera um vetor de temporal para um dado no tempo

    Args:
        data (ndarray or int): 
            - Caso seja um escalar, é gerado um vetor
              com len(time_vector) = data.
            - Caso seja um vetor, é gerado um vetor com
                len(time_vector) = len(data)
        fs (float): Frequência de amostragem do sinal

    Returns:
        ndarray: Um vetor temporal correspondente aos dados de entrada
    """
    if np.isscalar(data):
        n_spl = data
    else:
        n_spl = len(data)
    max_time = (n_spl - 1)/fs
    time_vector = np.linspace(0,max_time,n_spl)
    return time_vector
