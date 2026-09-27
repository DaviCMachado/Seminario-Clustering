import numpy as np
def modified_kneedle(x,y,axis=-1):
    """Retorna indíce onde se encontra knee point utilizando 
    implementação simplificada de Kneedle. O método considera a inflexão 
    como o ponto de maior distância perpendicular à um segmento de 
    reta formado por (x0,y0) e (xn,yn), em que n representa o 
    último par ordenado do meu conjunto de dados.

    A distância entre ponto e reta, então, é determinada por
    formula clássica da geometria analítica [1]: 
    ```
        d = abs((y[-1]-y[0])*x -(x[-1]-x[0])*y + x[-1]*y[0] -y[-1]*x[0])
        d /= np.sqrt((y[-1] - y[0])**2 + (x[-1] - x[0])**2)
        return d.argmax()
    ```

    A maior diferença entre essa função e Kneedle é a ausência de procedimento
    para detecção de inflexões locais, visto que essa implementação era aplicada
    para detecção de inflexão em curva bem comportada(com um knee único)

    References:
        [1]: [Wikipedia: Distance from a point to a line](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line)
        [2]: Satopaa et al. Finding a “Kneedle” in a Haystack: Detecting Knee Points 
        in System Behavior (2011).

    Args:
        x (ndarray): Valores de x
        y (ndarray): Valores de y

    Returns:
        int: índice em que se encontra ponto de inflexão geométrico.

    """
    np_kw = dict(axis=axis,keepdims=True)
    # Normalizing x and y
    x = (x - np.min(x,**np_kw)) / (np.max(x,**np_kw) - np.min(x,**np_kw))
    y = (y - np.min(y,**np_kw)) / (np.max(y,**np_kw) - np.min(y,**np_kw))    

    d = abs((y[-1]-y[0])*x -(x[-1]-x[0])*y + x[-1]*y[0] -y[-1]*x[0])
    d /= np.sqrt((y[-1] - y[0])**2 + (x[-1] - x[0])**2)

    return abs(d).argmax()


def gap_grouping(arr,thresh,kind='min'):
    """ 
    When dealing with single dimentional sparse signals, the gap measurement 
    can serve as a reliable method for estimating boundaries. This function 
    computes separates boundaries based on the minimum threshold of data
    separation to be considered a gap. Then, the group representative position
    is given by the median of non zero values between the boundaries.

    Args:
        arr (_type_): _description_
        thresh (_type_): _description_

    Returns:
        _type_: _description_
    """
    if kind=='min':
        finder = np.min
    elif kind=='median':
        finder = np.median
    elif kind == 'mean':
        finder = np.mean
    bound = arr[np.where(np.diff(arr)>thresh)[0]]
    bound = np.hstack([0,bound])

    medians = np.zeros(len(bound)-1,dtype=int) #n groups
    for n in range(len(medians)): 
        bd_l = bound[n]
        bd_u = np.clip(bound[n+1])
        idx_in_bd = arr[(arr>bd_l) & (arr<=bd_u)]
        medians[n] = int(finder(idx_in_bd))
    return medians
