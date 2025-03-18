import pandas as pd
import numpy as np

def vf_cte(pago, n, i):

    #Pago -> Valor de la menusalidad
    #n -> Numero de periodos
    #i -> interes en la periodicidad n

    pagos = np.ones(n)*pago
    print(pagos.sum())
    cap_factor=np.ones(n)*(1+i)**np.arange(n,0,-1)
    resp_vector = pagos*cap_factor
    resp=resp_vector.sum()

    return resp