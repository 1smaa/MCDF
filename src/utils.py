import numpy as np
from typing import Callable

class Operators(object):
    @staticmethod
    def A(H: np.ndarray,lam: float) -> np.ndarray:
        return H-lam*np.identity(H.shape[0])
        
    @staticmethod
    def V(x: float,y: float) -> float:
        L=5
        if x<L: return 1
        else: return 0
    
    @staticmethod
    def coord_to_l(n: int,x: int,y: int) -> int:
        return n*y+x
    
    @staticmethod
    def l_to_coord(n: int,l: int) -> tuple:
        y=l//n
        x=l%n
        return x,y
    
    @staticmethod
    def H(n: int,h: float,V: Callable) -> np.ndarray:
        size=n**2
        hamil=np.zeros(shape=(size,size))
        hsq=h**2
        for i in range(0,n**2):
            hamil[i][i]=(4/hsq+V(*Operators.l_to_coord(n,i)))
            hamil[i][(i-1)%size]=-1/hsq
            hamil[i][(i+1)%size]=-1/hsq
            hamil[i][(i+n)%size]=-1/hsq
            hamil[i][(i-n)%size]=-1/hsq
        return hamil
    
    @staticmethod
    def G(H: np.ndarray,lam: float) -> np.ndarray:
        return -Operators.A(H,lam)