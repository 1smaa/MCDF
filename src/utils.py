import numpy as np
from typing import Callable
from scipy.sparse import lil_matrix, csr_matrix


class Operators(object):
    @staticmethod
    def V(x: float,y: float,h: float) -> float:
        x,y=x*h,y*h
        L=5
        if x<L: return 10
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
            hamil[i][i]=(4/hsq+V(*Operators.l_to_coord(n,i),h))
            hamil[i][(i-1)%size]=-1/hsq
            hamil[i][(i+1)%size]=-1/hsq
            hamil[i][(i+n)%size]=-1/hsq
            hamil[i][(i-n)%size]=-1/hsq
        return hamil
    
    @staticmethod
    def H_sparse(n:int,h:float,V:Callable)->csr_matrix:
        size=n**2
        hsq=h**2
        H=lil_matrix((size,size))  # Start with LIL for easy assignment
        for i in range(size):
            x,y=Operators.l_to_coord(n,i)
            H[i,i]=4/hsq+V(x,y,h)
            H[i,(i-1)%size]=-1/hsq
            H[i,(i+1)%size]=-1/hsq
            H[i,(i+n)%size]=-1/hsq
            H[i,(i-n)%size]=-1/hsq
        return H.tocsr()

    @staticmethod
    def G(H: np.ndarray,lam: float) -> np.ndarray:
        return -Operators.A(H,lam)