import numpy as np
from typing import Callable


class Operators(object):
    @staticmethod
    def V(x: float,y: float,hx: float,hy: float) -> float:
        '''
        Definisce il potenziale utilizzato per i calcoli, in questo caso a gradino.
        '''
        x,y=x*hx,y*hy
        L=5
        if x<L: return 10
        else: return 0
    
    @staticmethod
    def coord_to_l(n: int,x: int,y: int) -> int:
        '''
        Trasformazione da coordinate a indice all'interno dell'autostato appiattito.
        '''
        return n*y+x
    
    @staticmethod
    def l_to_coord(nx: int,ny: int,l: int) -> tuple:
        '''
        Trasformazione da indice all'interno dell'autostato appiattito a coordinate.
        '''
        y=l//nx
        x=l%nx
        return x,y
    
    @staticmethod
    def H(nx: int,ny: int,hx: float,hy:float, V: Callable) -> np.ndarray:
        '''
        Restituisce la matrice (operatore) Hamiltoniana associata al problema
        '''
        size=nx*ny #Dimensione del vettore appiattito
        hamil=np.zeros(shape=(size,size)) #Inizializzazione della matrice
        hsqx=hx**2 #Precalcolato per calcoli più efficienti
        hsqy=hy**2
        for i in range(0,size):
            hamil[i][i]=(2/hsqx+2/hsqy+V(*Operators.l_to_coord(nx,ny,i),hx,hy)) #Sulla diagonale, potenziale calcolato in base ovviamente alla posizione
            hamil[i][(i-1)%size]=-1/hsqx #Condizioni periodiche sulle x
            hamil[i][(i+1)%size]=-1/hsqx
            hamil[i][(i+nx)%size]=-1/hsqy #Condizioni periodiche sulle y
            hamil[i][(i-nx)%size]=-1/hsqy
        return hamil