import numpy as np
from typing import Callable
from scipy.constants import hbar as scipy_hbar
from scipy.sparse import lil_matrix

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
    def coord_to_l(nx: int,x: int,y: int) -> int:
        '''
        Trasformazione da coordinate a indice all'interno dell'autostato appiattito.
        '''
        return nx*y+x
    
    @staticmethod
    def l_to_coord(nx: int,ny: int,l: int) -> tuple:
        '''
        Trasformazione da indice all'interno dell'autostato appiattito a coordinate.
        '''
        y=l//nx
        x=l%nx
        return x,y
    
    @staticmethod
    def H(nx: int,ny: int,hx: float,hy:float, V: Callable,m: float=None) -> np.ndarray:
        '''
        Restituisce la matrice (operatore) Hamiltoniana associata al problema
        '''
        hbar=scipy_hbar #Costante di Planck ridotta
        if m is None:
            m=1
            hbar=1
        size=nx*ny #Dimensione del vettore appiattito
        hamil=np.zeros(shape=(size,size)) #Inizializzazione della matrice
        hsqx=hx**2 #Precalcolato per calcoli più efficienti
        hsqy=hy**2
        for i in range(0,size):
            x,y=Operators.l_to_coord(nx,ny,i)
            hamil[i][i]=(hbar**2/m)*(+1/hsqx+1/hsqy)+V(x,y,hx,hy) #Sulla diagonale, potenziale calcolato in base ovviamente alla posizione
            hamil[i][Operators.coord_to_l(nx,(x+1)%nx,y)]=-hbar**2/(2*m*hsqx) #Condizioni periodiche sulle x
            hamil[i][Operators.coord_to_l(nx,(x-1)%nx,y)]=-hbar**2/(2*m*hsqx)
            hamil[i][Operators.coord_to_l(nx,x,(y+1)%ny)]=-hbar**2/(2*m*hsqy) #Condizioni periodiche sulle y
            hamil[i][Operators.coord_to_l(nx,x,(y-1)%ny)]=-hbar**2/(2*m*hsqy)
        return hamil
    
    @staticmethod
    def H_sparse(nx: int, ny: int, hx: float, hy: float, V: Callable, m: float = None) -> np.ndarray:
        hbar=scipy_hbar #Costante di Planck ridotta
        if m is None:
            m=1
            hbar=1
        size=nx*ny #Dimensione del vettore appiattito
        hamil=lil_matrix((size,size)) #Inizializzazione della matrice
        hsqx=hx**2 #Precalcolato per calcoli più efficienti
        hsqy=hy**2
        for i in range(0,size):
            x,y=Operators.l_to_coord(nx,ny,i)
            hamil[i,i]=(hbar**2/m)*(+1/hsqx+1/hsqy)+V(x,y,hx,hy) #Sulla diagonale, potenziale calcolato in base ovviamente alla posizione
            hamil[i,Operators.coord_to_l(nx,(x+1)%nx,y)]=-hbar**2/(2*m*hsqx) #Condizioni periodiche sulle x
            hamil[i,Operators.coord_to_l(nx,(x-1)%nx,y)]=-hbar**2/(2*m*hsqx)
            hamil[i,Operators.coord_to_l(nx,x,(y+1)%ny)]=-hbar**2/(2*m*hsqy) #Condizioni periodiche sulle y
            hamil[i,Operators.coord_to_l(nx,x,(y-1)%ny)]=-hbar**2/(2*m*hsqy)
        return hamil.tocsr()