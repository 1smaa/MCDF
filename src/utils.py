import numpy as np
from typing import Callable


class Operators(object):
    @staticmethod
    def V(x: float,y: float,h: float) -> float:
        '''
        Definisce il potenziale utilizzato per i calcoli, in questo caso a gradino.
        '''
        x,y=x*h,y*h
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
    def l_to_coord(n: int,l: int) -> tuple:
        '''
        Trasformazione da indice all'interno dell'autostato appiattito a coordinate.
        '''
        y=l//n
        x=l%n
        return x,y
    
    @staticmethod
    def H(n: int,h: float,V: Callable) -> np.ndarray:
        '''
        Restituisce la matrice (operatore) Hamiltoniana associata al problema
        '''
        size=n**2 #Dimensione del vettore appiattito
        hamil=np.zeros(shape=(size,size)) #Inizializzazione della matrice
        hsq=h**2 #Precalcolato per calcoli più efficienti
        for i in range(0,n**2):
            hamil[i][i]=(4/hsq+V(*Operators.l_to_coord(n,i),h)) #Sulla diagonale, potenziale calcolato in base ovviamente alla posizione
            hamil[i][(i-1)%size]=-1/hsq #Condizioni periodiche sulle x
            hamil[i][(i+1)%size]=-1/hsq
            hamil[i][(i+n)%size]=-1/hsq #Condizioni periodiche sulle y
            hamil[i][(i-n)%size]=-1/hsq
        return hamil