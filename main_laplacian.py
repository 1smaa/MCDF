import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# Alberto ha l'oscilloscopio

from src.utils import Operators
NX=35
NY=15
LX=40
LY=160
HX=LX/NX
HY=LY/NY
EIGEN_N=5

def orthogonalize(psi, previous_states): 
    '''
    Procedura di ortonormalizzazione di Gram-Schmidt
    '''
    for phi in previous_states:
        psi -= (np.vdot(phi, psi)/np.vdot(phi,phi)) * phi
    psi /= np.linalg.norm(psi)
    return psi


def energy_functional(state: np.ndarray,H: np.ndarray,states: list[np.ndarray]) -> np.ndarray:
    '''
    Funzionale dell'energia da minimizzare per la ricerca degli autovalori
    '''
    psi=state[:-1]
    lam=state[-1]
    psi=orthogonalize(psi,states)
    energy=np.vdot(psi, H.dot(psi))-lam*(np.vdot(psi,psi)-1)
    return energy

def main() -> None:
    states=[] #Record degli stati passati (per essere ortogonali agli stati precedenti)
    h=Operators.H(NX,NY,HX,HY,Operators.V) #Operatore Hamiltoniano
    #vals, vecs = eigsh(h, k=5, which='SM')
    #print(f"Double check eigenvalues: {vals}")
    for i in range(EIGEN_N): #Per i primi EIGEN_N autovalori richiesti
        psi_start = np.random.rand(NX*NY) #Crea uno stato random da cui partire
        psi_start /= np.linalg.norm(psi_start) #Normalizzalo
        lam=0
        sol=minimize(lambda x:energy_functional(x,h,states), #funzione da minimizzare (si può fare anche passando gli argomenti, forse più ottimale in stack size, da vedere)
                     x0=np.append(psi_start,lam), #Stato di partenza
                     method="CG", #Metodo del Gradiente Coniugato
                     options={"disp":False} #Non mostrare le statistiche di ottimizzazione (output più pulito)
                    )
        psi_f=sol.x[:-1]
        lambda_f=sol.x[-1]
        states.append(psi_f) #Aggiungi al record di stati passati quello calcolato
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d') #Impostazioni di plot
        plt.title(f"Stato {i+1}, Autoenergia > {lambda_f}")
        x,y=np.linspace(0,LX,NX),np.linspace(0,LY,NY) #Spazio delle coordinate
        X,Y=np.meshgrid(x,y) #Creazione della grid di base per il plot 3d
        psi_matrix=np.zeros(shape=(NX,NY)) #Creazione della matrice di base dello stato, che è stato calcolato precedentemente in modo appiattito
        for idx,value in enumerate(psi_f): #Trasformazione dell'autostato da vettore a matrice sulle coordinate
            x_p,y_p=Operators.l_to_coord(NX,NY,idx)
            psi_matrix[x_p][y_p]=np.abs(value)**2
        ax.plot_surface(X,Y,psi_matrix.T,cmap="viridis") #Plot dell'autostato
        plt.show()
        

if __name__=="__main__":
    main()