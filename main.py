import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# Alberto ha l'oscilloscopio

from src.utils import Operators
N=20

L=40
H=L/N
EIGEN_N=5

def orthogonalize(psi, previous_states): 
    '''
    Procedura di ortonormalizzazione di Gram-Schmidt
    '''
    for phi in previous_states:
        psi -= (np.vdot(phi, psi)/np.vdot(phi,phi)) * phi
    psi /= np.linalg.norm(psi)
    return psi


def energy_functional(psi: np.ndarray,H: np.ndarray,states: list[np.ndarray]) -> np.ndarray:
    '''
    Funzionale dell'energia da minimizzare per la ricerca degli autovalori
    '''
    psi=orthogonalize(psi,states)
    energy=np.vdot(psi, H.dot(psi))
    return energy

def main() -> None:
    states=[] #Record degli stati passati (per essere ortogonali agli stati precedenti)
    h=Operators.H(N,H,Operators.V) #Operatore Hamiltoniano
    #vals, vecs = eigsh(h, k=5, which='SM')
    #print(f"Double check eigenvalues: {vals}")
    for i in range(EIGEN_N): #Per i primi EIGEN_N autovalori richiesti
        psi_start = np.random.rand(N**2) #Crea uno stato random da cui partire
        psi_start /= np.linalg.norm(psi_start) #Normalizzalo
        sol=minimize(lambda x:energy_functional(x,h,states), #funzione da minimizzare (si può fare anche passando gli argomenti, forse più ottimale in stack size, da vedere)
                     x0=psi_start, #Stato di partenza
                     method="CG", #Metodo del Gradiente Coniugato
                     options={"disp":False} #Non mostrare le statistiche di ottimizzazione (output più pulito)
                    ) 
        minimum=orthogonalize(sol.x,states) #Ortogonalizza lo stato finale rispetto ai precedenti
        #if len(states): print(f"Overlap {max([np.vdot(minimum,state) for state in states])}") #Pezzo tolto per verificare che effettivamente fosse ortogonale
        #else: pass
        states.append(minimum) #Aggiungi al record di stati passati quello calcolato
        energy=np.vdot(minimum.conj(),h.dot(minimum)) #Calcola autoenergia associata a questo autostato
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d') #Impostazioni di plot
        plt.title(f"Stato {i+1}, Autoenergia > {energy}")
        x,y=np.linspace(0,L,N),np.linspace(0,L,N) #Spazio delle coordinate
        X,Y=np.meshgrid(x,y) #Creazione della grid di base per il plot 3d
        psi_matrix=np.zeros(shape=(N,N)) #Creazione della matrice di base dello stato, che è stato calcolato precedentemente in modo appiattito
        for i,value in enumerate(minimum): #Trasformazione dell'autostato da vettore a matrice sulle coordinate
            x_p,y_p=Operators.l_to_coord(N,i)
            psi_matrix[x_p][y_p]=np.abs(value)**2
        ax.plot_surface(X,Y,psi_matrix.T,cmap="viridis") #Plot dell'autostato
        plt.show()
        

if __name__=="__main__":
    main()