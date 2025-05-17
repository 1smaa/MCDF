import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator,eigsh
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


def energy_functional(psi: np.ndarray,H: LinearOperator,states: list[np.ndarray]) -> np.ndarray:
    psi/=np.linalg.norm(psi)
    psi=orthogonalize(psi,states)
    energy=np.vdot(psi, H.dot(psi))
    return energy

def main() -> None:
    states=[]
    h=Operators.H_sparse(N,H,Operators.V)
    vals, vecs = eigsh(h, k=5, which='SM')
    print(f"Double check eigenvalues: {vals}")
    for i in range(EIGEN_N):
        psi_start = np.random.rand(N**2)
        psi_start /= np.linalg.norm(psi_start)
        def matvec(psi):
            return h.dot(psi)
        H_op = LinearOperator((N**2,N**2), matvec=matvec)
        sol=minimize(lambda x:energy_functional(x,H_op,states),psi_start,method="CG",options={"disp":False})
        minimum=orthogonalize(sol.x,states)
        #if len(states): print(f"Overlap {max([np.vdot(minimum,state) for state in states])}")
        #else: pass
        states.append(minimum)
        energy=np.vdot(minimum.conj(),h.dot(minimum))
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        plt.title(f"Stato {i+1}, Autoenergia > {energy}")
        x,y=np.linspace(0,L,N),np.linspace(0,L,N)
        X,Y=np.meshgrid(x,y)
        psi_matrix=np.zeros(shape=(N,N))
        for i,value in enumerate(minimum):
            x_p,y_p=Operators.l_to_coord(N,i)
            psi_matrix[x_p][y_p]=np.abs(value)**2
        ax.plot_surface(X,Y,psi_matrix.T,cmap="viridis")
        plt.show()
        

if __name__=="__main__":
    main()