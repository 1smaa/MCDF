# Introduzione

Il progetto rappresenta la base della prova orale per l'esame di Metodi
Computazionali della Fisica, riferito al Dipartimento di Fisica e in
particolare al corso di laurea triennale in Fisica presso l'Università
degli Studi di Padova.

## Consegna

::: center
> 7\. Equazione di Schrödinger tempo-indipendente 2D: utilizzo di un
> algoritmo di minimizzazione, come il CG (gradiente coniugato),
> implementato in `scipy`, per trovare lo stato fondamentale e i primi
> stati eccitati. Si consideri un potenziale a gradino e condizioni al
> contorno periodiche.
:::

## Obiettivo

L'obiettivo è risolvere l'equazione agli autovalori per l'Hamiltoniana,
nel contesto dell'equazione di Schrödinger bidimensionale. Sebbene la
consegna non lo specifichi esplicitamente, è richiesto il calcolo sia
degli autostati che degli autovalori: $$H\psi=E\psi$$ dove $\psi$
rappresenta l'autostato ed $E$ l'energia corrispondente.

# Implementazione Numerica dell'Hamiltoniana

L'Hamiltoniana del problema in esame è generalmente espressa come:
$$H = -\frac{\hbar^2}{2m} \nabla^2 + V(x,y) = -\frac{\hbar^2}{2m} \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right) + V(x,y)$$
Il problema viene numericamente semplificato scegliendo costanti
adimensionali (sistema di unità naturali o atomiche), ovvero:
$$\hbar=1 \hspace{10pt} m=1$$ Le energie risultanti saranno quindi
espresse in Hartree ($\approx 27.2\,\text{eV}$).

## Discretizzazione

Consideriamo un dominio rettangolare di dimensioni $L_x, L_y$,
discretizzato lungo ciascuna delle due dimensioni in $N_x, N_y$
segmenti, di lunghezza $h_x, h_y$ rispettivamente. Indicheremo con
$\psi_{i,j}$ l'elemento dell'autostato, rappresentato da un vettore di
lunghezza $N_x \cdot N_y$, che "appiattisce" il dominio bidimensionale.
L'indice $l$ è definito come $l = x + N_x \cdot y$ per indicizzare
correttamente la griglia bidimensionale.\
Approssimiamo le derivate parziali nel modo seguente:
$$\frac{\partial^2 \psi_{i,j}}{\partial x^2}=\frac{\psi_{i-1,j}-2\psi_{i,j}+\psi_{i+1,j}}{h_x^2}$$
e analogamente per la derivata seconda rispetto a $y$.

Possiamo quindi riformulare $H\psi$ come:
$$H\psi_{i,j} = -\frac{\psi_{i-1,j} - 2\psi_{i,j} + \psi_{i+1,j}}{2h_x^2}
                 - \frac{\psi_{i,j-1} - 2\psi_{i,j} + \psi_{i,j+1}}{2h_y^2}
                 + V_{i,j} \psi_{i,j}$$ che può essere riscritto
facilmente in forma matriciale.\
L'implementazione del potenziale è banale e viene dunque tralasciata.
L'adattamento a un potenziale generico può essere effettuato tramite una
funzione che, date le coordinate, restituisce il valore del potenziale
nel punto.\
Vengono imposte condizioni al contorno periodiche, implementate
facilmente tramite l'operatore modulo, che consente di accedere agli
elementi della matrice \"rientrando\" dal lato opposto del dominio.\
Da tutte queste osservazioni si ricava che l'operatore Hamiltoniano è
dato da:

    def H(nx: int,ny: int,hx: float,hy:float, V: Callable) -> np.ndarray:
        size=nx*ny # Dimensione del vettore appiattito
        hamil=np.zeros(shape=(size,size)) # Inizializzazione della matrice
        hsqx=hx**2 # Precalcolo per maggiore efficienza
        hsqy=hy**2
        for i in range(0,size):
            hamil[i][i]=(1/hsqx+1/hsqy+V(*Operators.l_to_coord(nx,ny,i),hx,hy)) # Diagonale: potenziale in funzione della posizione
            hamil[i][(i-1)%size]=-1/(2*hsqx) #Condizioni periodiche sulle x
            hamil[i][(i+1)%size]=-1/(2*hsqx)
            hamil[i][(i+nx)%size]=-1/(2*hsqy) #Condizioni periodiche sulle y
            hamil[i][(i-nx)%size]=-1/(2*hsqy)
        return hamil

## Problema degli autovalori

Sono possibili diversi approcci alla risoluzione del problema degli
autovalori, in questo caso è richiesta la risoluzione tramite
minimizzazione del funzionale di energia
$$E(\ket{\psi})=\bra{\psi} H \ket{\psi} - \lambda(\langle \psi|\psi \rangle-1)$$
Utilizzando il moltiplicatore di Lagrange $\lambda$ come vincolo sulla
normalizzazione della funzione d'onda. Dalle condizioni di minimo
$\frac{\partial E(\ket{\psi})}{\partial \bra{\psi}}=0=H\ket{\psi}-\lambda\ket{\psi}$
e
$\frac{\partial E(\ket{\psi})}{\partial \lambda}=0=\langle \psi | \psi \rangle -1$
otteniamo la soluzione al nostro problema.\
Il progetto contiene due diverse implementazioni di questo funzionale:

-   `main_laplacian.py`: minimizzazione effettuata con il funzionale
    completo anche del moltiplicatore di Lagrange

-   `main.py`: normalizzazione costretta all'interno della procedura di
    ortonormalizzazione di Gram-Schmidt, sperimentalmente più veloce e
    risultati identici, ma matematicamente meno completa.

L'algoritmo di minimizzazione richiesto è il metodo del gradiente
coniugato. Si tratta di un metodo di discesa del gradiente più costoso
dal punto di vista computazionale rispetto al metodo standard dello
steepest descent, ma che consente una convergenza più efficiente.
Permette di generare le direzioni di ricerca in maniera tale che la
minimizzazione lungo la direzione successiva non rovini le
minimizzazioni lungo le direzioni precedenti. In questo caso è stata
richiesto l'utilizzo dell'implementazione della libreria `scipy`.
