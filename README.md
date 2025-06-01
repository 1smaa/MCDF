# Metodi Computazionali della Fisica  
**Università degli Studi di Padova**  
*Ismaele Lorenzon*  
📅 _Data: oggi_

---

## Introduzione

Il progetto rappresenta la base della prova orale per l'esame di Metodi Computazionali della Fisica, riferito al Dipartimento di Fisica e in particolare al corso di laurea triennale in Fisica presso l'Università degli Studi di Padova.

### Consegna

> **7. Equazione di Schrödinger tempo-indipendente 2D**: utilizzo di un algoritmo di  
> minimizzazione, come il CG (gradiente coniugato), implementato in `scipy`, per trovare lo  
> stato fondamentale e i primi stati eccitati. Si consideri un potenziale a gradino e  
> condizioni al contorno periodiche.

### Obiettivo

L'obiettivo è risolvere l'equazione agli autovalori per l'Hamiltoniana, nel contesto dell’equazione di Schrödinger bidimensionale. Sebbene la consegna non lo specifichi esplicitamente, è richiesto il calcolo sia degli autostati che degli autovalori:

```math
H\psi = E\psi
```

dove $\psi$ rappresenta l'autostato ed $E$ l'energia corrispondente.

---

## Implementazione Numerica dell'Hamiltoniana

L’Hamiltoniana del problema è generalmente espressa come:

```math
H = -\frac{\hbar^2}{2m} \nabla^2 + V(x,y) = -\frac{\hbar^2}{2m} \left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} \right) + V(x,y)
```

Semplifichiamo scegliendo costanti adimensionali (unità naturali o atomiche):

```math
\hbar = 1, \quad m = 1
```

Le energie risultanti saranno espresse in Hartree ($\approx 27.2\,\text{eV}$).

### Discretizzazione

Consideriamo un dominio rettangolare di dimensioni $L_x, L_y$, discretizzato lungo ciascuna delle due dimensioni in $N_x, N_y$ segmenti, di lunghezza $h_x, h_y$. Indichiamo con $\psi_{i,j}$ l’elemento dell’autostato, rappresentato da un vettore di lunghezza $N_x \cdot N_y$, ottenuto appiattendo il dominio bidimensionale.

L’indice $l$ è definito come:

```math
l = x + N_x \cdot y
```

Le derivate seconde sono approssimate come:

```math
\frac{\partial^2 \psi_{i,j}}{\partial x^2} \approx \frac{\psi_{i-1,j} - 2\psi_{i,j} + \psi_{i+1,j}}{h_x^2}
```

e analogamente per la $y$.

L’Hamiltoniana diventa quindi:

```math
H\psi_{i,j} = -\frac{\psi_{i-1,j} - 2\psi_{i,j} + \psi_{i+1,j}}{2h_x^2}
             - \frac{\psi_{i,j-1} - 2\psi_{i,j} + \psi_{i,j+1}}{2h_y^2}
             + V_{i,j} \psi_{i,j}
```

Il potenziale è implementato tramite una funzione $V(x, y)$ che restituisce il valore nel punto. Le condizioni al contorno periodiche sono implementate con l’operatore modulo.

### Operatore Hamiltoniano in Python

```python
def H(nx: int, ny: int, hx: float, hy: float, V: Callable) -> np.ndarray:
    size = nx * ny  # Dimensione del vettore appiattito
    hamil = np.zeros(shape=(size, size))
    hsqx = hx ** 2
    hsqy = hy ** 2
    for i in range(0, size):
        hamil[i][i] = (1/hsqx + 1/hsqy + V(*Operators.l_to_coord(nx, ny, i), hx, hy))
        hamil[i][(i - 1) % size] = -1 / (2 * hsqx)
        hamil[i][(i + 1) % size] = -1 / (2 * hsqx)
        hamil[i][(i + nx) % size] = -1 / (2 * hsqy)
        hamil[i][(i - nx) % size] = -1 / (2 * hsqy)
    return hamil
```

---

## Problema degli Autovalori

Il problema degli autovalori viene risolto tramite minimizzazione del funzionale di energia:

```math
\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle #1|}
E(\ket{\psi}) = \bra{\psi} H \ket{\psi} - \lambda(\langle \psi|\psi \rangle - 1)
```

dove $\lambda$ è il moltiplicatore di Lagrange che impone la normalizzazione $\langle \psi | \psi \rangle = 1$.

Dalle condizioni di minimo:

```math
\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle #1|}
\frac{\partial E(\ket{\psi})}{\partial \bra{\psi}} = 0 \Rightarrow H\ket{\psi} = \lambda \ket{\psi}
```

### Implementazioni disponibili

Il progetto contiene due implementazioni del funzionale:

- `main_laplacian.py`: utilizza il funzionale completo con moltiplicatore di Lagrange.
- `main.py`: utilizza normalizzazione tramite ortonormalizzazione di Gram-Schmidt. È più veloce sperimentalmente, ma meno rigorosa matematicamente.

---

## Metodo del Gradiente Coniugato

L’algoritmo richiesto per la minimizzazione è il **metodo del gradiente coniugato (CG)**, fornito dalla libreria `scipy`.

Questo metodo:

- È più costoso computazionalmente rispetto al metodo della discesa ripida (steepest descent).
- Offre una convergenza più rapida.
- Genera direzioni di ricerca che non interferiscono negativamente con i progressi delle iterazioni precedenti.
