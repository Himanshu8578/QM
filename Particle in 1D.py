
import numpy as np
import matplotlib.pyplot as plt
import math

L = 1.0
N = 200

x = np.linspace(0, L, N + 2)
dx = x[1] - x[0]

H = np.zeros((N, N))

for i in range(N):

    H[i, i] = 2 / dx**2

    if i > 0:
        H[i, i-1] = -1 / dx**2

    if i < N-1:
        H[i, i+1] = -1 / dx**2

energy, wavefunctions = np.linalg.eigh(H)

print("Energy Eigenvalues")
print("n    Numerical       Analytical")

for n in range(1, 6):

    E_num = energy[n-1]
    E_ana = (n**2 * math.pi**2) / L**2

    print(n, " ", round(E_num, 5),
          " ", round(E_ana, 5))

plt.figure(figsize=(10, 6))

for n in range(4):

    psi = np.zeros(N + 2)
    psi[1:-1] = wavefunctions[:, n]

    norm = math.sqrt(np.trapezoid(psi**2, x))
    psi = psi / norm

    plt.plot(x, psi, label=f"State n={n+1}")

plt.xlabel("Position (x)")
plt.ylabel("Wavefunction")
plt.title("Particle in an Infinite Potential Box")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))

for n in range(4):

    psi = np.zeros(N + 2)
    psi[1:-1] = wavefunctions[:, n]

    norm = math.sqrt(np.trapezoid(psi**2, x))
    psi = psi / norm

    plt.plot(x, psi**2, label=f"State n={n+1}")

plt.xlabel("Position (x)")
plt.ylabel("Probability Density")
plt.title("Particle in a Box - Probability Density")
plt.legend()
plt.grid()
plt.show()
