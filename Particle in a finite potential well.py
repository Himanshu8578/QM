
import numpy as np
import matplotlib.pyplot as plt
import math

V0 = float(input("Enter potential well depth (V0): "))
a = float(input("Enter half-width of well (a): "))
N = int(input("Enter number of grid points (N): "))
XMAX = float(input("Enter plotting range (XMAX): "))
SHOW_STATES = int(input("Enter number of states to plot: "))

x = np.linspace(-XMAX, XMAX, N + 2)
dx = x[1] - x[0]

V = np.zeros(N + 2)
V[np.abs(x) < a] = -V0

H = np.zeros((N, N))

for i in range(N):
    H[i, i] = 2 / dx**2 + V[i + 1]

    if i > 0:
        H[i, i-1] = -1 / dx**2

    if i < N-1:
        H[i, i+1] = -1 / dx**2

energy, wavefunctions = np.linalg.eigh(H)

bound = np.where(energy < 0)[0]

print("\nFINITE POTENTIAL WELL")
print("----------------------------")
print("Potential depth:", V0)
print("Half-width:", a)
print("Number of bound states:", len(bound))

print("\nNumerical Energy Eigenvalues")

for j, i in enumerate(bound):
    print("State", j+1, "Energy =", round(energy[i], 6))

plt.figure(figsize=(9, 5))
plt.plot(x, V)
plt.xlabel("Position (x)")
plt.ylabel("Potential Energy")
plt.title("Finite Potential Well")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))

for j, i in enumerate(bound[:SHOW_STATES]):
    psi = np.zeros(N + 2)
    psi[1:-1] = wavefunctions[:, i]

    norm = math.sqrt(np.trapezoid(psi**2, x))
    psi = psi / norm

    plt.plot(x, psi, label=f"State {j+1}")

plt.axvline(-a, color="black", linestyle="--")
plt.axvline(a, color="black", linestyle="--")
plt.xlabel("Position (x)")
plt.ylabel("Wavefunction")
plt.title("Wavefunctions")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))

for j, i in enumerate(bound[:SHOW_STATES]):
    psi = np.zeros(N + 2)
    psi[1:-1] = wavefunctions[:, i]

    norm = math.sqrt(np.trapezoid(psi**2, x))
    psi = psi / norm

    plt.plot(x, psi**2, label=f"State {j+1}")

plt.axvline(-a, color="black", linestyle="--")
plt.axvline(a, color="black", linestyle="--")
plt.xlabel("Position (x)")
plt.ylabel("Probability Density")
plt.title("Probability Density")
plt.legend()
plt.grid()
plt.show()
