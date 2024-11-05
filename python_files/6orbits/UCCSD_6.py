# This cell is added by sphinx-gallery
# It can be customized to whatever you like
# %matplotlib inline

from jax import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

import pennylane as qml
import pickle

with open('qubits_acai_6.pkl', 'rb') as file:
    qubits = pickle.load(file)
    
with open('H_acai_6.pkl', 'rb') as fp:
    H = pickle.load(fp) 
print("Number of qubits = ", qubits)

dev = qml.device("lightning.qubit", wires=qubits)

electrons = int(qubits/2)
hf = qml.qchem.hf_state(electrons, qubits)
print(hf)

singles, doubles = qml.qchem.excitations(electrons, qubits)
wires = range(qubits)

@qml.qnode(dev)
def circuit(weights, hf, singles, doubles):
    qml.templates.AllSinglesDoubles(weights, wires, hf, singles, doubles)
    return qml.expval(H)

def cost_fn(param):
    return circuit(param, hf, singles=singles, doubles=doubles)

import optax

max_iterations = 1000
conv_tol = 1e-8

opt = optax.sgd(learning_rate=0.5)

import numpy as np

theta = np.random.normal(0, np.pi, len(singles) + len(doubles))

#theta = np.array(0.)

# store the values of the cost function
energy = [cost_fn(theta)]

# store the values of the circuit parameter
angle = [theta]

opt_state = opt.init(theta)

for n in range(max_iterations):
    gradient = jax.grad(cost_fn)(theta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)
    
    angle.append(theta)
    energy.append(cost_fn(theta))

    conv = np.abs(energy[-1] - energy[-2])

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
# print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)

# Full configuration interaction (FCI) energy computed classically
E_fci = -549.9281445557749

import numpy as np
import pandas as pd

# Crie um DataFrame vazio
df = pd.DataFrame(columns=['step', 'energy'])

# Adicione as listas ao DataFrame
df['step'] = array_pares = np.arange(0, n+2)
df['energy'] = energy

# Salvar o DataFrame em um arquivo CSV
df.to_csv('VQE-UCCSD-6-1000.csv', index=False)