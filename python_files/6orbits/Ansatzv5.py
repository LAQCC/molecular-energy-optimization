# This cell is added by sphinx-gallery
# It can be customized to whatever you like
#%matplotlib inline

from jax import numpy as np
import jax
import numpy as np
import pandas as pd
import pennylane as qml
import pickle
import optax
import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)


with open('qubits_acai_6.pkl', 'rb') as file:
    qubits = pickle.load(file)
    
with open('H_acai_6.pkl', 'rb') as fp:
    H = pickle.load(fp) 
print("Number of qubits = ", qubits)

dev = qml.device("lightning.qubit", wires=qubits)

electrons = int(qubits/2)
hf = qml.qchem.hf_state(electrons, qubits)
print(hf)

@qml.qnode(dev)
def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    for i in range(electrons*2):
        qml.Hadamard(wires=i)
    for i in range(electrons*2):
        qml.RY(param[i], wires=i)  
        
    for i in range (int(electrons/2)):
        qml.CNOT(wires=[i, i+ int(electrons/2)])
    
    return qml.expval(H)

def cost_fn(param):
    return circuit(param, wires=range(qubits))


max_iterations = 1000
conv_tol = 1e-8

opt = optax.sgd(learning_rate=0.5)

import numpy as np


theta = np.full(qubits, 0.)
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
        # Adicionar valores ao DataFrame

    # if conv <= conv_tol:
    #     break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")


# Crie um DataFrame vazio
df = pd.DataFrame(columns=['step', 'energy'])

# Adicione as listas ao DataFrame
df['step'] = array_pares = np.arange(0, n+2)
df['energy'] = energy

# Salvar o DataFrame em um arquivo CSV
df.to_csv('VQE-teste2_HE-1000.csv', index=False)