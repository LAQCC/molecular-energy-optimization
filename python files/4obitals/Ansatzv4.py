from jax import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

import pennylane as qml
import pickle

with open('qubits_acai_4.pkl', 'rb') as file:
    qubits = pickle.load(file)
    
with open('H_acai_4.pkl', 'rb') as fp:
    H = pickle.load(fp) 
print("Number of qubits = ", qubits)

dev = qml.device("lightning.qubit", wires=qubits)

electrons = int(qubits/2)
hf = qml.qchem.hf_state(electrons, qubits)
print(hf)

@qml.qnode(dev)
def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    #Apply Hadamard to HOMO
    for i in range(electrons):
        qml.Hadamard(wires=i)
    
    #Apply RY to HOMO
    for i in range(electrons):
        qml.RY(param[i], wires=i)
        
    #Apply CNOT to HOMO(i) as control and LUMO(i) as target
    for i in range(electrons):
        qml.CNOT(wires=[i,i+electrons])
        
    #Apply RX to HOMO-LUMO(2*i)
    #Apply CNOT
    for i in range(electrons):
        qml.RX(param[i+4], wires=2*i)
        qml.CNOT(wires=[2*i, (2*i +1)])
    return qml.expval(H)

def cost_fn(param):
    return circuit(param, wires=range(qubits))

import optax

max_iterations = 1000
conv_tol = 1e-8

opt = optax.sgd(learning_rate=0.5)

theta = np.array(np.full(qubits, 0.))

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

    # if conv <= conv_tol:
    #     break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")

# import matplotlib.pyplot as plt

# fig = plt.figure()
# fig.set_figheight(5)
# fig.set_figwidth(12)

# # Full configuration interaction (FCI) energy computed classically
# E_fci = -548.0992098097419

# # Add energy plot on column 1
# ax1 = fig.add_subplot(121)
# ax1.plot(range(n + 2), energy, "go", ls="dashed")
# ax1.plot(range(n + 2), np.full(n + 2, E_fci), color="red")
# ax1.set_xlabel("Optimization step", fontsize=13)
# ax1.set_ylabel("Energy (Hartree)", fontsize=13)
# #ax1.text(0.5, -1.1176, r"$E_\mathrm{HF}$", fontsize=15)
# ax1.text(0, -548.0992098097419, r"$E_\mathrm{FCI}$", fontsize=15)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# # Add angle plot on column 2
# ax2 = fig.add_subplot(122)
# ax2.plot(range(n + 2), angle, "go", ls="dashed")
# ax2.set_xlabel("Optimization step", fontsize=13)
# ax2.set_ylabel("Gate parameter $\\theta$ (rad)", fontsize=13)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.subplots_adjust(wspace=0.3, bottom=0.2)
# plt.show()

import numpy as np
import pandas as pd

# Crie um DataFrame vazio
df = pd.DataFrame(columns=['step', 'energy'])

# Adicione as listas ao DataFrame
df['step'] = array_pares = np.arange(0, n+2)
df['energy'] = energy

# Salvar o DataFrame em um arquivo CSV
df.to_csv('Ansatzv4-4orb.csv', index=False)