import numpy as np
from pennylane import qchem
import pickle

symbols, coordinates = qchem.read_structure("acai-bohr.xyz")
H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, method="pyscf", active_electrons=6, active_orbitals=6)

file_path = 'H_acai_6.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(H, file)

file_path2 = 'qubits_acai_6.pkl'
with open(file_path2, 'wb') as file2:
    pickle.dump(qubits, file2)
    
# print(qubits)
# print(H)