from pennylane import numpy as np
import pennylane as qml
import pickle
   
with open('H_acai_4.pkl', 'rb') as fp:
    H = pickle.load(fp) 

energia = np.linalg.eigh(H.sparse_matrix().toarray())[0].min()

print(energia) 