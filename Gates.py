from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit, Gate

sns.set_theme(style="dark")

def X():
    # Pauli-X
    mat = np.array([[0.,   1.],
                    [1., 0.]])
    return Qobj(mat, dims=[[2], [2]])

def Y():
    # Pauli-Y
    mat = np.array([[0.,   -1j],
                    [1j, 0.]])
    return Qobj(mat, dims=[[2], [2]])

def Z():
    # Pauli-Z
    mat = np.array([[1.,   0.],
                    [0., -1.]])
    return Qobj(mat, dims=[[2], [2]])

def I():
    # Identitiy
    mat = np.array([[1.,   0.],
                    [0., 1.]])
    return Qobj(mat, dims=[[2], [2]])

#H = 1/np.sqrt(2)*np.matrix('1 1; 1 -1')
#X = np.matrix('0,1;1,0')
#Y = np.matrix('0,-1j; 1j,0')
#Z = np.matrix('1,0;0,-1')
#S = np.matrix('1,0;0,1j')
#T = np.matrix('1,0;0,0.70710678+0.70710678j')

snot()*X()*snot() #HXH=Z
snot()*Y()*snot() #HYH=-Y
snot()*Z()*snot() #HXH=X

#Since the Hadamard matrix only applies to one qubit, we have to tensor it with the identity to obtain the global unitary acting on the two qubits

A1 = tensor(snot(),I())
A2 = tensor(I(),snot())

#A2*csign()*A2 #In this case, the product of two Hadamard and the controlled-Z gate (csign()) yields the CNOT gate.

#tensor(snot(),snot())*cnot()*tensor(snot(),snot())

#qc1 = QubitCircuit(2)
#qc1.add_gate("H", [0,1])
#qc1.add_gate("C", 0, 1)
#qc1.add_gate("H", [0,1])
#qc1.png

#cnot()*tensor(X(),I())*cnot()

tensor(I(),snot())*cnot()*tensor(I(),snot())
