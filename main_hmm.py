"""
Constructing an HMM:
T = length of observation sequence
N = number of states in the model
M = number of observation symbols
Q = distinct states of the Markov process
V = set of possible observations
A = state transition probabilities
B = observation probability matrix
pi = initial state distribution
O = observation sequence

T - (multiple)
N - 4 (number of hidden states)
M - 21 (number of unique amino acids, including Stop Codon (*)) - check observations.txt 
Q - {A, C, G, T}
V - {*, Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val}
A - import from the file initial_dna_transitions.py
B - 
pi - computed from the initial matrix (??)
O - (multiple)

HIDDEN STATES - A, C, G, T
"""

