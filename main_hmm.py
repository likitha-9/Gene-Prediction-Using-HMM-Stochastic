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
Q - [A, C, G, T]
V - [*, Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val]
A - import from the file hidden_state_transitions.py
B - import from the file emission_probabilities
pi - start --> [0.25, 0.25, 0.25, 0.25]
O - (multiple)

HIDDEN STATES - Q - [A, C, G, T]
OBSERVATION STATES  - V - [*, Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val]
"""

