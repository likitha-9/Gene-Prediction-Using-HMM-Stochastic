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
V - import from the file ./data/observations.txt - [*, Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val]
A - import from the file hidden_state_transitions.py
B - import from the file emission_probabilities.py
pi - start --> [0.25, 0.25, 0.25, 0.25]
O - (multiple)

HIDDEN STATES - Q - [A, C, G, T]
OBSERVATION STATES  - V - [*, Ala, Arg, Asn, Asp, Cys, Gln, Glu, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val]
"""

#program imports
import emission_probabilities as emissions, hidden_state_transitions as hidden
#data imports
import importlib.util
genome_data_import = importlib.util.spec_from_file_location("genome_data.py", "./data")
genomes = importlib.util.module_from_spec(genome_data_import)
genome_data_import.loader.exec_module(genomes)

scaffold_data_import = importlib.util.spec_from_file_location("unplaced_data.py", "./data")
scaffolds = importlib.util.module_from_spec(scaffolds_data_import)
scaffold_data_import.loader.exec_module(scaffolds)

#library imports
import numpy as np, matplotlib.pyplot as plt




