"""
Initial transition probabilities
NxN matrix:
"""

import initial_amino_sequence 

def probs(sequence):
      keys = []
      for i in sequence:
            if i not in keys:
                  keys.append(i)
      print(keys)
      diction = {}
      for key in sorted(keys):
            if key not in diction:
                  diction[key] = {}
                  diction[key] = { diction[key][j]:0 for j in sorted(keys) }
      print(diction)

def compute_probabilities(diction):
      pass      

amino = initial_amino_sequence.compile_amino_sequence() #list of tokens

diction = probs(amino)    #dictionary of probabilities
transition_probs = compute_probabilities(diction)
print(transition_probs)
