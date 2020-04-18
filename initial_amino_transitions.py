"""
Initial transition probabilities
NxN matrix:
"""

import initial_amino_sequence 

def create_dictionary(sequence):
      keys = []
      for i in sequence:
            if i not in keys:
                  keys.append(i)
      print(keys)
      diction = {}
      for key in sorted(keys):
            if key not in diction:
                  diction[key] = {}
                  for j in sorted(keys):
                        diction[key][j]=0
      return diction

def compute_probabilities(diction,amino):
      for i in range:
            
      pass      

amino = initial_amino_sequence.compile_amino_sequence() #list of tokens

diction = create_dictionary(amino)    #dictionary of probabilities
transition_probs = compute_probabilities(diction,amino)
print(transition_probs)
