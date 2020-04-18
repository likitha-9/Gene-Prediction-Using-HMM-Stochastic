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
      for key in sorted(keys):      #for every key in the list of amino acids
            if key not in diction:        #if key isn't already present in dictionary
                  diction[key] = {}             #then create a key in dictionary
                  for j in sorted(keys):              #then iterate over the list of keys and
                        diction[key][j]=0             #set {key: {k1:0, k2:0,...kN:0}}, where kN is the Nth key in keys
      return diction

def compute_probabilities(diction,amino):
      for i in range:
            
      pass      

amino = initial_amino_sequence.compile_amino_sequence() #list of tokens

diction = create_dictionary(amino)    #dictionary of probabilities
transition_probs = compute_probabilities(diction,amino)
print(transition_probs)
