"""
Initial amino acid transition probabilities:
NxN matrix
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

def fill_dictionary(diction,amino):
      for i in range(0,len(amino)-1):
            diction[amino[i]][amino[i+1]] += 1
      return diction            
      
def compute_probabilities(fill,amino):
      for i in fill:
            count=0
            for j in fill[i]:
                  count += fill[i][j]     #count the number of transitions per amino acid
            for j in fill[i]: 
                  fill[i][j] /= count     #compute the probability
      return fill
            

amino = initial_amino_sequence.compile_amino_sequence() #list of tokens

diction = create_dictionary(amino)    #empty dictionary
fill = fill_dictionary(diction,amino)     #filled dictionary
transitions = compute_probabilities(fill,amino)   #dictionary of probabilities
print(transitions)
