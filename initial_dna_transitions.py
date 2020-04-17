"""
Initial transition probabilities
4x4 matrix:

	A	C      	G	T
A	-	-	-	-

C	-	-	-	-

G	-	-	-	-

T	-	-	-	-
"""

import initial_dna_sequence

def probs(sequence):
      #initial dictionaries
      a = {'c':0, 'g':0, 't':0 }
      c = {'a':0, 'g':0, 't':0 }
      g = {'c':0, 'a':0, 't':0 }
      t = {'c':0, 'g':0, 'a':0 }
      print(sequence)
      return 0
      

dna = initial_dna_sequence.compile_dna_sequence()
matrix = probs(dna)



      



