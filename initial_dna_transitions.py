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
      diction = { 'a' : {'c':0, 'g':0, 't':0 },
                  'c' : {'a':0, 'g':0, 't':0 },
                  'g' : {'c':0, 'a':0, 't':0 },
                  't' : {'c':0, 'g':0, 'a':0 },
            }
      for i in range(0,len(sequence)-1):
            diction[sequence[i].lower()][sequence[i+1].lower()]+=1
            
      print(diction)
      return 0
      

dna = initial_dna_sequence.compile_dna_sequence()
matrix = probs(dna)



      



