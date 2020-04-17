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
      diction = { 'a' : {'a':0, 'c':0, 'g':0, 't':0 },
                  'c' : {'a':0, 'c':0, 'g':0, 't':0 },
                  'g' : {'a':0, 'c':0, 'g':0, 't':0 },
                  't' : {'a':0, 'c':0, 'g':0, 't':0 } }
      for i in range(0,len(sequence)-1):
            try:
                  diction[sequence[i].lower()][sequence[i+1].lower()]+=1
            except:
                  print(i,sequence[i])
            
      print(diction)
      return diction
      

dna = initial_dna_sequence.compile_dna_sequence()
matrix = probs(dna)



      



