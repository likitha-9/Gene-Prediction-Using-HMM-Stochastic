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
      for i in range(0,len(sequence)-1):  #last character of the DNA sequence (STRING) is '\n', so it's omitted
            try:
                  diction[sequence[i].lower()][sequence[i+1].lower()]+=1
            except:
                  if(i==len(sequence)-2):
                        pass  #second to last character (STRING) doesn't play a role in the transition
                  else:
                        print(i)
            
      print(diction)
      return diction

def compute_probabilities(diction):
      for key in diction:
            count = 0
            for each_value in diction[key]:
                  count += diction[key][each_value]
            for value in diction[key]:
                  diction[key][value] = float(diction[key][value]/count)
      return diction

dna = initial_dna_sequence.compile_dna_sequence()
diction = probs(dna)    #dictionary of probabilities
transition_probs = compute_probabilities(diction)
print(transition_probs)


      



