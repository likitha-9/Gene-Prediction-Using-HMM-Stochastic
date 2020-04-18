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

"""
Initial transition probabilities (after computation)
4x4 matrix:

	      	A      	      	      	      C      	      	      	G	      	      	      	T
A	0.239710240368785      	    0.24234441883437602      	0.37141916364833716      	0.1465261771485018

C	0.2737484737484738          0.35262515262515265         0.09084249084249084             0.2827838827838828

G	0.20955595949104128         0.2890158400415476          0.33575694624772784             0.1656712542196832

T	0.135087095627444           0.28546036260220403         0.37611091361535726             0.20334162815499468
"""

      



