"""
Initial Emission Probabilities

Transitions from hidden states to observations (amino acids)
"""
import initial_amino_sequence
amino = initial_amino_sequence.amino

hidden = ["a","c","g","t"]

file = open("./observations.txt",'r')
obs = []
for line in file.readlines():
      obs.append(line[:-1])   #'\n' not needed
print(obs)

#file = open("./observations.txt","r")
#amino 
