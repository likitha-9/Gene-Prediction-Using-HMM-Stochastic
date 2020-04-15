"""
A dictionary of {amino_acid: {list of DNA codons}}
"""

import os

#open needed files
obs = open("./observations.txt", "r")
train = open("./training_set.txt", "r")

pairings = {}

def remove_whitespace(line):
      while '' in line:
            line.remove('')
      return line

for i in obs.readlines():
      pairings[i[:-1]] = []   #the slicing is to remove any immediate delimiter (following the name)

count=0
while True:
      try:
            line1 = train.readline().split(" ") #DNA codons
            line2 = train.readline().split(" ") #amino acids
            whitespace = train.readline() #discarded

            line1 = remove_whitespace(line1)
            line2 = remove_whitespace(line2)
            
            for i in range(0,len(line2)):
                  try:
                        if line2[i] in pairings:
                              if line1[i] not in pairings[line2[i]]:
                                    pairings[line2[i]].append(line1[i])
                  except:
                        break
            count+=1
            print(count)
      except:
            break
print(pairings)
