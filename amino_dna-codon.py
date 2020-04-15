"""
A dictionary of {amino_acid: {list of DNA codons}}
"""

import os

#open needed files
obs = open("./observations.txt", "r")
train = open("./training_set.txt", "r")

pairings = {}

for i in obs.readlines():
      pairings[i[:-1]] = []   #the slicing is to remove any immediate delimiter (following the name)

while True:
      line1 = train.readline().split(" ")
      line2 = train.readline().split(" ")
      whitespace = train.readline() #discarded

      print(line1, line2)
