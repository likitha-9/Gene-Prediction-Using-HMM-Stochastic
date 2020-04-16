"""
Merge all of the amino acids into a single list ==> list of tokens
"""

import os

def compile_amino_sequence():
      amino = []
      file = open("./training_set.txt", "r")
      index=0
      for line in file.readlines():
            if index%3==1:
                  print(index)
                  split = line.split(" ")
                  amino += split
            index+=1
      cleanup(amino)

amino = compile_amino_sequence()
                  
      
