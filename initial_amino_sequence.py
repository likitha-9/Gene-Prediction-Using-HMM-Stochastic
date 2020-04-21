"""
Merge all of the amino acids into a single list ==> list of tokens
"""

import os

def compile_amino_sequence():
      amino = []
      file = open("./data/training_set.txt", "r")
      index=0
      for line in file.readlines():
            if index%3==1:
                  split = line.split(" ")
                  amino += split
            index+=1
      print(amino)
      amino = cleanup(amino)
      return amino

def cleanup(amino):
      while '' in amino:
            amino.remove('')
      while '\n' in amino:
            amino.remove('\n')
      while 'Xaa' in amino:
            amino.remove('Xaa')     #DNA codon: NNN
      return amino
      
amino = compile_amino_sequence()
