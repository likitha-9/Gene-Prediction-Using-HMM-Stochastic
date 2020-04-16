"""
Merge all of the DNA codons into a single sequence ==> a string excluding NNN
"""

def compile_dna_sequence():
      dna = ""
      file = open("./training_set.txt", "r")
      index=0
      for line in file.readlines():
            if index%3==0:
                  split = line.split(" ")
                  for i in split:
                        if i!="NNN" and check_type(i):
                              dna += i
            index+=1
      return dna

def check_type(i):
      try:
            type(int(i))
            return False      #the integer at the end of the file isn't needed. So, return False if the type(i)=int
      except:
            return True
            
dna = compile_dna_sequence()
      
      
