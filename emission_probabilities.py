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
file.close()

diction = {}
for i in hidden:
      diction[i] = {}
      for state in sorted(obs):
            diction[i][state]=0
      
def compute_emissions(diction,amino):
      """for i in range(0,len(amino)-1):
            try:
                  \"""
                  ********************************************************
                                  EDIT THE BELOW STATEMENT.
                  ********************************************************
                  \"""
                  diction[amino[i][0].lower()][amino[i+1]] += 1  #computed counts
            except:
                  pass
      for i in diction:
            count=0
            for j in diction[i]:
                  count += diction[i][j]
            for j in diction[i]:
                  diction[i][j] /= count
      return diction"""

emissions = compute_emissions(diction,amino)
      
            
