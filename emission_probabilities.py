"""
Initial Emission Probabilities

Transitions from hidden states to observations (amino acids)
"""
import initial_amino_sequence, initial_dna_sequence
amino = initial_amino_sequence.amino
dna = initial_dna_sequence.dna

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
      
def compute_emissions(diction,dna,amino):
      dna = [dna[i:i+3] for i in range(0,len(dna),3)]
      dna.remove('\n')        #dump '\n'                 
      print(amino)
      print(dna)

      #this loop returns only counts
      for i in range(len(amino)):
            diction[dna[i][0].lower()][amino[i]] += 1  #dna[i][0].lower() --> returns one of the hidden states; [amino[i]] --> returns corresponding acid

      #this loop returns probabilities
      for i in diction:
            count = 0
            for j in diction[i]:
                  count += diction[i][j]
            for j in diction[i]:
                  diction[i][j] /= count
      
      return diction

emissions = compute_emissions(diction,dna,amino)
      
            
