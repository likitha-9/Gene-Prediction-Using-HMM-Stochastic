import os

file = open("./data/training_set.txt", "r")

#observations
obs = []

#read the file
for index, line in enumerate(file.readlines()):
      if (index+1)%3==2:      #lines 2, 5, 8, etc.
            #under each corresponding DNA codon, check if each amino acid is in list of observations
            x = line.split(" ")
            for i in x:
                  if i not in obs:
                        obs.append(i)

#remove extraneous ones
extra = ['Xaa', '', '\n', '\t']
for i in extra:
      while i in obs:
            obs.remove(i)

file.close()

print(sorted(obs))

#store the observations into a file
file = open("./data/observations.txt", "w")
for i in sorted(obs):
      file.write(i+'\n')
file.close()
                        
