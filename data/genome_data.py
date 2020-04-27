import pandas as pd
from csv import reader

file = reader(open("./genome_sequences_from_22_X_Y_chromosomes.csv",'r'))
data = [[]]
for row in file:
    for i in row:
        if 'KB' not in i:  #last column of the csv. Because the CSV is messy/jagged, all the elements will be split with respect to an element that has 'KB' within in
            data[-1].append(i)
        else:
            data[-1].append(i)
            data.append([])
data[0]=data[0][5:]
clean = []
for i in data:
    clean.append([])
    for j in i:
        if len(j)>0:
            clean[-1].append(j)
    element = ''
    for k in clean[-1][2:-2]:
        element += k    # combining the strings (i.e., genome sequences that are split) together
    the_last_two_elems = clean[-1][-2:]

    #recreating the list
    clean[-1] = clean[-1][:2]
    clean[-1].append(element)
    clean[-1].append(the_last_two_elems)
