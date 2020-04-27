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
data=data[:-1]

genomes = []

for i in data:
    genomes.append([])
    for j in i:
        if len(j)>0:
            genomes[-1].append(j)
    element = ''
    for k in genomes[-1][2:-2]:
        element += k    # combining the strings (i.e., genome sequences that are split) together
    the_last_two_elems = genomes[-1][-2:]

    #recreating the list
    genomes[-1] = genomes[-1][:2]
    genomes[-1].append(element)
    genomes[-1].append(the_last_two_elems[0])
    genomes[-1].append(the_last_two_elems[1])

    
