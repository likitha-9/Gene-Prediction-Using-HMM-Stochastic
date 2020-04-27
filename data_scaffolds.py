from csv import reader

file = reader(open("./data/unplaced_sequences_126.csv",'r'))
data = [[]]
for row in file:
    for i in row:
        if 'NT_' not in i:  #first column of the csv. Because the CSV is messy/jagged, all the elements will be split with respect to an element that starts with 'NT_' within in
            data[-1].append(i)
        else:
            data.append([])
            data[-1].append(i)
            
data.remove(data[0])

unplaced = []

for i in range(0,len(data)):
    while '' in data[i]:
        data[i].remove('')
    string=''
    for j in range(1,len(data[i])):
        string += data[i][j]
    data[i] = [ data[i][0], string ]

#file.close()

file2 = open("./data_scaffolds.txt", "w")
file2.write(str(data))
