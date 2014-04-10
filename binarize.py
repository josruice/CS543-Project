#!/usr/bin/python

import csv

filename1 = "training_solutions_rev1.csv"
filename2 = "binary_training_solutions.csv"

ifile = open(filename1, "r")
reader = csv.reader(ifile, delimiter=',')

ofile = open(filename2, 'w')
ofile.write("GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n")
ignore_flag = 1

for row in reader:
    if ignore_flag == 1:
        ignore_flag = 0
        continue
    
    
    galaxy_id = row[0]    
    class1 = [float(x) for x in [row[1], row[2], row[3]]]
    class2 = [float(x) for x in [row[4], row[5]]]
    class3 = [float(x) for x in [row[6], row[7]]]
    class4 = [float(x) for x in [row[8], row[9]]]
    class5 = [float(x) for x in [row[10], row[11], row[12], row[13]]]
    class6 = [float(x) for x in [row[14], row[15]]]
    class7 = [float(x) for x in [row[16], row[17], row[18]]]
    class8 = [float(x) for x in [row[19], row[20], row[21], row[22], row[23], row[24], row[25]]]
    class9 = [float(x) for x in [row[26], row[27], row[28]]]
    class10 = [float(x) for x in [row[29], row[30], row[31]]]
    class11 = [float(x) for x in [row[32], row[33], row[34], row[35], row[36], row[37]]]

    class1val = class1.index(max(class1))
    class2val = class2.index(max(class2))
    class3val = class3.index(max(class3))
    class4val = class4.index(max(class4))
    class5val = class5.index(max(class5))
    class6val = class6.index(max(class6))
    class7val = class7.index(max(class7))
    class8val = class8.index(max(class8))
    class9val = class9.index(max(class9))
    class10val = class10.index(max(class10))
    class11val = class11.index(max(class11))

    class1res = [0,0,0]
    if sum(class1) != 0:
        class1res[class1val] = 1
        
    class2res = [0,0]
    if sum(class2) != 0:
        class2res[class2val] = 1
        
    class3res = [0,0]
    if sum(class3) != 0:
        class3res[class3val] = 1
        
    class4res = [0,0]
    if sum(class4) != 0:
        class4res[class4val] = 1
        
    class5res = [0,0,0,0]
    if sum(class5) != 0:
        class5res[class5val] = 1
        
    class6res = [0,0]
    if sum(class6) != 0:
        class6res[class6val] = 1
        
    class7res = [0,0,0]
    if sum(class7) != 0:
        class7res[class7val] = 1
        
    class8res = [0,0,0,0,0,0,0]
    if sum(class8) != 0:
        class8res[class8val] = 1
        
    class9res = [0,0,0]
    if sum(class9) != 0:
        class9res[class9val] = 1
        
    class10res = [0,0,0]
    if sum(class10) != 0:
        class10res[class10val] = 1
        
    class11res = [0,0,0,0,0,0]
    if sum(class11) != 0:
        class11res[class11val] = 1

    
    line = "%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (int(galaxy_id), str(class1res).strip('[]'), str(class2res).strip('[]'), str(class3res).strip('[]'), str(class4res).strip('[]'), str(class5res).strip('[]'), str(class6res).strip('[]'), str(class7res).strip('[]'), str(class8res).strip('[]'), str(class9res).strip('[]'), str(class10res).strip('[]'),str(class11res).strip('[]'))
    ofile.write(line)

ifile.close()
ofile.close()
