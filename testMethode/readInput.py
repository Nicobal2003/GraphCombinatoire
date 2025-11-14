import numpy as np

    # first, read input with the given r.
    def readInput(fileName):
        file = open(fileName,"r")
        readFirstRow(file)
        readMatrix(file)
        #print(fileContent)
        file.close()

#read first row of the file to obtain parameters
    def readFirstRow(file):
        firstRow =  file.readline().strip().split()
        m,n,r,lw,uw,lh,uh = map(int, firstRow)
        
#read matrix, without the first row ?
    def readMatrix(file):
        #do we start at first row ?
        X = [] 
        for _ in range(m):
            row = list(map(int, file.readline().strip().split()))
            X.append(row)

