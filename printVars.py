import numpy as np

numFiles = 10
for aa in range(1, numFiles+1):
    file = np.load("./FinalVals/Trial" + str(aa) + ".npy", allow_pickle=True)
    print(file)
