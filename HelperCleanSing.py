import numpy as np 
import csv
import os
import time

class CleanSequence():
    def __init__(self):
        pass

    def cleanSequenceM1(self,sequencePredict,frameCon = 5):

        # sign need to concat morethan 5 times that will be sign frame
        # if not it will be non sign status
        # 
        frameCon +=1
        sequencePredict = np.asarray(sequencePredict)
        print(sequencePredict)
        considerList = sequencePredict[0:frameCon]
        for indexFrame in range(len(sequencePredict) - (frameCon-1)):
            considerList = sequencePredict[indexFrame:indexFrame + frameCon]
            # print(considerList)
            lastPredict = considerList[-1]
            for considerIndex in range(3, -1, -1):
                if considerList[considerIndex] == lastPredict:
                    considerList[considerIndex:] = lastPredict
                elif considerList[considerIndex] > 0 and lastPredict > 0:
                    # print(considerList[considerIndex+1:-1])
                    if( np.sum(considerList[considerIndex+1:-1])) == 0:
                        considerList[considerIndex+1:] = lastPredict

        print(sequencePredict)
        return (sequencePredict)

if __name__ == "__main__":
    # listsequencePredict = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 5, 5, 5, 0, 0]
    listsequenc1 = [0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 3, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0]
    listsequenc1 = [0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 0, 3, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0]

    
    
    cs = CleanSequence()
    # cs.cleanSequenceM1(listsequencePredict)
    cs.cleanSequenceM1(listsequenc1)
    

                                    
                    
