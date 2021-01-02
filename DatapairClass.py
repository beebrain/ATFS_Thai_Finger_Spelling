import DictClass
import numpy as np
import torch
import random


class DatasetManagePair():

    def prepareData(self,feature = "class_seq",Train=True,reverse=True):
        input_lang, output_lang = self.readLangs(feature,reverse)
        print("Counted char:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang
    


    def readLangs(self,feature="class_seq",Train=True,reverse=False):
        print("Reading lines...")

            
        # pairs = []
        # for indexData,eachData in enumerate(dataX):
        #     pairs += [[dataY[indexData],eachData]]

        input_lang = DictClass.Dict_sign("Sign")
        output_lang = DictClass.Dict_thaiAlphabet("Thai")

        input_lang.genDict(start=3)  #skip for 0 padding 1 start 2 stop used word2token <convert sign number to token>
        output_lang.genDict() ## convert token class to alphabet

        return input_lang, output_lang

if __name__ == "__main__":
    # dmc = DatasetManageCombile(["./1Combination_dataset.txt", \
    #     "./2Combination_dataset.txt"], \
    #     [0.5, 0.5])
    
    dmc = DatasetManagePair()
    input_lang, output_lang = dmc.prepareData()