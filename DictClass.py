import numpy as np
import codecs
class Dict_sign():
    def __init__(self,name):
        self.name = name
        self.word2token = {"0":0,"1":1,"2":2}
        self.token2word = {0:"p",1:"e",2:"s"}
        self.n_words = len(self.token2word)
        
    def genDict(self,start =1):
        for word in range(start,start+25):
            self.word2token[str(word)] = self.n_words    ### sign number 1 is token number 2 (token 0 is s and token 1 is e)
            self.token2word[self.n_words] = str(word)
            self.n_words +=1

class Dict_thaiAlphabet():
    def __init__(self,name):
        self.name = name
        self.word2token = {"p":0,"e":1,"s":2}
        self.token2word =  {0:"p",1:"e",2:"s"}
        self.word2count = {}
        self.n_words = len(self.token2word)
            
    def genDict(self):
        firstword = bytes('ก', 'utf8')
        for index in range(46):
            lastbyte = firstword[2]+index
            
            if(index==37 or index == 35 or index == 2 or index == 4):
                continue
            newword = bytes([firstword[0],firstword[1],lastbyte])
            self.word2token[newword.decode('utf-8')] = self.n_words
            self.token2word[self.n_words] = newword.decode('utf-8')
            self.n_words +=1
 
class GenThaiAlphabet():

    def __init__(self, combination_num,nameFile):
        self.CB_num = combination_num
        self.NameFile = nameFile

    def mapCharactor(self, listofAlphabet, arrayIndex):
        listChar = []
        for indexA in arrayIndex:
            listChar +=[str(listofAlphabet[indexA].decode())]
        return listChar

    def genDictAlphabet(self):
        indexCom = np.zeros((self.CB_num), dtype=np.uint8)
        firstword = bytes('ก', 'utf8')
        listofAlphabet = []
        for index in range(46):
            lastbyte = firstword[2]+index
            
            if(index==37 or index == 35 or index == 2 or index == 4):
                continue
            newword = bytes([firstword[0],firstword[1],lastbyte])
            listofAlphabet += [newword]

        langeIndex = self.CB_num
        indexAlphabet = np.zeros([langeIndex], dtype=int)
        totalLoop = np.prod(np.ones([langeIndex],dtype=int)*42)
        indexpointer = 0
        with codecs.open(self.NameFile, 'w', encoding='utf8') as f:
            for in_com in range(totalLoop):
                listOfchar = self.mapCharactor(listofAlphabet, indexAlphabet)
                printChar = "".join(listOfchar)
                textprint = "{}\n".format(printChar)
                # print(textprint)
                f.write(textprint)
                for pointerindex,pointerValue in enumerate(indexAlphabet):
                    indexAlphabet[pointerindex] += 1
                    if indexAlphabet[pointerindex] <= 41:
                        break
                    else:
                        indexAlphabet[pointerindex] = 0
                    

if __name__ == "__main__":
    Dta = GenThaiAlphabet(2,"2Combination.txt")
    Dta.genDictAlphabet()
    