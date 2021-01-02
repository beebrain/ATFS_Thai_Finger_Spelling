import numpy as np 
import os
import cv2
import pickle
import random
import re
import pandas as pd

class RandomHandClass:

    def __init__(self, Data_fileImage="./25SignHand_size256_dataset_Train", \
        pathDataset="./ClassImage",ratio_val = 10,test_flag = False):
        self.pathImage = Data_fileImage
        self.pathDataset = pathDataset
        self.pathDatasetTrain = "{}_{}".format(pathDataset, "train")
        self.pathValidation = "{}_{}".format(pathDataset, "val")
        self.pathTest = "{}_{}".format(pathDataset, "test")
        self.ratio_val = ratio_val
        self.test_flag = test_flag

        ## Self dataset All
        self.labelAll = []
        self.ImageAll = []
        

    def resizeImage(self, image,desired_size=64):

        old_size = image.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(image, (new_size[1], new_size[0]))

        ### fill black color to image
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        return new_im

    def createStartStopImage(self, totalImage, valueImage, desired_size=64):
        listImage = []
        for i in range(totalImage):
            imageSelect = np.ones((3, desired_size, desired_size), dtype=np.uint8) * valueImage
            listImage += [imageSelect]
        return listImage

    def packImagetoSameShape(self, desired_size=64, shuffle_list=True):
        totalDataset_train = 0
        totalDataset_test = 0
        totalDataset_val = 0

        with open(self.pathImage, "rb") as imagePickle:
            # list dict Image 
            imageData = pickle.load(imagePickle)
            for imagekey in imageData.keys():
                listImage = []
                reindex = "{:02d}".format(int(imagekey)+2)
                for eachImage in imageData[imagekey]:
                    image = np.asarray(eachImage, dtype=np.uint8)
                    cv2.imshow("showImage", image)
                    cv2.waitKey(0)
                    new_image = self.resizeImage(image, desired_size)
                    new_image = np.transpose(new_image, (2, 0, 1))

                    

                    listImage += [new_image]
                if shuffle_list:
                    random.shuffle(listImage)

                if not self.test_flag:
                    totalImage = len(listImage)
                    ratio_val = int(totalImage * self.ratio_val / 100)
                    index_train = totalImage - ratio_val
                    listImage_train = listImage[0:index_train]
                    listImage_val = listImage[index_train:totalImage]

                    totalDataset_val += len(listImage_val)
                    totalDataset_train += len(listImage_train)

                    if not os.path.exists(self.pathDatasetTrain):
                        os.makedirs(self.pathDatasetTrain)

                    path_filename  = "{}/{}_{}_{}".format(self.pathDatasetTrain,reindex,"resizeImage",desired_size)
                    outfile = open(path_filename, 'wb')
                    pickle.dump(listImage_train, outfile)

                    if not os.path.exists(self.pathValidation):
                        os.makedirs(self.pathValidation)

                    path_filename  = "{}/{}_{}_{}".format(self.pathValidation,reindex,"resizeImage",desired_size)
                    outfile = open(path_filename, 'wb')
                    pickle.dump(listImage_val, outfile)
                
                else:
                    if not os.path.exists(self.pathTest):
                        os.makedirs(self.pathTest)


                    path_filename  = "{}/{}_{}_{}".format(self.pathTest,reindex,"resizeImage",desired_size)
                    outfile = open(path_filename, 'wb')
                    pickle.dump(listImage, outfile)

                    totalDataset_test += len(listImage)

            if not self.test_flag:
                ###### write create stop Image
                listImage = self.createStartStopImage(int(totalDataset_train / 25), 255, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathDatasetTrain,"01","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

                ###### write create padding Image
                listImage = self.createStartStopImage(int(totalDataset_train / 25), 0, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathDatasetTrain,"00","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

                ###### write create start Image
                listImage = self.createStartStopImage(int(totalDataset_train / 25), 125, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathDatasetTrain,"02","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

                ######################## validation #################
                ###### write create stop Image
                listImage = self.createStartStopImage(int(totalDataset_val / 25), 255, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathValidation,"01","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

                ###### write create padding Image
                listImage = self.createStartStopImage(int(totalDataset_val / 25), 0, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathValidation,"00","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

                ###### write create start Image
                listImage = self.createStartStopImage(int(totalDataset_val / 25), 125, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathValidation,"02","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)
            else:
                ############## create Test dataset #######
                ###### write create stop Image
                listImage = self.createStartStopImage(int(totalDataset_test / 25), 255, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathTest,"01","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

                ###### write create padding Image
                listImage = self.createStartStopImage(int(totalDataset_test / 25), 0, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathTest,"00","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

                ###### write create start Image
                listImage = self.createStartStopImage(int(totalDataset_test / 25), 125, desired_size)
                path_filename  = "{}/{}_{}_{}".format(self.pathTest,"02","resizeImage",desired_size)
                outfile = open(path_filename, 'wb')
                pickle.dump(listImage, outfile)

    def CheckImageDataset(self):
            
        listDatasetFiles = os.listdir(self.pathDataset)
        for datasetFile in listDatasetFiles:

            with open("{}/{}".format(self.pathDataset,datasetFile), "rb") as imagePickle:
                # list dict Image 
                imageData = pickle.load(imagePickle)
                for eachImage in imageData:
                        image = np.asarray(eachImage, dtype=np.uint8)
                        cv2.imshow("showImage", image)
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
    
    def readAllDatabase(self,pathRead):
        listDatasetFiles = os.listdir(pathRead)
        if len(listDatasetFiles) == 0:
            self.packImagetoSameShape()

        self.dictImage = {}
        for datasetFile in listDatasetFiles:
            className  = datasetFile.split("_")[0]
            with open("{}/{}".format(pathRead,datasetFile), "rb") as imagePickle:
                # list dict Image 
                self.dictImage[className] = pickle.load(imagePickle)
                

                ## extract to list
                for imageA in self.dictImage[className]:
                    imageSelect = imageA
                    ## ready to feed to CNN
                    self.ImageAll += [imageSelect]
                    self.labelAll += [className]

    def helperReturnImage(self, signSequence, desired_size=128):
        listImageReturn = []
        for sign in signSequence:
            sign = sign +1
            sign = "{:02d}".format(sign)
            ListImage = self.dictImage[sign]
            lenTotalImage = len(ListImage)
            indexImage = random.randrange(0, lenTotalImage - 1)
            imageSelect = ListImage[indexImage]
            # gray = cv2.cvtColor(imageSelect, cv2.COLOR_BGR2GRAY)
            
            # imageSelect = np.reshape(imageSelect,(3,desired_size,desired_size))
            # gray = np.reshape(gray,(1,128,128))
            imageSelect = imageSelect.transpose(2,0, 1)
            listImageReturn += [imageSelect]
            # cv2.imshow("showImage", ListImage[0])
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            
        return np.array(listImageReturn)

    def helperReturnImageSequence(self, signSequence,desired_size = 64):
        listImageReturn = []
        for sign in signSequence:

            # if sign == 1: ## stop sign
            #     imageSelect = np.ones((3,desired_size,desired_size),dtype=np.uint8)*255
            # elif sign == 0: ## padding sign
            #     imageSelect = np.zeros((3,desired_size,desired_size),dtype=np.uint8)
            # elif sign == 2:
            #     imageSelect = np.zeros((3, desired_size, desired_size), dtype=np.uint8)
            # else:
            #     sign = sign -2
            sign = "{:02d}".format(sign)
            ListImage = self.dictImage[sign]
            lenTotalImage = len(ListImage)
            indexImage = random.randrange(0, lenTotalImage - 1)
            imageSelect = ListImage[indexImage]

            # imageSelect = np.transpose(imageSelect, (2, 0, 1))
            
            # imageSelect = np.reshape(imageSelect,(3,desired_size,desired_size))
            listImageReturn += [imageSelect]
            # imageShow = np.transpose(imageSelect,(1,2,0))
            # cv2.imshow("showImage", imageShow)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            
        return np.array(listImageReturn)

    def helperBatchAll(self):
        return label,image


class RandomFeature:
    def __init__(self, pathModelName,annotationFile,testlistFile):
        self.model_name = pathModelName
        self.model_path_name = pathModelName.replace(".","_")
        self.feature_seq = "./Feature_seq/{}/".format(self.model_path_name)
        self.annotationFile = annotationFile
        self.testlistFile = testlistFile
        
        ## Self dataset All
        


        self.train_featureAll = {}
        self.train_predictAll = {}

        self.test_featureAll = {}
        self.test_predictAll = {}

        self.val_featureAll = {}
        self.val_predictAll = {}

    def readAllDatabase(self):

        Annotation_frame = self.annotationFile
        signIndexFile = self.testlistFile

        annotationFile_data = pd.read_csv("./{}".format(Annotation_frame))
        annotationFile_test_index = pd.read_csv(signIndexFile, names=['indexTest'], header=None)


        index_sign_test = annotationFile_test_index["indexTest"] - 1
        
        # annotation_file_list = annotationFile_data['File Name']
        annotaion_file_list_train = annotationFile_data.iloc[~annotationFile_data.index.isin(index_sign_test)]
        annotaion_file_list_test = annotationFile_data.iloc[index_sign_test]
        print("Total testDataset {}".format(len(annotaion_file_list_test)))
        print("Total trainDataset {}".format(len(annotaion_file_list_train)))


        ##################### load train Predict feature ################
        for datasetFile in annotaion_file_list_train.T.iteritems():

            datasetFile = datasetFile[1]
            fileName_pickle = datasetFile["File Name"]
            alphabet_gt = datasetFile["Index"]
            sign_gt = datasetFile["concatSign"]

            

        


    #     ##################### load val Predict feature ################
    #     listDatasetFiles = os.listdir(self.feature_val)
    #     reg_compile = re.compile("\w*predict")
    #     predict_Files = []
    #     predict_Files = predict_Files + [filemodel for filemodel in listDatasetFiles if reg_compile.match(filemodel)]
        
    #     for datasetFile in predict_Files:
    #         className  = datasetFile.split("_")[0]
    #         with open("{}/{}".format(self.feature_val,datasetFile), "rb") as imagePickle:
    #             # list dict Image 
    #             self.val_predictAll[className] = pickle.load(imagePickle)
        
    #     ##################### load test Predict feature ################
    #     listDatasetFiles = os.listdir(self.feature_test)
    #     reg_compile = re.compile("\w*predict")
    #     predict_Files = []
    #     predict_Files = predict_Files + [filemodel for filemodel in \
    #         listDatasetFiles if reg_compile.match(filemodel)]
        
    #     for datasetFile in predict_Files:
    #         className  = datasetFile.split("_")[0]
    #         with open("{}/{}".format(self.feature_test,datasetFile), "rb") as imagePickle:
    #             # list dict Image 
    #             self.test_predictAll[className] = pickle.load(imagePickle)
        
    #     ##################### CNN Feature Zone Load #################

    #     ##################### load train CNN feature ################
    #     listDatasetFiles = os.listdir(self.feature_train)
    #     reg_compile = re.compile("\w*feature")
    #     predict_Files = []
    #     predict_Files = predict_Files + [filemodel for filemodel in listDatasetFiles if reg_compile.match(filemodel)]
        
    #     for datasetFile in predict_Files:
    #         className  = datasetFile.split("_")[0]
    #         with open("{}/{}".format(self.feature_train,datasetFile), "rb") as imagePickle:
    #             # list dict Image 
    #             self.train_featureAll[className] = pickle.load(imagePickle)


    #     ##################### load val CNN feature ################
    #     listDatasetFiles = os.listdir(self.feature_val)
    #     reg_compile = re.compile("\w*feature")
    #     predict_Files = []
    #     predict_Files = predict_Files + [filemodel for filemodel in listDatasetFiles if reg_compile.match(filemodel)]
        
    #     for datasetFile in predict_Files:
    #         className  = datasetFile.split("_")[0]
    #         with open("{}/{}".format(self.feature_val,datasetFile), "rb") as imagePickle:
    #             # list dict Image 
    #             self.val_featureAll[className] = pickle.load(imagePickle)
        
    #     ##################### load test CNN feature ################
    #     listDatasetFiles = os.listdir(self.feature_test)
    #     reg_compile = re.compile("\w*feature")
    #     predict_Files = []
    #     predict_Files = predict_Files + [filemodel for filemodel in \
    #         listDatasetFiles if reg_compile.match(filemodel)]
        
    #     for datasetFile in predict_Files:
    #         className  = datasetFile.split("_")[0]
    #         with open("{}/{}".format(self.feature_test,datasetFile), "rb") as imagePickle:
    #             # list dict Image 
    #             self.test_featureAll[className] = pickle.load(imagePickle)
        






    # def helperReturnFeatureSequence(self, signSequence, mode="train"):
    #     if mode == "train":
    #         dictFeature = self.train_predictAll
    #     elif mode == "val":
    #         dictFeature = self.val_predictAll
    #     elif mode == "test":
    #         dictFeature = self.test_predictAll
    #     else:
    #         print("select sequence mode train,validation or test")
    #         exit()

    #     listfeatureReturn = []
    #     for sign in signSequence:

    #         sign = "{:02d}".format(sign)
    #         Listfeature = dictFeature[sign]
    #         lenTotalfeaure = len(Listfeature)
    #         indexImage = random.randrange(0, lenTotalfeaure - 1)
    #         featureSelect = Listfeature[indexImage].reshape(-1)
    #         listfeatureReturn += [featureSelect]

    #     return np.array(listfeatureReturn)
    
if __name__ == "__main__":

    # rHC = RandomHandClass(Data_fileImage="25SignHand_size256_dataset_Train",test_flag=False)
    # rHC.packImagetoSameShape()
    # rHC.readAllDatabase("./ClassImage_train")
    # rHC.helperReturnImageSequence(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]))

    rFC = RandomFeature("CNN_Sign_0_946_T_0_810")
    rFC.readAllDatabase()
    feature = rFC.helperReturnFeatureSequence(np.array([3,1]))
    print(feature)
    
