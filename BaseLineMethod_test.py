import numpy as np
import os
from pathlib import Path
import pickle
import codecs
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import time
from VGG16ClassCNN import ConvNetExtract
import csv
import cv2
import HandProcess as hp
from jiwer import wer
import HelperCleanSing as hcs


def intVideo(videoFile="../VideoFile_2Combi/Nested Sequence 04.mp4"):
    camera = cv2.VideoCapture('{}'.format(videoFile))
    amount_of_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    # camera.set(cv2.CAP_PROP_FRAME_COUNT, current_frame - 1)
    # cv2.namedWindow('ca1', 0)
    res, frame = camera.read()
    return camera, amount_of_frames

def VideoProcess(videoFile, startFrame, endFrame ,size=64):
    cameraObj,_ = intVideo(videoFile)
    current_frame = startFrame
    signFrame = {"startSign": [], "stopSign": []}
    transitionFrame = {"startT":[],"stopT":[]}
    startSign = False
    startT = False
    handPro = hp.HandProcess()
    listImage_frame = []

    while current_frame <= endFrame:
        # choice = cv2.waitKey(0)


        cameraObj.set(1 ,current_frame)
        # print(int(cameraObj.get(1)))

        res, frame = cameraObj.read()
        ROI, (x, y, w, h), (cX, cY), mark = handPro.process(frame)
        
        if not (cX == 0 and cY == 0):


            crop_img = frame[y:(y + h), x:(x + w)]
            crop_img = cv2.resize(crop_img, (size, size))
            # print(crop_img.shape)
            # cv2.imshow("crop_img", crop_img)

            height, width, _ = frame.shape
            # cv2.resizeWindow('ca1', width, height)
            # cv2.imshow("ca1", frame)
            
            listImage_frame += [crop_img]
            # cv2.waitKey(0)

        current_frame += 1        
        
    return listImage_frame



def getFrameStartStop(FramePredict):
    signStatus = False
    listofFrame = []
    alphabetFrame = []
    for indexFrame, frame_value in enumerate(FramePredict):

        if signStatus == False:   #transition mode
            if frame_value == 0:
                continue
            else:
                alphabetFrame = []
                alphabetFrame += [indexFrame+1]  # video frame start with index 1
                signStatus = True
        else:                      #Alphabet mode
            if frame_value == 0:
                signStatus = False
                listofFrame += [alphabetFrame]    ## add sequence alphabet frame
            else:
                alphabetFrame += [indexFrame+1]  # video frame start with index 1
                signStatus = True
    return listofFrame
        


def imageseqPredict(CNN_model, image_seq):
    CNN_model.eval()
    list_feature = []
    list_predict = []
    classOut = []
    eachAlphabet = {}
    tensorList= []
    for image_still in image_seq:
        # cv2.imshow("check Image", image_still)
        # cv2.waitKey(0)
        imageSelect = np.transpose(image_still, (2, 0, 1))
        inputImageDataset = torch.from_numpy(imageSelect)

        #unsqueeze for batch dimension
        inputImageDataset = inputImageDataset.unsqueeze(0)
        inputImageDataset = inputImageDataset.to(device=device, dtype=torch.float)
        # print(torch.sum(inputImageDataset))

        output_pred = CNN_model(inputImageDataset)
        feature_pred = CNN_model.extractFeature(inputImageDataset)

        list_feature += [feature_pred.cpu().detach().numpy()]
        list_predict += [output_pred.cpu().detach().numpy()]
        _, topi = output_pred.topk(1)
        classOut += [topi.reshape(-1)]
        tensorList += [topi.reshape(-1).item() - 2]
    print(tensorList)
    eachAlphabet["Feature_seq"] = np.asarray(list_feature,dtype=np.float)
    eachAlphabet["Predict_seq"] = np.asarray(list_predict,dtype=np.float)
    eachAlphabet["class_seq"] = np.asarray(classOut, dtype=np.int8)

    return(eachAlphabet)



def readThaiConfig():
    dictIndexSignThai = {}
    dictIndexCharThai = {}
    CharMaptoSign_file = "ThaiindexToSign.txt"
    with codecs.open(CharMaptoSign_file, "r", "utf-8") as fileThai:
        for idxChar,charThailine in enumerate(fileThai):
            charThai,signnumber = charThailine.splitlines()[0].split('\t')
            dictIndexSignThai[signnumber] = idxChar+1
            dictIndexCharThai[idxChar + 1] = charThai
    return dictIndexCharThai, dictIndexSignThai
    

def ConvertCharThai(signCode):
    charReturn = "-"
    if signCode in signThaiDict:    # for name, age in dictionary.iteritems():  (for Python 2.x)
        charReturn = charThaiDict[signThaiDict[signCode]]
    
    return charReturn


def LongestFirstConvert(signCodes):
    signCodes = list(np.array(signCodes,dtype=np.int))
    subSignCode = ",".join(str(v) for v in signCodes)
    charThai = ConvertCharThai(subSignCode)
    
    return charThai


def translator(seq_Imagecode):
    ### if seq_sequence morethan 5 images sign seqence that is sign
    ## else sign noise analyse from plot_boxplot_time.R graph 
    dictImage = {}
    indexcode = 0
    index_Imagestart = 2
    previous_imageCode = seq_Imagecode[0] - index_Imagestart
    for imageCode in seq_Imagecode:
        ## reimage Code 
        imageCode = imageCode-index_Imagestart
        if imageCode != previous_imageCode:
            indexcode += 1
        if not indexcode in dictImage:
            dictImage[indexcode] = [imageCode]
        else:
            dictImage[indexcode] +=[imageCode]
        
        previous_imageCode = imageCode
    
    ### predict Section get key dict with condition
    list_signCode = []
    for (key, value) in dictImage.items():
        # Check if key is even then add pair to new dictionary
        if len(dictImage[key]) > 5 :
            list_signCode += [int(dictImage[key][0])]
    if list_signCode == []:
        list_signCode +=[0]
    return list_signCode



import argparse

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="A number of Hidden size default is 256"
    )
    parser.add_argument(
        "--emb_dimension",
        type=int,
        default=25,
        help="A number of emb_dimension default is 2"
    )

    parser.add_argument(
        "--folderModel",
        default="LSTM_CNN_Feature",
        help="A number of emb_dimension default is 2"
    )
    parser.add_argument(
        "--rep",
        default="1",
        help="A number of rep"
    )

    parser.add_argument(
        "--MethodCut",
        default="Test_Distance",
        help="The method cutoff name"

    )
    args = parser.parse_args()

    MAX_LENGTH = 100

    emb_dimension = args.emb_dimension
    hidden_size = args.hidden_size
    model_folder = args.folderModel
    method_cutAlphabet= args.MethodCut
    rep = args.rep

        
    Annotation_frame = 'Annotation_sign_frame.csv'
    VideoPath = "E://Video_rawFileHandsign\Extract_BindHead_2grams"
    alphabetFrame_Path = "./Evaluate/{}/".format(method_cutAlphabet)


    annotationFile_data = pd.read_csv("./{}".format(Annotation_frame))
    annotationFile_test_index = pd.read_csv("./Sign_test_index.txt", names=['indexTest'], header=None)
    index_sign_test = annotationFile_test_index["indexTest"]-1
    # annotation_file_list = annotationFile_data['File Name']
    annotaion_file_list_train = annotationFile_data.iloc[~annotationFile_data.index.isin(index_sign_test)]
    annotaion_file_list_test = annotationFile_data.iloc[index_sign_test]
    print("Total testDataset {}".format(len(annotaion_file_list_test)))
    print("Total trainDataset {}".format(len(annotaion_file_list_train)))


    charThaiDict, signThaiDict = readThaiConfig()

    # ## load CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_model_CNN = "./Model_CNN/CNN_Sign_0.947_T_0.818"
    CNN_model = torch.load("{}".format(path_model_CNN))
    CNN_model = CNN_model.to(device)
    CNN_model.eval()
    print("Load CNN Weight")


    File_Annotationlist = []
    GT_List = []
    Predict_list = []
    wer_error = []
    Predict_list_SM = []
    wer_error_SM = []
    hcs_obj = hcs.CleanSequence()

    nrow_test = len(annotaion_file_list_test)

    for indexTest in range(nrow_test):
        rowAnnotationFile = annotaion_file_list_test.iloc[indexTest]
        annotationFile = rowAnnotationFile["File Name"]
        GT_Word = rowAnnotationFile["Index"]

        # print(annotaion_file_list_test['Index'])
        fileVideo_read = Path(os.path.join(VideoPath, "{}.mp4".format(annotationFile)))
        filename = Path(os.path.join(alphabetFrame_Path))

        if method_cutAlphabet in ["Test_Distance","Test_HeatMap"] :
            files = [f for f in filename.iterdir() if f.match("{}.csv".format(annotationFile))]
        else:
            files = [f for f in filename.iterdir() if f.match("{}_L0.csv".format(annotationFile))]

        print("{},{}".format(indexTest,files))

        df=pd.read_csv(files[0], sep=',')

        # print(df.values)
        values = df.values
        PredictAlphabet = values[:, 3]
        # print(PredictAlphabet)
        PredictAlphabet_smoot = np.array(PredictAlphabet, copy=True)

        # print(PredictAlphabet)
        PredictAlphabet_smoot = hcs_obj.cleanSequenceM1(PredictAlphabet_smoot)

        listofAlphabetFrames = getFrameStartStop(PredictAlphabet)
        listofAlphabetFrames_sm = getFrameStartStop(PredictAlphabet_smoot)
        listAlphabet_predict = []
        listAlphabet_predict_sm = []
        
        for listAlphabetFrame in listofAlphabetFrames:
            listAlphabetFrame = np.asarray(listAlphabetFrame)
            start = np.min(listAlphabetFrame)
            stop = np.max(listAlphabetFrame)
            listImage_frame = VideoProcess(fileVideo_read, start, stop)

            signSequenceAlphabet_feature = imageseqPredict(CNN_model,listImage_frame)
            class_seq = signSequenceAlphabet_feature['class_seq']
            # class_seq = [23, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
            # alphabet_Predict = testManage(encoder, class_seq)  ## default lr = 0.0001

            predict_sign_code = translator(class_seq)
            alphabet_pre = [LongestFirstConvert(predict_sign_code)]

            listAlphabet_predict += alphabet_pre
        print("smooted")
        for listAlphabetFrame in listofAlphabetFrames_sm:
            listAlphabetFrame = np.asarray(listAlphabetFrame)
            start = np.min(listAlphabetFrame)
            stop = np.max(listAlphabetFrame)
            listImage_frame = VideoProcess(fileVideo_read, start, stop)

            signSequenceAlphabet_feature = imageseqPredict(CNN_model,listImage_frame)
            class_seq = signSequenceAlphabet_feature['class_seq']
            # class_seq = [23, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
            predict_sign_code = translator(class_seq)
            alphabet_pre = [LongestFirstConvert(predict_sign_code)]
            listAlphabet_predict_sm += alphabet_pre

        word_pre = ''.join(str(v) for v in listAlphabet_predict)
        word_pre_sm = ''.join(str(v) for v in listAlphabet_predict_sm)
        word_pre = word_pre.replace("-", '')
        word_pre_sm = word_pre_sm.replace("-", '')
        error = wer(' '.join(GT_Word), ' '.join(word_pre))
        error_sm = wer(' '.join(GT_Word), ' '.join(word_pre_sm))

        print("GT = {} => Predict = {} error ={},predict_sm = {}, error_sm = {}".format(\
            GT_Word,word_pre,error,word_pre_sm,error_sm))

        File_Annotationlist += [annotationFile]
        GT_List += [GT_Word]
        Predict_list += [word_pre]
        wer_error += [error]
        Predict_list_SM += [word_pre_sm]
        wer_error_SM += [error_sm]

    GT = np.asarray(GT_List)
    Pre = np.asarray(Predict_list)
    Pre_sm = np.asarray(Predict_list_SM)
    FileTest = np.asarray(File_Annotationlist)
    wererror = np.asarray(wer_error)
    werError_Sm = np.asarray(wer_error_SM)
    resultFile = np.column_stack([FileTest,GT,Pre,wer_error,Pre_sm,werError_Sm])
    name_col = ['FileName','GT', 'Predict','error','Predict_sm','error_sm']
    dataFrame_resultFile = pd.DataFrame(resultFile, columns=name_col)

    evaluate_folder = "./Result_TestAllPipe/{}/".format(method_cutAlphabet)  ### force code output

    current_folder = os.path.join(evaluate_folder)
    if not os.path.exists(current_folder):
        os.mkdir(current_folder)

    dataFrame_resultFile.to_csv("./{}/baseLine.csv"\
        .format(evaluate_folder),
    index=False, encoding="utf-8-sig")











    # for datasetFile in annotaion_file_list_train.T.iteritems():
    #     datasetFile = datasetFile[1]
    #     fileName_pickle = datasetFile["File Name"]
    #     alphabet_gt = datasetFile["Index"]
    #     sign_gt = datasetFile["concatSign"]

    #     currentfile = "./{}/{}.pk".format(working_path, fileName_pickle)
    #     with open(currentfile, "rb") as wordDataFile:
    #         wordData = pickle.load(wordDataFile)
    #         detailAlphabet = wordData["alphabet_detail"]
    #         alphabet_pre = []
    #         sign_code_pre = []
    #         for eachAlphabet in detailAlphabet:
    #             seq_signSequence_predict = eachAlphabet["class_seq"]
    #             predict_sign_code = translator(seq_signSequence_predict)
    #             alphabet_pre += [LongestFirstConvert(predict_sign_code)]
    #             sign_code_pre += [",".join(str(v) for v in predict_sign_code)]

    #         word_predict = "".join(alphabet_pre)
    #         signCode_predict = "-".join(str(v) for v in sign_code_pre)

    #     fileVideo += [fileName_pickle]
    #     alphabet_GT += [alphabet_gt]
    #     sign_class_GT += [sign_gt]
    #     alphabet_predict += [word_predict]
    #     sign_class_predict += [signCode_predict]


    #         ### write predict to file
    # predict_data = np.column_stack([fileVideo,alphabet_GT, sign_class_GT,alphabet_predict,sign_class_predict])
    # data_predict_group_df = pd.DataFrame(predict_data, columns=['File Name', 'alphabet_GT', 'sign_GT', 'alphabet_predict', 'sign_predict'])
    # if not os.path.exists("ResultBaseLine"):
    #     os.mkdir("ResultBaseLine")
    # data_predict_group_df.to_csv("./ResultBaseLine/ResultBaseLine.csv",encoding='utf-8-sig',index=False)


