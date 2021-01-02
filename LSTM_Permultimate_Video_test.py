import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import time
import random
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import LSTM_Class
import DatapairClass as Dpair
# import BiencoderClass_in
from pathlib import Path
import csv
import cv2
import HandProcess as hp
from VGG16ClassCNN import ConvNetExtract
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
        
def indexesFromSentence(lang, sentence):
    listwordSentence = [int(lang.word2token[str(word)]) for word in sentence]
    return np.asarray(listwordSentence)

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes = np.append(indexes,EOS_token)

    return indexes

def test(encoder,input_tensor):

    encoder.eval()
    (encoder_hidden, cell_hidden) = encoder.initHidden()

    #check point here
    (encoder_hidden, cell_hidden) = encoder.initHidden()    
    encoder_output, encoder_hidden,outFc,out_softmax = encoder(input_tensor, (encoder_hidden.detach(), cell_hidden.detach()))

    # print(encoder_output)
    topv, topi = out_softmax.topk(1)

    return (topi)

def testManage(encoder, inputList):
    seconds = time.time()

    #### convert Pair to tensor Dump programming
    input_tensor = torch.tensor(inputList, dtype=torch.float32, device=device)

    ### save Check point Here
    predict = test(encoder,input_tensor)

    Pre = [output_lang.token2word[predict.item()]]
    return Pre


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
        default="LSTM_CNN_Permultimate",
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

    SOS_token = 2
    EOS_token = 1
    PAD_token = 0
    batch_size = 1

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_forcing_ratio = 0.5

    Annotation_frame = 'Annotation_sign_frame.csv'
    
    VideoPath = "E://Video_rawFileHandsign\Extract_BindHead_2grams"

    annotationFile_data = pd.read_csv("./{}".format(Annotation_frame))
    annotationFile_test_index = pd.read_csv("./Sign_test_index.txt", names=['indexTest'], header=None)
    index_sign_test = annotationFile_test_index["indexTest"]-1
    # annotation_file_list = annotationFile_data['File Name']
    annotaion_file_list_train = annotationFile_data.iloc[~annotationFile_data.index.isin(index_sign_test)]
    annotaion_file_list_test = annotationFile_data.iloc[index_sign_test]

    dataset = Dpair.DatasetManagePair()
    input_lang, output_lang = dataset.prepareData()
    

    

    print("Total testDataset {}".format(len(annotaion_file_list_test)))
    print("Total trainDataset {}".format(len(annotaion_file_list_train)))

    # ## load CNN model
    path_model_CNN = "./Model_CNN/CNN_Sign_0.947_T_0.818"
    CNN_model = torch.load("{}".format(path_model_CNN))
    CNN_model = CNN_model.to(device)
    CNN_model.eval()
    print("Load CNN Weight")

    encoder = torch.load("./{}/{}/LSTM_CNN_Permultimate_Lowest_Loss_validation_emb_dimension_{}_hidden_{}"\
            .format(model_folder,rep,emb_dimension,hidden_size))
    
    encoder.to(device)
    encoder.eval()
    print("Loaded LSTM Weight")


    ## path of Extraction Folder 
    alphabetFrame_Path = "./Distance_centroidData/{}/".format(method_cutAlphabet)

    ### Alphabet extraction from Phase 2
    nrow_test = len(annotaion_file_list_test)
    File_Annotationlist = []
    GT_List = []
    Predict_list = []
    wer_error = []
    Predict_list_SM = []
    wer_error_SM = []
    hcs_obj = hcs.CleanSequence()
    
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
            class_seq = signSequenceAlphabet_feature['Predict_seq']
            class_seq = np.transpose(class_seq,(1,0,2))
            # class_seq = [23, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
            alphabet_Predict = testManage(encoder, class_seq)  ## default lr = 0.0001
            listAlphabet_predict += alphabet_Predict
        print("smooted")
        for listAlphabetFrame in listofAlphabetFrames_sm:
            listAlphabetFrame = np.asarray(listAlphabetFrame)
            start = np.min(listAlphabetFrame)
            stop = np.max(listAlphabetFrame)
            listImage_frame = VideoProcess(fileVideo_read, start, stop)

            signSequenceAlphabet_feature = imageseqPredict(CNN_model,listImage_frame)
            class_seq = signSequenceAlphabet_feature['Predict_seq']
            # class_seq = [23, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
            class_seq = np.transpose(class_seq,(1,0,2))
            alphabet_Predict = testManage(encoder, class_seq)  ## default lr = 0.0001
            listAlphabet_predict_sm += alphabet_Predict

        word_pre = ''.join(str(v) for v in listAlphabet_predict)
        word_pre_sm = ''.join(str(v) for v in listAlphabet_predict_sm)
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

    evaluate_folder = "./Result_TestAllPipe/{}/{}".format(method_cutAlphabet,rep)  ### force code output

    
    if not os.path.exists("./Result_TestAllPipe/{}".format(method_cutAlphabet)):
        os.mkdir("./Result_TestAllPipe/{}".format(method_cutAlphabet))
    
    current_folder = os.path.join(evaluate_folder)
    if not os.path.exists(current_folder):
        os.mkdir(current_folder)

    dataFrame_resultFile.to_csv("./{}/LSTM_CNN_Permultimate_result_emb_dimension_{}_hidden_{}.csv"\
        .format(evaluate_folder,emb_dimension, hidden_size),
    index=False, encoding="utf-8-sig")