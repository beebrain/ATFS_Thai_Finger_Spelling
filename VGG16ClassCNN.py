import torch 
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import random
import torch.optim as optim
#########################
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import RandomHandClass
import BatchHelper
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvNetExtract(nn.Module):
    def __init__(self, output_size, dropout_p=0.3):
        super(ConvNetExtract, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2)

        self.adaptor = nn.Linear(8 * 8 * 128, 1024)
        self.adaptor2 = nn.Linear(1024, output_size)

        self.adaptor = self.adaptor.to(self.device)
        self.adaptor2 = self.adaptor2.to(self.device)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        outputFeature = self.extractFeature(input)
        outputFeature = torch.relu(self.adaptor(outputFeature))
        outputFeature = self.dropout(outputFeature)

        outputFeature = torch.relu(self.adaptor2(outputFeature))
        class_output_log = torch.log_softmax(outputFeature,dim=1)
        return class_output_log
    
    def extractFeature(self, input):
        in_size = input.size(0)
        features = self.conv1(input)
        features = torch.relu(features)
        features = self.maxpool(features)
        features = self.conv2(features)
        features = torch.relu(features)
        features = self.maxpool(features)
        features = self.conv3(features)
        features = torch.relu(features)
        features = self.maxpool(features)        

        outputFeature = features.view(in_size, -1)  # flatten
        return outputFeature

class Net(nn.Module):

    def __init__(self,output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=1)
        self.mp = nn.MaxPool2d(2)
        ## what is number value ??
        self.fc = nn.Linear(30*30*128, output_size)   #########??

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(torch.relu(self.conv1(x)))
        x = self.mp(torch.relu(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        x = torch.log_softmax(x,dim=1)
        return x



if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 30
    num_classes = 28
    batch_size = 100
    learning_rate = 0.0001

    imageSize = 64

    models = ConvNetExtract(num_classes)
    models = models.to(device)
    rHC = RandomHandClass.RandomHandClass()
    rHC.readAllDatabase("./ClassImage_train")
    batchHelper_train = BatchHelper.BatchHelp(rHC.ImageAll,rHC.labelAll)
    totalSample_train = len(batchHelper_train._label)

    rHC_val = RandomHandClass.RandomHandClass()
    rHC_val.readAllDatabase("./ClassImage_val")
    batchHelper_val = BatchHelper.BatchHelp(rHC_val.ImageAll,rHC_val.labelAll)
    totalSample_val = len(batchHelper_val._label)

    rHC_test = RandomHandClass.RandomHandClass()
    rHC_test.readAllDatabase("./ClassImage_test")
    batchHelper_test = BatchHelper.BatchHelp(rHC_test.ImageAll,rHC_test.labelAll)
    totalSample_test = len(batchHelper_test._label)


    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(models.parameters(), lr=0.0001)  # apply optimizer to model 
    ACC = 0
    ACC_Max_val = 0
    Average_ACC_val = 0
    ACC_Max_test = 0
    Average_ACC_test = 0


    print("totalTime per epoach {}".format(int(totalSample_train/batch_size)))
    for y in range(num_epochs):
        models.train()
        ACC = 0
        batchHelper_train.resetIndex()
        while batchHelper_train._epochs_completed == 0:
            input_image,labelImage = batchHelper_train.next_batch(batch_size,True)
        # for i in range(int(totalSample/batch_size)):
            loss = 0
            optimizer.zero_grad()
            inputImageDataset = torch.from_numpy(input_image)
            inputImageDataset = inputImageDataset.to(device=device, dtype=torch.float)
            # print(torch.sum(inputImageDataset))
            target_output = torch.from_numpy(labelImage.astype(int))
            target_output = target_output.to(device=device, dtype=torch.long)
            output_pred = models(inputImageDataset)

            loss = criterion(output_pred, target_output)
            loss.backward()
            optimizer.step()
            topv, topi = output_pred.topk(1)
            check_target = (topi.reshape(-1)==target_output)
            ACC += check_target.float().sum()      
            if batchHelper_train._index_in_epoch % batch_size == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.9f} current ACC {} '\
                    .format(y, batchHelper_train._index_in_epoch, totalSample_train,\
                    (batchHelper_train._index_in_epoch * 100.) / totalSample_train, \
                        loss.item(),(check_target.float().sum()/batch_size)*100))
        
        

        ############# validation #####################
        batchHelper_val.resetIndex()
        models.eval()
        ACC=0
        coutBatch = 0
        while batchHelper_val._epochs_completed == 0:
            input_image,labelImage = batchHelper_val.next_batch(batch_size,True)
            inputImageDataset = torch.from_numpy(input_image)
            inputImageDataset = inputImageDataset.to(device=device, dtype=torch.float)

            target_output = torch.from_numpy(labelImage.astype(int))
            target_output = target_output.to(device=device, dtype=torch.long)
            output_pred = models(inputImageDataset)
            topv, topi = output_pred.topk(1)
            check_target = (topi.reshape(-1)==target_output)
            ACC += check_target.float().sum()
            coutBatch +=1

        Average_ACC_val=(ACC / (coutBatch*batch_size))
        print("Validation ACC average {}".format(Average_ACC_val))
        if ACC_Max_val < Average_ACC_val:
            print("ACC val Update")
            # torch.save(models, "./Model_CNN/CNN_SignModel_{:.3f}".format(Average_ACC_val))
            ACC_Max_val = Average_ACC_val


            ############### test ##################
            batchHelper_test.resetIndex()
            coutBatch = 0
            ACC = 0
            while batchHelper_test._epochs_completed == 0:
                input_image,labelImage = batchHelper_test.next_batch(batch_size,True)
                inputImageDataset = torch.from_numpy(input_image)
                inputImageDataset = inputImageDataset.to(device=device, dtype=torch.float)

                target_output = torch.from_numpy(labelImage.astype(int))
                target_output = target_output.to(device=device, dtype=torch.long)
                output_pred = models(inputImageDataset)
                topv, topi = output_pred.topk(1)
                check_target = (topi.reshape(-1)==target_output)
                ACC += check_target.float().sum()
                coutBatch += 1
            
            Average_ACC_test=(ACC / (coutBatch*batch_size))
            print("Test ACC average {}".format(Average_ACC_test))
            print("SaveModel")
            torch.save(models, "./Model_CNN/CNN_Sign_{:.3f}_T_{:.3f}".format(Average_ACC_val,Average_ACC_test))
            ACC_Max_test = Average_ACC_test
            print("Save Completed")


