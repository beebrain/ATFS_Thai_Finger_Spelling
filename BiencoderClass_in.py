import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import random

class Biencoder(nn.Module):
    def __init__(self, input_size, hidden_size,emb_dimension,batch_size = 1,num_layer=1):
        super(Biencoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.embedding = nn.Embedding(input_size, emb_dimension,padding_idx=0)  ## (input_size, embedded_vector_size)
        self.embedding.to(self.device)
        self.gru = nn.LSTM(emb_dimension, hidden_size,num_layers=self.num_layer, batch_first=True, bidirectional=False)  ## (inputSize, hidden_size)
        self.gru.to(self.device)
        self.batch_size = batch_size
        self.emb_dimension = emb_dimension
        


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, 1, self.emb_dimension)  #(1,1,-1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layer, self.batch_size, self.hidden_size, device=self.device)
