import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM_Class(nn.Module):
    def __init__(self, input_size, hidden_size,emb_dimension,outputdim = 45,batch_size = 1,num_layer=1):
        super(LSTM_Class, self).__init__()

        if torch.cuda.is_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        self.batch_size = batch_size
        self.emb_dimension = emb_dimension
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.embedding = nn.Embedding(input_size, emb_dimension, padding_idx=0)  ## (input_size, embedded_vector_size)
        self.embedding.to(self.device)

        ## final Adaptor
        self.fc = nn.Linear(hidden_size, outputdim)


        self.lstm = nn.LSTM(emb_dimension, hidden_size,num_layers=self.num_layer, batch_first=True, bidirectional=False)  ## (inputSize, hidden_size)
        self.lstm.to(self.device)


        
    def forward(self, input, hidden):

        embedded = self.embedding(input)
        embedded = embedded.view(1, -1, self.emb_dimension)  #(1,-1,emb_dimension)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        # print(output[:, -1,:])   ### get last output
        out = self.fc(output[:, -1,:])
        
        ### please use log softmax compat with loss 
        ### I use NLLLoss() function
        out_softmax = torch.log_softmax(out,dim=1)
        
        return output, hidden, out,out_softmax

    def initHidden(self):
        return (torch.zeros(self.num_layer, self.batch_size, self.hidden_size, device=self.device),
        torch.zeros(self.num_layer, self.batch_size, self.hidden_size, device=self.device))




class LSTM_Onehot(nn.Module):
    def __init__(self, input_size, hidden_size,outputdim = 45,batch_size = 1,num_layer=1):
        super(LSTM_Onehot, self).__init__()

        if torch.cuda.is_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        self.batch_size = batch_size
        self.input_dim = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        # self.embedding = nn.Embedding(input_size, emb_dimension, padding_idx=0)  ## (input_size, embedded_vector_size)
        # self.embedding.to(self.device)

        ## final Adaptor
        self.fc = nn.Linear(hidden_size, outputdim)


        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=self.num_layer, batch_first=True, bidirectional=False)  ## (inputSize, hidden_size)
        self.lstm.to(self.device)


        
    def forward(self, input, hidden):

        # output shape (1,-1,input_dim)

        output = torch.zeros(len(input), self.input_dim)
        a = input.unsqueeze(1).to('cpu')
        output = output.scatter_(1, a, 1.)
        output = output.view(1, -1, self.input_dim)  #(1,-1,emb_dimension)
        output = output.to(self.device)

        output, hidden = self.lstm(output, hidden)
        # print(output[:, -1,:])   ### get last output
        out = self.fc(output[:, -1,:])
        
        ### please use log softmax compat with loss 
        ### I use NLLLoss() function
        out_softmax = torch.log_softmax(out,dim=1)
        
        return output, hidden, out,out_softmax

    def initHidden(self):
        return (torch.zeros(self.num_layer, self.batch_size, self.hidden_size, device=self.device),
        torch.zeros(self.num_layer, self.batch_size, self.hidden_size, device=self.device))



class LSTM_Feature(nn.Module):
    def __init__(self, input_size, hidden_size,feature_dim,outputdim = 45,batch_size = 1,num_layer=1):
        super(LSTM_Feature, self).__init__()

        if torch.cuda.is_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        ## final Adaptor
        self.lstm = nn.LSTM(feature_dim, hidden_size,num_layers=self.num_layer, batch_first=True, bidirectional=False)  ## (inputSize, hidden_size)
        self.lstm.to(self.device)

        self.fc = nn.Linear(hidden_size, outputdim)
        self.fc.to(self.device)


        
    def forward(self, input, hidden):

        output = input
        output, hidden = self.lstm(output, hidden)
        # print(output[:, -1,:])   ### get last output
        out = self.fc(output[:, -1,:])
        
        ### please use log softmax compat with loss 
        ### I use NLLLoss() function
        out_softmax = torch.log_softmax(out,dim=1)
        
        return output, hidden, out,out_softmax

    def initHidden(self):
        return (torch.zeros(self.num_layer, self.batch_size, self.hidden_size, device=self.device),
        torch.zeros(self.num_layer, self.batch_size, self.hidden_size, device=self.device))




class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        if torch.cuda.is_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
