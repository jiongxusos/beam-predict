import pyBeamSim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# now, it's only gpu tracewin_data(todo)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    :tracewin_data input: 3-D 
    :model2 input: 3-D
    :final_output : 2-D (batch_size, seq_len) or (batch_size, seq_len - 1) 
    
"""

class EncoderLstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLstm, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        # embedded = self.embedding(x).view(1, 1, -1)
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def initHidden(self, batch_size, hidden_device):
        return (torch.zeros((1, batch_size, self.hidden_size), device=hidden_device),
                torch.zeros((1, batch_size, self.hidden_size), device=hidden_device))

# output_encoder shape: [batch_size, seq_len, hidden_size]
# hidden_encoder shape : [1, batch_size, hidden_size]

# output_encoder[:, -1, :] as input_decoder , hidden same as  , et seq_len = 1
# output_decoder  shape: [batch_size, seq_len, hidden_size]

class DecoderLstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # embedded = self.embedding(x).view(1, 1, -1)

        # x.shape~X[:, -1, :]: [batch_size, input_size]
        output, hidden = self.lstm(x.unsqueeze(1), hidden)
        # output_shape: [batch_size, seq_len=1, hidden_size]
        # hidden shape: [1, batch_size, hidden_size]
        output = self.fc(output.squeeze(1))
        # output_decoder  shape: [batch_size, output_size]
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

# take input_encoder[:, -1, :] as input_decoder

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_len, device):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_len = output_len
        self.device = device
        # without encoder initial
        self.encoder = EncoderLstm(input_size, hidden_size).to(device)
        self.decoder = DecoderLstm(input_size, hidden_size, output_size).to(device)

    def forward(self, x):
        # x.shape: [batch_size, seq_len, input_size] (1000, 4, 1)
        hidden_state = self.encoder.initHidden(x.shape[0], x.device)
        output_encoder, hidden_encoder = self.encoder(x, hidden_state)
        # (1000, 4, 30) ,(1, 1000, 30)

        hidden_decoder = hidden_encoder

        # # more changes can do
        input_decoder = x[:, -1, :]   # (1000,1)
        # second, take x_avg_0 as input
        #input_decoder = y_true_first_column

        total = []
        for i in range(self.output_len):
            # recurse for input_decoder
            input_decoder, hidden_decoder = self.decoder(input_decoder, hidden_decoder)
            # output_decoder.shape: [batch_size, output_size]
            total.append(input_decoder)
            # input_decoder = output_decoder.unsqueeze(1)

        return torch.concat(total, dim=1)


class Model2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_len):
        super(Model2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_len = output_len
        # without encoder initial
        self.encoder = EncoderLstm(input_size, hidden_size)
        self.decoder = DecoderLstm(input_size, hidden_size, output_size)

    def forward(self, x, y_true_first_column):
        # x.shape: [batch_size, seq_len, input_size] (1000, 4, 1)
        hidden_state = self.encoder.initHidden(x.shape[0], x.device)    # self.encoder.device????
        output_encoder, hidden_encoder = self.encoder(x, hidden_state)
        # (1000, 4, 30) ,(1, 1000, 30)

        hidden_decoder = hidden_encoder

        # # more changes can do
        # input_decoder = x[:, -1, :]   # (1000,1)
        # second, take x_avg_0 as input
        input_decoder = y_true_first_column

        total = []
        for i in range(self.output_len - 1):  # output_len - 1
            # loss changes as well
            # recurse for input_decoder
            input_decoder, hidden_decoder = self.decoder(input_decoder, hidden_decoder)
            # output_decoder.shape: [batch_size, output_size]
            total.append(input_decoder)
            # input_decoder = output_decoder.unsqueeze(1)

        return torch.concat(total, dim=1)


class MLP(nn.Module):
    def __init__(self, input_size=8, hidden_size1=64, hidden_size2=32,  output_size=4, dropout=0.1):
        super(MLP, self).__init__()
        self.input_size = input_size  # x.shape[1]
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        y = self.fc3(x)
        return y