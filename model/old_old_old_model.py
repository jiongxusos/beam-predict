import pyBeamSim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# now, it's only gpu tracewin_data(todo)
# output_len - 1, so loss changes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def initHidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))

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

class Model2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_len, device):
        super(Model2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_len = output_len
        self.device = device
        # without encoder initial
        self.encoder = EncoderLstm(input_size, hidden_size).to(device)
        self.decoder = DecoderLstm(input_size, hidden_size, output_size).to(device)

    def forward(self, x, y_true_first_column):
        # x.shape: [batch_size, seq_len, input_size] (1000, 4, 1)
        hidden_state = self.encoder.initHidden(x.shape[0])
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
            output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_decoder)
            # output_decoder.shape: [batch_size, output_size]
            total.append(output_decoder)
            # input_decoder = output_decoder.unsqueeze(1)

        return torch.concat(total, dim=1)



