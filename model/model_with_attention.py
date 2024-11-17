import pyBeamSim
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import r2_score

from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for now , output_len = 1(todo)
# generate a lstm-lstm encoder-decoder with two layers, two directions, teacher-forcing and attention


# encoder must be bidirectional
# decoder must be unidirectional
class Encoder(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, decoder_hidden_size, num_layers=2, num_directions=2, dropout=0):
        """
        input_size: 1
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.num_directions = num_directions
        self.bi = True if num_directions == 2 else False
        # deal with hidden with multi layers and directions
        if self.bi:
            # self.fc = nn.Linear(encoder_hidden_size * 2, num_layers * decoder_hidden_size)
            self.fc = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
            self.fc2 = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        self.lstm = nn.GRU(input_size, encoder_hidden_size,
                           batch_first=True, num_layers=self.num_layers, bidirectional=self.bi, dropout=dropout)

    # include init_hidden into itself
    def init_hidden(self, hidden_device, batch_size):
        if isinstance(self.lstm, nn.LSTM):
            return (torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                                device=hidden_device),
                    torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                                device=hidden_device))
        elif isinstance(self.lstm, nn.GRU):
            return torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                               device=hidden_device)

    def forward(self, x):
        """
        input : (batch_size, seq_len, input_size)
        input_hidden : (num_layer * num_direction, batch_size, encoder_hidden_size) * 2
        output : (batch_size, seq_len, encoder_hidden_size * num_directions)

        input_hidden :  (num_layer * num_direction, batch_size, encoder_hidden_size) * 2      , same as input_hidden
        output_hidden :  (batch_size, decoder_hidden_size),             repeat in seq2seq(todo)
        """
        batch_size = x.shape[0]
        hidden = self.init_hidden(x.device, batch_size)
        output, hidden = self.lstm(x, hidden)
        if self.bi:
            # the last layer forward and backward
            # hidden = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            hidden = torch.stack((self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)),
                                 self.fc2(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))), dim=0)
        else:
            hidden = hidden[-1]  # the last layer
        return output, hidden


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, bi_decoder=True):
        super(Attention, self).__init__()
        self.attention = nn.Linear(encoder_hidden_size * 2 + decoder_hidden_size, decoder_hidden_size)
        self.relu = nn.ReLU()
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(self, output_encoder, last_hidden_decoder):
        """
        :output_encoder: (batch_size, seq_len, encoder_hidden_size * 2)
        :last_hidden_decoder: (batch_size, decoder_hidden_size) * 2
        :return:context: (batch_size, 1, encoder_hidden_size * 2)
        """
        batch_size = output_encoder.shape[0]
        seq_length = output_encoder.shape[1]
        # extend hidden_decoder to the length same as output_decoder
        last_hidden_decoder = last_hidden_decoder.unsqueeze(1).repeat(1, seq_length, 1)
        # calculate weights
        energy = torch.tanh(self.attention(torch.cat((output_encoder, last_hidden_decoder), dim=2)))  # similar
        # energy: (batch_size, seq_len, decoder_hidden_size)
        attention = self.v(energy).squeeze(2)   # (batch_size, seq_len)
        attention_weights = torch.softmax(attention, dim=1)  # (batch_size, seq_len)
        context = torch.bmm(attention_weights.unsqueeze(1), output_encoder)
        return context


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size,
                 num_layers=2,
                 num_directions=2,
                 attention=None,
                 dropout=0):
        super(Decoder, self).__init__()
        """
        input_size: (x_avg, x_sig) : 2 dims
        output_size: same as input_size
        """
        self.input_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.bi = True if num_directions == 2 else False

        self.lstm = nn.GRU(output_size + hidden_size * (2 if attention else 1), hidden_size,
                            batch_first=True,
                            num_layers=self.num_layers,
                            bidirectional=self.bi, dropout=dropout)

        self.fc = nn.Linear(hidden_size * num_directions, output_size)
        self.attention = attention
        self.init_weights()

    def init_hidden(self, hidden_device, batch_size):
        if isinstance(self.lstm, nn.LSTM):
            return (torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                                device=hidden_device),
                    torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                                device=hidden_device))
        elif isinstance(self.lstm, nn.GRU):
            return torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                               device=hidden_device)

    def init_weights(self):
        # 初始化 GRU 权重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)  # in-to-hidden Xavier
            elif 'weight_hh' in name:
                init.kaiming_normal_(param)  # hidden-to-hidden Kaiming
            elif 'bias' in name:
                init.constant_(param, 0)  # set bias to 0

    def forward(self, tgt, output_encoder, hidden):
        # transport every shape_change into the tracewin_data
        """
        :param tgt: (batch_size, output_size=1)
        :param output_encoder: (batch_size, src_len, encoder_hidden_size * 2)
        :param hidden: first and after: (num_layers * num_directions, batch_size, decoder_hidden_size)

        :return:
        output_decoder: (batch_size, output_size=1)
        hidden: (num_layers * num_directions, batch_size, decoder_hidden_size)
        """

        input_decoder = tgt.unsqueeze(1)   # (batch_size, tgt_len = 1, output_size=1)
        if self.attention is not None:
            # hidden_decoder[-1]: the last of (num_layers * num_directions, batch_size, decoder_hidden_size)
            context = self.attention(output_encoder, hidden[-1])  # (batch_size, 1, encoder_hidden_size*2)
            input_decoder = torch.cat((tgt.unsqueeze(1), context), dim=2)
            # input_decoder: (batch_size, tgt_len = 1, output_size + encoder_hidden_size*2)

        # input_hidden: (num_layers * num_directions, batch_size, decoder_hidden_size)
        output, hidden = self.lstm(input_decoder, hidden)
        # output: (batch_size, tgt_len = 1, decoder_hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch_size, decoder_hidden_size)
        output_decoder = self.fc(output).squeeze(1)
        # output_decoder: (batch_size, output_size=1)
        return output_decoder, hidden

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, decoder_hidden_size, output_size, model_device,
                 num_layers=2, num_directions=1, teacher_forcing=0.5, dropout=0):
        super(Encoder_Decoder, self).__init__()
        self.input_size = input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.device = model_device
        self.num_layers = num_layers
        self.tf = teacher_forcing

        self.encoder = Encoder(input_size, encoder_hidden_size, decoder_hidden_size, num_layers, dropout=dropout)
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)
        self.decoder = Decoder(output_size, decoder_hidden_size, num_layers, num_directions,
                               self.attention, dropout=dropout)

    def forward(self, src, tgt, use_tf=False):
        """
        src_length: [quad1_len, quad1_gradient, dipole1_k1, ...]
        tgt_length: [x_avg0, x_avg1, x_avg2, ...]
        output_size: [x_avg0, y_sig0, loss, ...]

        :param use_tf:
        :param src: (batch_size, src_length, input_size=1)
        :param tgt: (batch_size, tgt_length + 1, output_size=1)
        :return:
        """
        batch_size = src.shape[0]
        tgt_length = tgt.shape[1] - 1
        outputs = torch.empty((batch_size, tgt_length, self.output_size), device=self.device)
        outputs_list = []
        output_encoder, hidden = self.encoder(src)
        # output_encoder: (batch_size, seq_len, encoder_hidden_size * 2)
        # hidden_encoder: (batch_size, decoder_hidden_size)  , after repeat as the first hidden_input of decoder

        # hidden = hidden.repeat(self.decoder.num_layers * self.decoder.num_directions, 1, 1)
        # hidden = hidden.reshape(self.num_layers, batch_size, self.decoder_hidden_size)
        # hidden: (num_layers * num_directions, batch_size, decoder_hidden_size)
        tgt_t = tgt[:, 0, :]   # (batch_size, output_size)
        for t in range(1, tgt_length + 1):
            output_decoder, hidden = self.decoder(tgt_t, output_encoder, hidden)
            # output_decoder: (batch_size, output_size=1)
            # hidden: (batch_size, decoder_hidden_size)
            outputs_list.append(output_decoder)
            # recurse or teacher-forcing
            if use_tf and random.random() < self.tf:
                tgt_t = tgt[:, t, :]
            else:
                tgt_t = output_decoder
        return torch.stack(outputs_list, dim=1)   # (batch_size, tgt_length - 1, output_size)


def count_all_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)




    # add teach_forcing
    """
    def forward(self, x, y):
        # x : 3D (batch_size, param_dim, input_size=1 for now)    param:(quad1_len, ..., quad2_gradient)
        # y : 3D (batch_size, seq_len, input_size=1 for now)   seq:(x_avg0, x_avg1...)
        output_encoder, hidden_encoder = self.encoder(x)
        hidden_decoder = hidden_encoder

        output_decoder, hidden_decoder = self.decoder(y, hidden_decoder)
        # output_decoder : (batch_size, seq_len, output_size=1 for now)
        return output_decoder
    """

class Encoder_Decoder2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_len, model_device,
                 num_layers=2, num_directions=2):
        super(Encoder_Decoder2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_len = output_len
        self.device = model_device

        self.losses = []
        self.mape = []
        self.bi = True if num_directions == 2 else False

        self.encoder = Encoder(input_size, hidden_size, num_layers, num_directions)
        self.decoder = Decoder(input_size, hidden_size, output_size, num_layers, num_directions)

    def train_model(self, train_loader, num_epochs, learning_rate, target_len,
                    method='teach_forcing', criterion=nn.MSELoss(), teaching_forcing=0.5):

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            if epoch == 0:
                print(f'the training is using {device}')
            total_loss = 0
            self.train()
            for X, y in train_loader:
                X = X.to(self.device)  # (batch_size, param_dim, input_size=1)
                y = y.to(self.device)   # (batch_size, seq_len=4)
                y_pred = torch.empty((X.shape[0], target_len, self.output_size)).to(self.device)
                # if epoch == 0:  target_len = y.shape[1] - 1

                # take 0~5 columns for predict , 1~6 columns for cal loss
                output_encoder, hidden_encoder = self.encoder(X)
                hidden_decoder = hidden_encoder
                if method == 'recurse':
                    output_decoder, hidden_decoder = self.decoder(y[:, 0].reshape(-1, 1, 1), hidden_decoder)
                    y_pred[:, 0, :] = output_decoder.squeeze(1)
                    # use the last y_pred as the next input of decoder
                    for t in range(target_len - 1):       # 0, 1, 2 ,   target_len - 1 = 3
                        input_decoder = output_decoder
                        output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_decoder)
                        y_pred[:, t + 1, :] = output_decoder.squeeze(1)

                elif method == 'teach_forcing':
                    # the first column, time-step
                    output_decoder, hidden_decoder = self.decoder(y[:, 0].reshape(-1, 1, 1), hidden_decoder) # add two dims
                    y_pred[:, 0, :] = output_decoder.squeeze(1)
                    # torch.autograd.set_detect_anomaly(True)

                    for t in range(target_len - 1):
                        if random.random() < teaching_forcing:
                            # use the true y
                            input_decoder = y[:, t + 1].reshape(-1, 1, 1)    # remove seq_len=1 dim?
                            output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_decoder)
                            y_pred[:, t + 1, :] = output_decoder.squeeze(1)
                        else:
                            # use the last y_pred -----recurse
                            input_decoder = output_decoder
                            """problems in output_decoder, _ = self,decoder()"""
                            output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_decoder)
                            y_pred[:, t + 1, :] = output_decoder.squeeze(1)
                            # y_pred[:, t+1, :] = self(X, y_pred[:, t, :].unsqueeze(1)).squeeze(1)

                loss = criterion(y_pred, y[:, 1:].unsqueeze(2))
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # self.losses.append(total_loss.cpu().detach().numpy())
            # avg_loss = total_loss / len(train_loader)
            print(f'epoch{epoch + 1}: loss: {total_loss:.10f}')

    def get_val_mape(self, val_loader, target_len):
        """with torch_no_grad()"""
        self.eval()
        total_mape = 0
        total_sample = 0
        # recurse predict
        with torch.no_grad():
            for X, y in val_loader:
                batch_size = X.shape[0]
                X = X.to(self.device)
                y = y.to(self.device)
                y_pred = self.predict_recurse(X, y, target_len)
                mape_batch = get_mape(y[:, 1:].unsqueeze(2), y_pred)
                total_mape += mape_batch * batch_size
                total_sample += batch_size
        return (total_mape / total_sample).cpu().detach().numpy()  # or item()

    def predict_recurse(self, X, y, target_len):
        y_pred = torch.empty((X.shape[0], target_len, self.output_size)).to(self.device)
        output_encoder, hidden_encoder = self.encoder(X)
        hidden_decoder = hidden_encoder
        output_decoder, hidden_decoder = self.decoder(y[:, 0].reshape(-1, 1, 1), hidden_decoder)
        y_pred[:, 0, :] = output_decoder.squeeze(1)
        # use the last y_pred as the next input of decoder
        for t in range(target_len - 1):  # 0, 1, 2 ,   target_len - 1 = 3
            input_decoder = output_decoder
            output_decoder, hidden_decoder = self.decoder(input_decoder, hidden_decoder)
            y_pred[:, t + 1, :] = output_decoder.squeeze(1)
        return y_pred

    # 2dims for now
    def get_r2_score(self, val_loader):
        self.eval()
        total_r2 = 0
        for X, y in val_loader:
            batch_size = X.shape[0]
            X = X.to(self.device)
            y = y.to(self.device)
            output_encoder, hidden_encoder = self.encoder(X)
            hidden_decoder = hidden_encoder
            output_decoder, hidden_decoder = self.decoder(y.unsqueeze(2), hidden_decoder)
            y_pred = output_decoder[:, :-1]
            r2_batch = get_r2_score(y[:, 1:].unsqueeze(2).cpu(), y_pred.cpu())
            total_r2 += r2_batch * batch_size
        return total_r2 / len(val_loader)

    def predict_cpu(self, input:torch.Tensor, target_len):
        # input.shape:(7 + 1, )   1D tensor or 2D
        # self.to('cpu')
        pred = []

        if input.dim() == 1:
            x = input[:-1]
            y = input[-1]
            output_encoder, hidden_encoder = self.encoder(x.reshape(1, -1, 1))
            hidden_decoder = hidden_encoder
            for i in range(target_len):
                output_decoder, hidden_decoder = self.decoder(y.reshape(1, 1, 1), hidden_decoder)   # (1, 1, 1)
                pred.append(output_decoder.item())
            return pred

        if input.dim() == 2:
            x = input[:, :-1]
            y = input[:, -1].unsqueeze(1)
            output_encoder, hidden_encoder = self.encoder(x.unsqueeze(2))
            hidden_decoder = hidden_encoder
            for i in range(target_len):
                output_decoder, hidden_decoder = self.decoder(y.unsqueeze(2), hidden_decoder)
                pred.append(output_decoder)
            return torch.concat(pred, dim=1)

        return 0



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
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        y = self.fc3(x)
        return y



# use mean absolute percent error as criterion
def get_mape(tensor_true, tensor_pred):
    return torch.mean(torch.abs((tensor_pred - tensor_true) / tensor_true)) * 100


def get_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)



