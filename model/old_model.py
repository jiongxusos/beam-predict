import pyBeamSim
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for now , output_len = 1(todo)
# generate a lstm-lstm encoder-decoder with two layers, two directions, teacher-forcing and attention


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_directions=2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.bi = True if num_directions == 2 else False
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bi)

    # include init_hidden into itself
    def init_hidden(self, hidden_device, batch_size):
        return (torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                            device=hidden_device),
                torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                            device=hidden_device))

    def forward(self, x):
        """
        input : (batch_size, seq_len, input_size)
        input_hidden : (num_layer * num_direction, batch_size, hidden_size) * 2
        output : (batch_size, seq_len, hidden_size * num_directions)
        output_hidden :  (num_layer * num_direction, batch_size, hidden_size) * 2      , same as input_hidden
        """
        batch_size = x.shape[0]
        hidden = self.init_hidden(x.device, batch_size)
        output, hidden = self.lstm(x, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_directions=2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.bi = True if num_directions == 2 else False

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bi)

        self.fc = nn.Linear(hidden_size * num_directions, output_size)

    def init_hidden(self, hidden_device, batch_size):
        return (torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                            device=hidden_device),
                torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size),
                            device=hidden_device))

    def forward(self, x, hidden):
        # transport every shape_change into the model
        # hidden: output_hidden from encoder
        output, output_hidden = self.lstm(x, hidden)
        output_fc = self.fc(output)  # (batch_size, seq_len, output_size)
        return output_fc, output_hidden

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_len, model_device,
                 num_layers=2, num_directions=2):
        super(Encoder_Decoder, self).__init__()
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

    # add teach_forcing
    """
    def forward(self, x, y):
        # x : 3D (batch_size, param_dim, input_size=1 for now)    param:(quad1_len, ..., quad2_gradient)
        # y : 3D (batch_size, seq_len, input_size=1 for now)
        output_encoder, hidden_encoder = self.encoder(x)
        hidden_decoder = hidden_encoder

        output_decoder, hidden_decoder = self.decoder(y, hidden_decoder)
        # output_decoder : (batch_size, seq_len, output_size=1 for now)
        return output_decoder
    """

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
    def __init__(self, input_size=8, hidden_size1=32, hidden_size2=64, hidden_size3=32,  output_size=4, dropout=0.1):
        super(MLP, self).__init__()
        self.input_size = input_size  # x.shape[1]
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        x = self.drop3(torch.relu(self.fc3(x)))
        y = self.fc4(x)
        return y


class MLP2(nn.Module):
    def __init__(self, input_size=8, hidden_size1=32, hidden_size2=64, hidden_size3=128,
                 hidden_size4=64, hidden_size5=32,
                 output_size=4, dropout=0.1):
        super(MLP2, self).__init__()
        self.input_size = input_size  # x.shape[1]
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.drop3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.drop4 = nn.Dropout(dropout)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.drop5 = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size5, output_size)

    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.drop4(F.relu(self.fc4(x)))
        x = self.drop5(F.relu(self.fc5(x)))
        y = self.output(x)
        return y



# use mean absolute percent error as criterion
def get_mape(tensor_true, tensor_pred):
    return torch.mean(torch.abs((tensor_pred - tensor_true) / tensor_true)) * 100


def get_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)

