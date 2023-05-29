from preprocesamiento import training_seq, SEQ_LEN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


#es el tamaño del diccionario
OUTPUT_UNITS = 38
NUM_UNITS = 256
LR = 0.001
EPOCH = 35
BATCH_SIZE = 64
SAVE_MODEL_PATH = 'model.pt'

#clase para generar una red lstm
class model_lstm(nn.Module):
    def __init__(self, output_units, num_units):
        super(model_lstm, self).__init__()
        self.hidden_size = num_units
        self.lstm = nn.LSTM(input_size = 1, hidden_size = num_units, num_layers = 1, batch_first = True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(num_units, output_units) 
        
        #agragamos una capa softmax
        # self.softmax = nn.Softmax(dim = 1)

    def feed_forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        output = self.linear(lstm_out)
        # output = self.softmax(output)

        return output

def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, lr = LR, num_epochs = EPOCH):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model = model_lstm(output_units, num_units)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    try:
        model.load_state_dict(torch.load(SAVE_MODEL_PATH))
        print('Modelo cargado')
    except:
        print('Modelo nuevo')
    
    inputs, targets = training_seq(SEQ_LEN)
    targets = torch.from_numpy(targets).long()

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_model = None
    best_loss = np.inf

    for epoch in range(num_epochs):
        #mini batch
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()

            #hacemos feed forward
            outputs = model.feed_forward(batch_inputs)
            loss = criterion(outputs, batch_targets)

            #backpropagation
            loss.backward()
            optimizer.step()

            
        #validamos
        model.eval()
        error_eval = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                outputs = model.feed_forward(batch_inputs)
                error_eval += criterion(outputs, batch_targets)
                
            #si encontramos un mejor modelo lo guardamos
            if error_eval < best_loss:
                best_loss = error_eval
                best_model = model.state_dict()
            
            print(f'Epoch: {epoch} | Error: {error_eval}')

    torch.save(best_model, SAVE_MODEL_PATH)



if __name__ == '__main__':
    train()


