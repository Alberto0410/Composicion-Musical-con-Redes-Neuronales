from preprocesamiento import training_seq, SEQ_LEN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# import tensorflow.keras as keras

#es el tamaño del diccionario
OUTPUT_UNITS = 38
NUM_UNITS = 256
LOSS =  nn.CrossEntropyLoss()
LR = 0.1
EPOCH = 10
BATCH_SIZE = 32
SAVE_MODEL_PATH = 'model.pt'

#clase para generar una red lstm
class model_lstm(nn.Module):
    def __init__(self, output_units, num_units):
        super(model_lstm, self).__init__()
        # self.lstm = nn.LSTM(output_units, num_units)
        self.hidden_size = num_units
        self.lstm = nn.LSTM(output_units, num_units)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(num_units, output_units) 
        
        #agragamos una capa softmax
        self.softmax = nn.Softmax(dim = 1)

    def feed_forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out[:, -1, :])
        output = self.softmax(output)

        return output

def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, lr = LR, num_epochs = EPOCH, learning_rate = LR):
    criterion = nn.CrossEntropyLoss()
    model = model_lstm(output_units, num_units)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    try:
        model.load_state_dict(torch.load(SAVE_MODEL_PATH))
        print('Modelo cargado')
    except:
        print('Modelo nuevo')
    
    inputs, targets = training_seq(SEQ_LEN)
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).long()

    for epoch in range(num_epochs):
        #error
        loss = 0

        #mini batch
        for i in range(0, len(inputs), BATCH_SIZE):
            batch_inputs = inputs[i:i+BATCH_SIZE]
            batch_targets = targets[i:i+BATCH_SIZE]

            optimizer.zero_grad()

            #hacemos feed forward
            outputs = model.feed_forward(batch_inputs)
            loss = criterion(outputs, batch_targets)

            #backpropagation
            loss.backward()
            optimizer.step()

            loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{ num_epochs}], Loss: {loss.item()}')

    torch.save(model.state_dict(), SAVE_MODEL_PATH)



if __name__ == '__main__':
    train()


