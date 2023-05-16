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
        self.lstm = nn.LSTM(output_units, num_units)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(num_units, output_units)
        
        #agragamos una capa softmax
        self.softmax = nn.Softmax(dim = 1)

    def feed_forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.softmax(output)

        return output


def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, lr = LR):
    inputs, targets = training_seq(SEQ_LEN)
    targets = np.eye(38)[targets]
    #los pasamos a tensores
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).long()

    #creamos la red y el optimizador
    red_lstm = model_lstm(output_units, num_units)
    optimizer = optim.Adam(red_lstm.parameters(), lr = lr)

    #entrenamos
    for epoch in range(EPOCH):
        loss_total = 0

        #mini batch
        for i in range(0, len(inputs), BATCH_SIZE):
            batch_inputs = inputs[i:i+BATCH_SIZE]
            batch_targets = targets[i:i+BATCH_SIZE]

            optimizer.zero_grad()

            #hacemos feed forward
            outputs = red_lstm.feed_forward(batch_inputs)
            loss_val = loss(outputs, batch_targets)

            #backpropagation
            loss_val.sum().backward()
            optimizer.step()

            loss_total += loss_val.sum().item()

        print(f'Epoch: {epoch + 1} \t Loss: {loss_total:.5f}')

    #guardamos el modelo
    torch.save(red_lstm.state_dict(), SAVE_MODEL_PATH)


    # #emepezamos a entrenar la red
    # for epoch in range(EPOCH):
    #     optimizer.zero_grad()
    #     output = red_lstm.feed_forward(inputs)

    #     #calculamos el error y actualizamos los pesos
    #     # loss_val = loss(output.permute(0, 2, 1), targets)
    #     loss_val = loss(output.view(-1, output_units), targets(-1))
    #     loss_val.backward()
    #     optimizer.step()

    #     print(f'Epoch: {epoch + 1} \t Loss: {loss_val.item():.5f}')

    # #guardamos el modelo
    # torch.save(red_lstm.state_dict(), SAVE_MODEL_PATH)


if __name__ == '__main__':
    train()


