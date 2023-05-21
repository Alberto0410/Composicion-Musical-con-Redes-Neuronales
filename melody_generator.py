import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from preprocesamiento import SEQ_LEN, MAP_PATH
from entrenamiento import model_lstm


class MelodyGen:
    def __init__(self, path_model):
        self.path_model = path_model

        #cargamos la red de pytorch
        # self.model = model_lstm()
        self.model = model_lstm(38, 256)
        self.model.load_state_dict(torch.load(self.path_model))

        #cargamos el diccionario
        with open(MAP_PATH, 'r') as f:
            self.map = json.load(f)

        #recreamos el inicio de la secuencia tal cual como se hizo en el preprocesamiento
        self.start =  ['/'] * SEQ_LEN


    def generation(self, start_melody, steps, max_len, temp):
        '''Entrada:
        start_melody: Secuencia inicial de cómo queremos que empiece la melodía.
                    Tiene que estar en formato simbólico
        steps: Número de pasos que queremos que genere la red
        max_len: Longitud máxima de la melodía
        temp: Número que controla la aleatoriedad de la nota escogida por la red. 
        Entre mayor sea el valor de temp, la red va a elegir de manera uniforme una nota.
        Entre menor sea el valor de temp, la red va a guiarse usando los resultados del softmax.

        '''

        #agregamos la secuencia inicial de diagonales
        start_melody = start_melody.split()
        melody = start_melody
        start_melody = self.start + start_melody

        #usamos el traductor para pasar a números
        start_melody = [self.map[note] for note in start_melody]

        #vamos generando la melodía
        for _ in range(steps):
            #tomamos max_len notas a partir del final
            start_melody = start_melody[-max_len:]

            #pasamos a one hot encoding la semilla para poder
            #hacer predicciones con la red entrenada
            # start_oh = np.eye(len(self.map))[start_melody]
            start_oh = F.one_hot(torch.tensor(start_melody), num_classes = len(self.map))
            start_oh = start_oh.unsqueeze(0).float()

            with torch.no_grad():
                # print(f'start_oh: {start_oh.shape}')
                output1 = self.model.feed_forward(start_oh)[0]

            print(f'output1: {output1.shape}')

            #obtenemos la nota usando la temperatura
            output_int = self.sample_temp(output1.detach().numpy(), temp)

            #agregamos la nota a la melodía
            start_melody.append(output_int)

            #usamos el traductor para pasar el numero a simbolo
            symbol = [key for key, value in self.map.items() if value == output_int][0]

            #terminamos si la red predice un final
            if symbol == '/':
                break
            
            #agregamos el simbolo a la melodía
            melody.append(symbol)

        return melody


    def sample_temp(self, output, temp):
        #reescalamos y despues aplicamos softmax
        pred = np.log(output) / temp
        output = np.exp(pred) / np.sum(np.exp(pred))

        #elegimos una nota de acuerdo a la distribución de probabilidad obtenida
        print(output.shape)
        output_int = np.random.choice(range(len(output)), p = output)

        return output_int



if __name__ == "__main__":
    mg = MelodyGen('model.pt')
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = mg.generation(seed, 500, SEQ_LEN, 0.7)
    print(melody)