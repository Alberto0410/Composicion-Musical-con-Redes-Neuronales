import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from preprocesamiento import SEQ_LEN, MAP_PATH
from entrenamiento import model_lstm
import music21 as m21


class MelodyGen:
    def __init__(self, path_model):
        self.path_model = path_model

        #cargamos la red de pytorch
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
            # print(start_melody)
            start_np = np.reshape(start_melody, (1, len(start_melody), 1)) 
            start_ten = torch.tensor(start_np, dtype = torch.float32)           # start_oh = F.one_hot(torch.tensor(start_melody).to(torch.int64), num_classes = len(self.map))
            # start_oh = start_oh.unsqueeze(0).float()

            #obtenemos la prediccion
            with torch.no_grad():
                output1 = self.model.feed_forward(start_ten)
                # print(output1)
                # print()

            #obtenemos la nota usando la temperatura
            # output_int = self.sample_temp(output1.detach().numpy(), temp)

            #obtenemos la nota con mayor probabilidad
            output_int = int(output1.argmax())
            # print(output_int)

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
        # print(pred)
        output = np.exp(pred) / np.sum(np.exp(pred))

        #elegimos una nota de acuerdo a la distribución de probabilidad obtenida
        output_int = np.random.choice(range(len(output)), p = output)

        return output_int


    def midi(self, melody, file_name = 'melody_generated.midi', step_duration = 0.25):
        '''Función que toma una melodía generada por la red y la transforma en un 
        archivo .midi'''

        #creamos un stream
        stream = m21.stream.Stream()

        #dado que el simbolo "_" representa un cuarto de nota, entonces
        #vamos a ir contando cuantas veces se repite una nota
        start_symbol = None
        counter = 1

        #iteramos sobre la melodía
        for i, symbol in enumerate(melody):
            #dividimos los casos en los que es una nota/silencio o si se mantiene la nota
            #consideramos el caso en que la melodía termina con una nota y no con silencio
            if symbol != '_' or i + 1 == len(melody):
                
                #nos aseguraos de no sea la primera vez que entramos al ciclo
                if start_symbol is not None:
                    #calculamos la duracion de la nota anterior
                    len_duration = step_duration * counter

                    #dividimos entre si es una nota o un silencio
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength = len_duration)

                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength = len_duration)

                    #agregamos el evento al stream
                    stream.append(m21_event)

                    #como ya no estamos contando cuanto duro la nota pasada, reiniciamos el contador
                    counter = 1
                
                start_symbol = symbol

            else:
                counter += 1
            
        stream.write('midi', file_name)



if __name__ == "__main__":
    mg = MelodyGen('model.pt')
    seed = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generation(seed, 500, SEQ_LEN, 0.7)
    print(melody)
    mg.midi(melody)