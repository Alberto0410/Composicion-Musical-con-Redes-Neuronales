# !pip install music21
import os
import json
#music21 nos permite manipular musica simbolica en python
# sirve para convertir archivos simbolicos a otros formatos 
import music21 as m21

#indicamos que queremos usar Musescore 4 para abrir los archivos
us = m21.environment.UserSettings()
m21.environment.set("musescoreDirectPNGPath",     "C:/MuseScore4/bin/MuseScore4.exe")
m21.environment.set("musicxmlPath", "C:/MuseScore4/bin/MuseScore4.exe")
us['musicxmlPath']

SONGS_PATH = 'deutschl/test'
SAVE_DIR = 'data_preprocesed'
FINAL_PATH = 'dataset_doc'
MAP_PATH = 'dic.json'
LONG_NOTES = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]

def load_music(path):
    '''Carga todos los archivos de música que están en un directorio.'''
    songs = []

    #iteramos sobre todos los archivos del directorio
    for path, _, files in os.walk(path):
        for file in files:
            #tomamos unicamente los archivos krn
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def filtro_duracion(song, duration):
    '''Función que toma las canciones que unicamente contiene las duraciones de la lista duración
    Entrada: Duración es una lista que contiene numeros los cuales indicar las duraciones de una nota
    
    Las duraciones estan expresadas un 4/4, es decir que:
    4 = una redonda
    3 = blanca con puntillo
    2 = blanca
    1 = negra
    0.75 = negra con puntillo
    0.5 = corchea
    0.25 = semicorchea
    '''
    #iteramos sobre todas las notas de la cancion, para ello pasamos
    # todas las notas a una lista usando .flat.notes
    for note in song.flat.notesAndRests:
        #si la duracion de la nota no esta en la lista de duraciones, regresamos Falso
        if note.duration.quarterLength not in duration:
            return False
    return True


def transpose_song(song):
    '''Vamos a pasar la melodía de la tonalidad a la que esté a C Maj/A min
    Primero se extrae la tonalidad de la canción dividiendola por partes hasta 
    llegar a obtener el compás 0, el cual contiene la tonalidad.
    
    En caso de que no tenga la armadura escrita, entonces la estimamos'''
    
    #Obtenemos todos los elementos de la cancion, como el score, partes, etc.
    parts = song.getElementsByClass(m21.stream.Part)
    #Obtenemos todos los compases de la primera parte
    measure0 = parts[0].getElementsByClass(m21.stream.Measure)
    #La tonalidad esta en la posicion 5 del compas 0
    key = measure0[0][4]

    #consideramos el caso de que la armadura no este escrita
    if not isinstance(key, m21.key.Key):
        #estimamos la tonalidad
        key = song.analyze('key')

    #Obtenemos la distancia (intervalo) para transponer
    if key.mode == 'major':
        #transponemos a C Maj
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    
    elif key.mode == 'minor':
        #transponemos a A min
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

    #transponemos la cancion
    trans_song = song.transpose(interval)

    return trans_song


def encode_song(song, min_dur = 0.25):
    '''Dada una canción, se pasa a un formato de str, es decir, en formato 
    simbólico. Se representan como series de tiempo
    
    En caso de que sea la primera vez que aparece una nota, se guardara su num de nota
    En caso de que se repita, guardaremos un _ por cada cuarto de nota que dure
    Y en caso de que sea un silencio, guardaremos una r'''
    time_serie = []

    #tomamos todos los items de la cancion
    for item in song.flat.notesAndRests:
        #consideramos si es una nota, entonces obtenemos el num de nota
        if isinstance(item, m21.note.Note):
            symbol = item.pitch.midi
        
        #ahora el caso en que sea silencio
        elif isinstance(item, m21.note.Rest):
            symbol = 'r'

        #convertimos la melodia a una serie de tiempo
        duration = int(item.duration.quarterLength / min_dur)
        for time in range(duration):
            if time == 0:
                time_serie.append(symbol)
            else:
                time_serie.append('_')
    
    #pasamos la lista a un string
    enc_song = " ".join(map(str, time_serie))
    return enc_song


def preproces(path):
    songs = load_music(path)
    print(f'El número de archivos cargados es {len(songs)}')

    #verificamos que todas las cancioes cumplan el filtro de la duracion
    for i, song in enumerate(songs):
        if not filtro_duracion(song, LONG_NOTES):
            continue

        #transponemos las canciones
        song = transpose_song(song)

        #codificamos las canciones
        enc_song = encode_song(song)

        #guardamos las canciones en formato simbolico en un directorio
        save_path = os.path.join(SAVE_DIR, str(i))

        with open(save_path, 'w') as f:
            f.write(enc_song)


def load(file_path):
    'Función que dado un archivo, lo carga y regresa su contenido'
    with open(file_path, 'r') as f:
        song = f.read()
    return song

def doc_dataset(data_path, final_path, max_lenght = 64):
    '''Función que crea un solo documento donde estan 
    guardadas todas las canciones.

    Para separa cada canción de otra se usará un delimitador.
    Entrada:
    data_path : directorio donde están las canciones en formato simbólico
    final_path : directorio donde se guardara el documento final
    max_length : longitud máxima que tendrá cada canción para poder meterla a la RN
    '''
    delim = '/ ' * max_lenght

    songs = ''

    #iteramos sobre todas las canciones
    for path, _, files in os.walk(data_path):
        for file in files:
            #leemos todas las canciones
            song = load(os.path.join(path, file))
            
            #agregamos la cancion al dataset
            songs += song + ' ' + delim

    #el delimitador cuenta con un estpacio al final, el cual no consideramos
    songs = songs[:-1]
    
    #guardamos el dataset y lo regresamos
    with open(final_path, 'w') as f:
        f.write(songs)

    return songs


def translate(songs, map_path):
    '''Función que asocia a cada simbolo un entero con el objetivo de 
    poder usarlo en la RN
    
    Entrada:
    songs : dataset de canciones en formato simbolico
    map_path : directorio donde se guardara el diccionario en formato .json'''
    mapping = {}

    #obtenemos todos los simbolos
    songs = songs.split()
    voc = list(set(songs))

    #asignamos un entero a cada simbolo
    for i, symbol in enumerate(voc):
        mapping[symbol] = i

    #guardamos el diccionario en un archivo json
    with open(map_path, 'w') as f:
        json.dump(mapping, f, indent = 4)

def main():
    preproces(SONGS_PATH)
    songs = doc_dataset(SAVE_DIR, FINAL_PATH)
    translate(songs, MAP_PATH)


if __name__ == '__main__':
    main()