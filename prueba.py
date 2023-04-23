# !pip install music21
import os
#music21 nos permite manipular musica simbolica en python
# sirve para convertir archivos simbolicos a otros formatos 
import music21 as m21

#indicamos que queremos usar Musescore 4 para abrir los archivos
us = m21.environment.UserSettings()
m21.environment.set("musescoreDirectPNGPath",     "C:/MuseScore4/bin/MuseScore4.exe")
m21.environment.set("musicxmlPath", "C:/MuseScore4/bin/MuseScore4.exe")
us['musicxmlPath']

songs_path = 'deutschl/test'
long_nota = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]

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

def preprocesamiento(path):
    songs = load_music(path)
    print(f'El número de archivos cargados es {len(songs)}')

    #verificamos que todas las cancioes cumplan el filtro de la duracion
    for song in songs:
        if not filtro_duracion(song, long_nota):
            continue

    #transponemos las canciones
    song = transpose_song(song)


if __name__ == '__main__':
    songs = load_music(songs_path)
    print(f'Se han cargado {len(songs)} archivos')
    song = songs[0] 

    print(f'Tiene duracion aceptable? {filtro_duracion(song, long_nota)}')
    transpose_song = transpose_song(song)
    
    song.show()
    transpose_song.show()