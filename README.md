# Composición Musical con Redes Neuronales

En este proyecto realicé cree una Red Neuronal LSTM que toma como entrada un fragmento musical y continua la composición de la obra. 
Se uso pytorch para poder crear la red neuronal.

# Procesamiento de los datos
Los datos fueron obtenidos del siguiente [link](https://kern.humdrum.org/cgi-bin/browse?l=essen/europa/deutschl), el cual es una página de la [Essen Associative Code (EsAC)](http://www.esac-data.org/#:~:text=Essen%20Associative%20Code%20(EsAC)%20was,(standing%20for%20rhythmic%20durations).)
 
 Las piezas están en formato **.krn**, por lo que podemos usar la paqueteria de **music21** para poder manejar los archivos.
 
 El primer paso fue leer todas las canciones guardadas y filtrar aquellas que continenen unicamente figuras rítmicas que son multiplos de 
 dieciseisavos (corcheas, negras, negras con puntillo, blancas, etc.), es decir, vamos a eliminar las obras que continene figuras irregulares 
 como por ejemplo los tresillos, quintillos, etc.
 
 
 Despues procedemos a transponer todas las melodías en tonalidad mayor a Do Major, y las menores a la menor. El siguiente paso es pasar todas las notas a
 su respectiva representación en midi, por ejemplo, la nota Do5 es representada por un 62. Consideraremos que los silencios son el símbolo 'r' y si una nota dura
 más de una corchea entonces se agregaran guiones bajo, por ejemplo un Do5 con duración de negra está representado como '60 _ _ _'
 
 Una vez que ya tenemos todas las piezas en formato simbólico, procedemos a crear un único archivo que contiene todas las canciones. Para delimitar entre
 una canción y otra agregamos el siguiente símbolo '/' (es el archivo **dataset_doc**)
 
 
 Para poder manejar de una mejor manera las notas vamos a hacer una especie de diccionario en donde a cada símbolo (nota o silencio) se le asocia un entero.
 En este caso nuestro diccionario es de 38 números (Es el arhivo **dic.json**)
 
 
 Finalmente, creamos las secuencias con las cuales vamos a alimentar la red lstm. Dichas secuencias tienen longitud de 64 y la variable respuesta es la nota 
 número 65, de manera que el objetivo es que la red tome como entrada la secuencia y nos devuelva la nota siguiente.
 
 
