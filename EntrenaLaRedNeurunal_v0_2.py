import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras_tuner import HyperModel, RandomSearch

def process_file(file_name):
    # Cargar el archivo en un DataFrame
    df = pd.read_csv(file_name, header=None)

    '''
# Calcular el índice que representa la mitad del DataFrame
percentage = 1
percentge_index = int(len(df) * percentage)

# Seleccionar la primera mitad de las filas
df = df.iloc[:percentge_index]
    '''

    df = df.sample(frac=.25).reset_index(drop=True)    # Seleccionar el 25% de las filas de forma aleatoria, una especie de dropout

    #Eliminar la última columna
    df = df.drop(df.columns[-1], axis=1)

    # Convertir las etiquetas categóricas a valores numéricos
    label_encoder = LabelEncoder()
    df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1])

    # Convertir la última columna a enteros
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')

    # Filtrar las filas donde la última columna esté en el rango 1 a 12
    df = df[(df.iloc[:, -1] >= 1) & (df.iloc[:, -1] <= 12)]

    return df

if __name__ == '__main__':
    # Ruta del archivo
    file_name = '/Users/ocorzo/Borrar/temp.txt1'

    # Procesar el archivo y obtener el DataFrame
    df = process_file(file_name)

    # Separar las columnas en X (características) y y (etiquetas)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convertir etiquetas a formato categórico (one-hot encoding)
    y = to_categorical(y, num_classes=12)

    # Separar los datos en entrenamiento (90%) y prueba (10%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# función para crear el modelo de la red neuronal
def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(32, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Compilar el modelo
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

    # Crear el modelo de la red neuronal
model = create_model(X_train.shape[1], y_train.shape[1])

    # Entrenar el modelo con 10 iteraciones (epochs)
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

    # Guardar el modelo entrenado en un archivo
model.save('/Users/ocorzo/Borrar/modelo_10000_32neuronas_mse.keras')

    # Evaluar el modelo con los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Mostrar la efectividad del modelo
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
