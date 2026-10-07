import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def process_file(file_name):
    # Cargar el archivo en un DataFrame
    df = pd.read_csv(file_name, header=None)

    # Eliminar la primera columna
    df.drop(columns=[0], inplace=True)

    # Eliminar la última columna
    df.drop(columns=[df.columns[-1]], inplace=True)

    # Convertir la última columna a enteros
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')

    # Filtrar las filas donde la última columna esté en el rango 1 a 12
    df = df[(df.iloc[:, -1] >= 1) & (df.iloc[:, -1] <= 12)]

    return df

if __name__ == '__main__':
    # Ruta del archivo
    file_name = '/Users/ocorzo/Borrar/MancalaLog/lineasGanadoras.txt'

    # Procesar el archivo y obtener el DataFrame
    df = process_file(file_name)

    # Separar las columnas en X (características) y y (etiquetas)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convertir las etiquetas categóricas a valores numéricos
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Convertir etiquetas a formato categórico (one-hot encoding)
    y = to_categorical(y, num_classes=12)

    # Separar los datos en entrenamiento (90%) y prueba (10%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Crear el modelo de la red neuronal
    model = Sequential()
    model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))  # Primera capa oculta
    model.add(Dense(30, activation='relu'))  # Segunda capa oculta
    model.add(Dense(12, activation='softmax'))  # Capa de salida

    # Compilar el modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenar el modelo con 30 iteraciones (epochs)
    model.fit(X_train, y_train, epochs=30, batch_size=10, verbose=1)

    # Guardar el modelo entrenado en un archivo
    model.save('modelo_mancala_9000.h5')

    # Evaluar el modelo con los datos de prueba
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Mostrar la efectividad del modelo
    print(f"Precisión del modelo: {accuracy * 100:.2f}%")
