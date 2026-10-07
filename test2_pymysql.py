import os
import pymysql

def buscar_datos_por_pos(host, database, username, password, valor_pos):
    # Configurar la conexión a la base de datos
    conexion = pymysql.connect(host=host, user=username, password=password, database=database)

    cursor = conexion.cursor()

    # Consulta para buscar el valor específico en la tabla "Posicion"
    consulta = "SELECT * FROM Posicion WHERE Pos = %s"
    cursor.execute(consulta, (valor_pos,))

    # Obtener los resultados
    resultados = cursor.fetchall()

    # Si no se encuentran resultados, insertar un nuevo registro
    if not resultados:
        print(f"No se encontró el valor '{valor_pos}'. Creando una nueva entrada.")
        insertar_nuevo(cursor, valor_pos)
        conexion.commit()
        # Ejecutar la consulta nuevamente para obtener el nuevo registro
        cursor.execute(consulta, (valor_pos,))
        resultados = cursor.fetchall()

    # Cerrar la conexión
    cursor.close()
    conexion.close()

    return resultados

def insertar_nuevo(cursor, valor_pos):
    # Crear un nuevo registro con valores predeterminados
    consulta_insert = "INSERT INTO Posicion (Pos) VALUES (%s)"
    valores = (valor_pos)  # Cambia estos valores según sea necesario
    cursor.execute(consulta_insert, valores)

# Ejemplo de uso
host = os.environ["MANCALA_DB_HOST"]
database = os.environ["MANCALA_DB_NAME"]
username = os.environ["MANCALA_DB_USER"]
password = os.environ["MANCALA_DB_PASSWORD"]
valor_pos = "Prueba@222233@"

resultados = buscar_datos_por_pos(host, database, username, password, valor_pos)
#print(resultados)

jugadas, a, b, c, d, e, f, updates = resultados[0]

# Imprimir los resultados
print(a, b, c, d, e, f)
