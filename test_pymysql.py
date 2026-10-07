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

    # Cerrar la conexión
    cursor.close()
    conexion.close()

    return resultados

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
