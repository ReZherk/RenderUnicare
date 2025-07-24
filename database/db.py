import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        user="tu_usuario",
        password="tu_contraseña",
        database="unicare"
    )
