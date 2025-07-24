from flask import json, request, jsonify
from database.db import get_connection

def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, nombre_completo FROM users WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user:
        information={
            "username":user[0],
            "nombre_completo":user[1]
        }
        return jsonify(success=True,data=information)#Metodo que convierte un objeto de Python en una respuesta JSON.
    else:
        return jsonify(success=False,message="Credenciales incorrectas"), 401