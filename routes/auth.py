from flask import request, jsonify
from database.db import get_connection

def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    conn = get_connection()
    cursor = conn.cursor()  # O usa DictCursor si prefieres campos por nombre
    cursor.execute(
        "SELECT username, nombre_completo FROM users WHERE username=%s AND password=%s",
        (username, password)
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user:
        information = {
            "username": user[0],
            "nombre_completo": user[1]
        }
        return jsonify(success=True, data=information)
    else:
        return jsonify(success=False, message="Credenciales incorrectas"), 401
