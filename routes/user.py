from flask import json, request, jsonify
from database.db import get_connection

def formData(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT nombre_completo, codEstudiante, facultad_id FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        conn.close()
        return jsonify(success=False, message="Información no encontrada"), 404

    nombre_completo = user[0]
    codEstudiante = user[1]
    facultad_id = user[2]

    cursor.execute("SELECT nombre_facultad FROM facultades WHERE id = %s", (facultad_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        responses = {
            'nom_apell': nombre_completo,
            'codigo': codEstudiante,
            'facultad': result[0],
        }
        return jsonify(success=True, responses=responses)
    else:
        return jsonify(success=False, message="Datos del estudiante no encontrados")


def actualizar_datos(username):
    data = request.json
    nom_apell = data.get('nom_apell')
    codigo = data.get('codigo')
    facultad = data.get('facultad')

    if not (nom_apell and codigo and facultad):
        return jsonify(success=False, message="Faltan datos"), 400

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        conn.close()
        return jsonify(success=False, message="Usuario no encontrado"), 404

    cursor.execute("""
        UPDATE users 
        SET nombre_completo = %s, codEstudiante = %s, facultad_id = (
            SELECT id FROM facultades WHERE nombre_facultad = %s LIMIT 1
        )
        WHERE username = %s
    """, (nom_apell, codigo, facultad, username))

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify(success=True, message="Datos actualizados correctamente")
