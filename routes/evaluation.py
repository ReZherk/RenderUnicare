from flask import json, request, jsonify
from database.db import get_connection
from models.predictor import predict_depression


def evaluation():
    data = request.json
    username = data.get('username')
    month = data.get('month')
    responses = data.get('responses')

    if not (username and month and responses):
        return jsonify(success=False, message="Faltan datos"), 400

    # Validaciones básicas
    if len(responses) != 9:
        return jsonify(success=False, message="Se requieren exactamente 9 respuestas"), 400

    if not all(isinstance(val, (int, float)) and 0 <= val <= 3 for val in responses):
        return jsonify(success=False, message="Todas las respuestas deben ser valores entre 0 y 3"), 400

    # Realizar predicción con IA
    prediction_result = predict_depression(responses)

    if prediction_result['success']:
        print(f"🔮 Predicción realizada: {prediction_result['percentage']:.2f}% - {prediction_result['interpretation']}")
    else:
        print(f"❌ Error en predicción: {prediction_result['error']}")

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if not user:
            return jsonify(success=False, message="Usuario no encontrado"), 404

        user_id = user[0]
        responses_json = json.dumps(responses)

        ai_result = json.dumps({
            'percentage': prediction_result.get('percentage', 0),
            'result': prediction_result.get('result', 0),
            'interpretation': prediction_result.get('interpretation', 'Sin predicción'),
            'error': prediction_result.get('error', None),
            'success': prediction_result.get('success', False)
        })

        # PostgreSQL-compatible UPSERT (requiere UNIQUE constraint en user_id + month)
        cursor.execute("""
            INSERT INTO monthly_responses (user_id, month, responses, results, submitted_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, month) DO UPDATE SET
                responses = EXCLUDED.responses,
                results = EXCLUDED.results,
                submitted_at = CURRENT_TIMESTAMP
        """, (user_id, month, responses_json, ai_result))

        conn.commit()

        return jsonify(
            success=True,
            message="Respuestas guardadas correctamente",
            prediction={
                'depression_probability': prediction_result.get('percentage', 0),
                'has_depression_indicators': prediction_result.get('result', 0) == 1,
                'interpretation': prediction_result.get('interpretation', 'Sin predicción'),
                'prediction_success': prediction_result.get('success', False),
                'error': prediction_result.get('error', None)
            }
        )

    except Exception as e:
        conn.rollback()
        print(f"❌ Error en base de datos: {str(e)}")
        return jsonify(success=False, message=f"Error al procesar: {str(e)}"), 500

    finally:
        cursor.close()
        conn.close()


def get_evaluation(username, month):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if not user:
            return jsonify(success=False, message="Usuario no encontrado"), 404

        user_id = user[0]

        cursor.execute("""
            SELECT responses, results FROM monthly_responses 
            WHERE user_id = %s AND month = %s
        """, (user_id, month))

        result = cursor.fetchone()

        if result:
            responses = json.loads(result[0]) if result[0] else []
            results = json.loads(result[1]) if result[1] else {}

            return jsonify(
                success=True,
                responses=responses,
                results=results
            )
        else:
            return jsonify(success=False, message="Sin respuestas registradas")

    except Exception as e:
        return jsonify(success=False, message=f"Error al obtener evaluación: {str(e)}"), 500

    finally:
        cursor.close()
        conn.close()
