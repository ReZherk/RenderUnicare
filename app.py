import os
from flask import Flask
from flask_cors import CORS


from routes.auth import login
from routes.evaluation import evaluation, get_evaluation  
from routes.user import formData, actualizar_datos

app = Flask(__name__)
CORS(app)


app.add_url_rule('/api/login', 'login', login, methods=['POST'])
app.add_url_rule('/api/evaluation', 'evaluation', evaluation, methods=['POST'])
app.add_url_rule('/api/evaluation/<username>/<month>', 'get_evaluation', get_evaluation, methods=['GET'])
app.add_url_rule('/api/datos/<username>', 'formData', formData, methods=['GET'])
app.add_url_rule('/api/datos_actualizados/<username>', 'actualizar_datos', actualizar_datos, methods=['PUT'])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)