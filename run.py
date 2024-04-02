from flask import Flask
from app.routes.models import models_blueprint

app = Flask(__name__)
app.register_blueprint(models_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
