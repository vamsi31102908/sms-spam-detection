from flask import Flask
from app.routes import routes  # Import the correct Blueprint

def create_app():
    app = Flask(__name__)

    # Register Blueprints
    app.register_blueprint(routes)

    return app
