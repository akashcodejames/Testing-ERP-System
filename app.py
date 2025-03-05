import os
import logging
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv
from extensions import db, login_manager, mail
from flask_mail import Mail  # Import Flask-Mail

# Load environment variables from .env file if it exists
load_dotenv()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

# Initialize extensions
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "a secret key"

# Configure the database
if os.environ.get("DATABASE_URL"):
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
else:
    print("Database not connected")
    exit()

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'                     # SMTP server (Gmail example)
app.config['MAIL_PORT'] = 587                                    # Port for TLS
app.config['MAIL_USE_TLS'] = True                                 # Use TLS
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")     # Your email from .env
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")     # Your email password from .env
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get("MAIL_USERNAME")  # Default sender (optional)

# Initialize Flask-Mail
mail = Mail(app)

# Initialize the app with the extensions
db.init_app(app)
login_manager.init_app(app)
mail.init_app(app)
login_manager.login_view = 'auth.login'

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Register blueprints
with app.app_context():
    import models
    from auth import bp as auth_bp
    app.register_blueprint(auth_bp)

    from timetable import timetable_bp as timetable_bp
    app.register_blueprint(timetable_bp, url_prefix='/timetable')

    # Create database tables
    db.create_all()

    # Initialize test data if needed
    try:
        if not models.UserCredentials.query.filter_by(email='admin@example.com').first():
            logger.info("Creating test users and data")
            models.create_test_data()
        else:
            logger.info("Test data already exists")
    except Exception as e:
        logger.error(f"Error creating test data: {str(e)}")
        raise
