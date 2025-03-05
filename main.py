import logging
from app import app

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5001, debug=True)
except Exception as e:
    logger.error(f"Failed to start server: {str(e)}")
    raise