import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# ================ Creating log folder ==================== #
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path, exist_ok= True)

# ============== log file with timestamp ================== #
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)


# =================== Creating Logger ===================== #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ======================== log Formate ==================== #
log_format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

### Rotating File Handler
file_handler = RotatingFileHandler(
    filename = LOG_FILE_PATH,
    maxBytes=1_000_000,
    backupCount=5
)
file_handler.setFormatter(formatter)

# ================= attach handlers to logger ============= #
if not logger.hasHandlers():
    logger.addHandler(file_handler)

# =================== log message ====================== #
logger.info("RotatingFileHandler logger is ready")