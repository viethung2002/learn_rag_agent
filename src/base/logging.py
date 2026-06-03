# app/core/base_logging.py
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

# Thư mục log
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

# Format log
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Handlers
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# Cấu hình root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)

# Shortcut để import dễ dàng
logger = logging.getLogger("app")
