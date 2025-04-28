# 配置日志
import logging
from logging.handlers import TimedRotatingFileHandler


logger = logging.getLogger("aibot_800a")
logger.setLevel(logging.DEBUG)

import os
if not os.path.exists("logs"):
    os.makedirs("logs")

handler = TimedRotatingFileHandler(
    filename="logs/vec_db.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console_handler)