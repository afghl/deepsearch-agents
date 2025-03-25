import logging
from logging import getLogger, INFO, StreamHandler
import re
from typing import Any, Dict, Union
from colorama import init, Fore, Style

# 初始化 colorama
init()


class ColoredFormatter(logging.Formatter):

    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # 添加日志级别的颜色
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


# 创建logger
logger = getLogger("deepsearch")
logger.setLevel(INFO)

# 创建控制台处理器
console_handler = StreamHandler()
console_handler.setLevel(INFO)

# 设置彩色日志格式
formatter = ColoredFormatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(console_handler)
