import logging
from logging import getLogger, INFO, StreamHandler
import re
import json
from rich.logging import RichHandler
from rich.console import Console
from rich import print_json


# 创建Rich控制台，使用自定义主题
console = Console()


class ColoredDictLogHandler(RichHandler):

    def emit(self, record):
        if isinstance(record.msg, dict):
            msg = record.msg
            try:
                record.msg = ""
                super().emit(record)
                print_json(data=msg, indent=4)
            except Exception as e:
                print(e)
                record.msg = msg
                super().emit(record)
        else:
            # 对于非字典消息，使用默认处理
            super().emit(record)


# 创建logger
logger = getLogger("deepsearch")
logger.setLevel(INFO)

# 创建自定义Rich控制台处理器
rich_handler = ColoredDictLogHandler(
    console=console, rich_tracebacks=True, show_time=False
)
rich_handler.setLevel(INFO)

# 设置日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

rich_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(rich_handler)


if __name__ == "__main__":
    logger.info("Hello, world!")

    logger.info(
        {
            "action": "search",
            "think": "thinking about the query...",
        }
    )
