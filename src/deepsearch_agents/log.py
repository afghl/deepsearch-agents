import logging
from logging import getLogger, INFO, StreamHandler
import re
import json
from rich.logging import RichHandler
from rich.console import Console
from rich import print_json


# create a rich console
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
            super().emit(record)


# create a logger
logger = getLogger("deepsearch")
logger.setLevel(INFO)

# create a custom rich console handler
rich_handler = ColoredDictLogHandler(
    console=console, rich_tracebacks=True, show_time=False
)
rich_handler.setLevel(INFO)

formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

rich_handler.setFormatter(formatter)

logger.addHandler(rich_handler)

if __name__ == "__main__":
    logger.info("Hello, world!")

    logger.info(
        {
            "action": "search",
            "think": "thinking about the query...",
        }
    )
