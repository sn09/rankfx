"""Module with logging setup implementation."""

import logging
import logging.config

import yaml

_CONF = """version: 1
disable_existing_loggers: False
formatters:
  default:
    format: '[%(asctime)s]{%(filename)s:%(lineno)s} - %(levelname)s - %(message)s'
handlers:
  default:
    formatter: default
    class: logging.StreamHandler
    stream: ext://sys.stderr
root:
  level: INFO
  handlers:
    - default
  propagate: no
"""


def get_logger(name: str) -> logging.Logger:
    """Get logger with given name.

    Args:
        name: logger name
    """
    logger = logging.getLogger(name)
    logging.config.dictConfig(yaml.safe_load(_CONF))
    return logger
