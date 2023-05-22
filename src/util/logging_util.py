import logging, sys

PROJECT_NAME = "TAS Electoral Spending"

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(PROJECT_NAME)
