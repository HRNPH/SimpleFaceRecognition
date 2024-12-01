from loguru import logger
import sys

# Remove default logger
logger.remove()

# Add console logger
logger.add(
    sys.stdout,
    level="DEBUG",  # Adjust log level as needed
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)
logger.add(
    sys.stderr,
    level="ERROR",
    format="<red>{time:YYYY-MM-DD HH:mm:ss}</red> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)
logger.add(
    sys.stderr,
    level="CRITICAL",
    format="<red>{time:YYYY-MM-DD HH:mm:ss}</red> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)

# Add file logger
# logger.add(
#     "logs/{time:YYYY-MM-DD}.log",
#     level="INFO",
#     format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
#     rotation="10 MB",
#     retention="7 days",
#     compression="zip",
# )
