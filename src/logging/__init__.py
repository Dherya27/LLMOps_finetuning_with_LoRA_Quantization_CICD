import os
import sys
import logging

# What do we want to log? Date Time, logging level, name of module, and a Message
# Create logger string for that
# Then specify a log directory and logfile name
# COnfigure the logger and add handler
# Create logger object




logging_str = "[%(asctime)s,%(levelname)s,%(module)s,%(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    # level
    level=logging.INFO,
    # format
    format=logging_str,
    # Handler
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("MLOps")
