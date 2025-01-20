import logging
from datetime import datetime
import os


class SimLogger:
    VERBOSE = 5
    DEBUG = 4
    INFO = 3
    TRAINING = 2
    ERROR = 1
    NONE = 0

    def __init__(self, level=TRAINING, log_file=None):
        self.level = level

        # Set up logging to file if specified
        if log_file:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(f"{log_dir}/{log_file}_{timestamp}.log", "w")
        else:
            self.log_file = None

    def log(self, message, level=INFO):
        if level <= self.level:
            if level == self.TRAINING:
                print("\033[94m" + message + "\033[0m")  # Blue color for training
            elif level == self.ERROR:
                print("\033[91m" + message + "\033[0m")  # Red color for errors
            else:
                print(message)

            if self.log_file:
                self.log_file.write(f"{message}\n")
                self.log_file.flush()

    def verbose(self, message):
        self.log(message, self.VERBOSE)

    def debug(self, message):
        self.log(message, self.DEBUG)

    def info(self, message):
        self.log(message, self.INFO)

    def training(self, message):
        self.log(message, self.TRAINING)

    def error(self, message):
        self.log(message, self.ERROR)

    def close(self):
        if self.log_file:
            self.log_file.close()
