from .Config import Config
import numpy as np


class Logger:
    def __init__(self, config, sandbox):
        self.config = config
        self.sandbox = sandbox
        self.record = {}
    def reset(self):
        self.record = {}
    def __call__(self):
        return self.record

    def add_record(self, record):
        for key, value in record.items():
            if key not in self.record.keys():
                self.record[key] = [value]
            else:
                self.record[key].append(value)
