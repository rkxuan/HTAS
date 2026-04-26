#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from typing import Dict
from loguru import logger

from hats.utils.const import HATS_ROOT

class result_logger():
    def __init__(self, experiment_name:str, record_time:str, args:Dict):
        self.experiment_name = experiment_name
        self.record_time = record_time
        self.args = args
        self.train_batch_records = []
        self.pretrain_time = -1
        self.train_time = -1
        self.eval_time = -1
        self.eval_accuracy = -1
        self.train_tokens = -1
        self.eval_tokens = - 1
        self.nedges = -1
        self.length = -1

    def get_train_batch_record(self, train_batch_record:str):
        self.train_batch_records.append(train_batch_record)
    
    def get_eval_accuracy(self, eval_accuracy):
        self.eval_accuracy = eval_accuracy

    def get_pretrain_time(self, pretrain_time):
        self.pretrain_time = pretrain_time

    def get_eval_time(self, eval_time):
        self.eval_time = eval_time
    
    def get_train_time(self, train_time):
        self.train_time = train_time

    def get_eval_tokens(self, eval_tokens):
        self.eval_tokens = eval_tokens

    def get_train_tokens(self, train_tokens):
        self.train_tokens = train_tokens

    def get_nedges(self, nedges):
        self.nedges = nedges

    def get_length(self, length):
        self.length = length

    def output2txt(self, file_path=None):
        if not file_path:
            run_path = os.path.join(HATS_ROOT, "run")
            experiment_path = os.path.join(run_path, self.experiment_name)
            file_path = os.path.join(experiment_path, self.record_time) + ".txt"
        
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        with open(file_path, 'w') as file:
            file.write("{:<25}{}\n".format("KEY","VALUE"))
            for key, value in self.args.items():
                file.write(f"{key:<25}{value}\n")
            file.write("Pretraining Process:\n")
            
            file.write("{:.3f}s\n".format(self.pretrain_time))

            if self.train_batch_records:
                for train_batch_record in self.train_batch_records:
                    file.write(f"{train_batch_record}\n")
                file.write("Training time:\n")
                file.write("{:.3f}s\n".format(self.train_time))
                file.write("Training token consumption per task:\n")
                file.write("{:.3f}\n".format(self.train_tokens))

            file.write("Evaluation Result:\n")
            file.write(f"{self.eval_accuracy}\n")
            file.write("Evaluation token consumption per task:\n")
            file.write("{:.3f}\n".format(self.eval_tokens))
            file.write("Evaluation time per task:\n")
            file.write("{:.3f}s\n".format(self.eval_time))
            file.write("The number of edges:\n")
            file.write(f"{self.nedges}\n")
            file.write("The length of longest path:\n")
            file.write(f"{self.length}\n")

def configure_logging(
    print_level: str = "INFO", 
    logfile_level: str = "DEBUG"
) -> None:
    """
    Configure the logging settings for the application.

    Args:
        print_level (str): The logging level for console output.
        logfile_level (str): The logging level for file output.
    """
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    logger.add(HATS_ROOT / 'logs/log.txt', level=logfile_level, rotation="10 MB")

def initialize_log_file(
    experiment_name: str, 
    time_stamp: str
) -> Path:
    """
    Initialize the log file with a start message and return its path.

    Args:
        mode (str): The mode of operation, used in the file path.
        time_stamp (str): The current timestamp, used in the file path.

    Returns:
        Path: The path to the initialized log file.
    """
    try:
        log_file_path = HATS_ROOT / f'result/{experiment_name}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path
