import os
import time
import logging
from io import BytesIO
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import numpy as np

class Agent:
    def __init__(self, name='agent', version=0, path='./runs/', subagents=[],
                set_subagent_data=True, save_run=True, save_images=False): 
        # Meta Data
        self.set_meta_data(name, version, path, subagents, set_subagent_data)
        self.models = {}
        self.loggers = {}
        
        self.setup_logger('agent')
        self.save_run = save_run
        self.save_images = save_images
        
    def set_meta_data(self, name=None, version=None, path=None, subagents=None,
                     set_subagent_data=True):
        if path is not None:
            if path[-1] != '/':
                path.append('/')
            self.path = path

        if name is not None:
            self.name = name

        if version is not None:
            self.version = version
            
        if subagents is not None:
            self.subagents = subagents
            
        self.run_path = "{}{}-{}/".format(self.path, self.name, self.version)
        
        if set_subagent_data:
            for i, agent in enumerate(self.subagents):
                agent.set_meta_data(
                    name='subagent',
                    version=i,
                    path=self.run_path + 'subagents/'
                )

        self.model_path = self.run_path + 'models/'
        self.data_path = self.run_path + 'data/'

    def create_new_run(self, update_version=True):
        # Creates a new run
        # i.e., gives the agent a unique name (through versioning)
        # and creates a new directory for the training run
        
        if update_version is True:
            extra_version = 0
            self.log('Creating log directory: {}'.format(self.run_path))
            while True:
                self.version = len(self.get_previous_run_versions()) + extra_version
                proposed_run_path = "{}{}-{}/".format(self.path, self.name, self.version)
                if os.path.isdir(proposed_run_path):
                    extra_version += 1
                else:
                    break
                if extra_version > 999:
                    raise StopIteration('Exceeded 999 runs of the name {}'.format(self.name))

        # Reset the agent's meta data
        self.set_meta_data()

        # Make sure our paths exist
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        for i, agent in enumerate(self.subagents):
            agent.create_new_run(update_version)

        self.log('Run Logs Created: {}'.format(self.run_path))
        
    def setup_logger(self, tag, sub_path=None, log_to_console=False, level=logging.INFO):
        if sub_path is None:
            sub_path = ''
        else:
            if sub_path[-1] != '/':
                sub_path += '/'
                
        logger_path = '{}{}'.format(sub_path, tag)
        logger_name = '{}-{}_{}'.format(self.name, self.version, logger_path)
        handler = logging.FileHandler(self.data_path + "{}.csv".format(logger_path))
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(handler)
        
        if log_to_console:
            console_handler = logging.StreamHandler()
            logger.addHandler(console_handler)
        
        self.loggers[logger_path] = logger
        return logger
    
    def log_scalar(self, tag, value, step, sub_path=None):
        if sub_path is None:
            sub_path = ''
        else:
            if sub_path[-1] != '/':
                sub_path += '/'
                
        logger_path = '{}{}'.format(sub_path, tag)
        self.loggers[logger_path].info("{},{},{}".format(step, time.time.now(), value))

    def log(self, value):
        self.loggers['agent'].info(value)

    def get_previous_run_versions(self):
        try:
            files = os.listdir(self.path)
            files = [f for f in files if self.name + '-' in f]
            return files
        except FileNotFoundError as e:
            self.log("Root path {} doesn't exist.  Creating it...".format(self.path))
            os.makedirs(self.path)
            return []
    
    def save_models(self):
        for model in self.models:
            self.models[model].save_weights(self.model_path + '{}-{}_{}.h5'.format(self.name, self.version, model))
            
    def load_models(self):
        for model in self.models:
            self.models[model].load_weights(self.model_path + '{}-{}_{}.h5'.format(self.name, self.version, model))
