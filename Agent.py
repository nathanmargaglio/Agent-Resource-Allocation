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
                set_subagent_data=True, save_run=True, save_images=False, verbose=0): 
        # Meta Data
        self.set_meta_data(name, version, path, subagents, set_subagent_data)
        self.models = {}
        self.loggers = {}
        self.handlers = {}
        
        self.log_formatter = logging.Formatter("%(asctime)s %(message)s")
        self.verbose = verbose
        self.save_run = save_run
        self.save_images = save_images
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close_log()

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
        os.makedirs(self.data_path, exist_ok=True)

        for i, agent in enumerate(self.subagents):
            agent.create_new_run(update_version)
            
        self.setup_logger('agent', verbose=self.verbose > 0, use_formatter=True)
        self.setup_logger('agent_debug', verbose=self.verbose > 1, use_formatter=True)
        
        self.log('Run Logs Created: {}'.format(self.run_path))
        self.logd('Debug logs created.')
        
    def setup_logger(self, tag, sub_path=None, verbose=False, level=logging.INFO, close_previous_version=True, use_formatter=False):
        
        if close_previous_version and type(sub_path) is int:
            previous_sub_path = str(sub_path - 1)
            if previous_sub_path[-1] != '/':
                previous_sub_path += '/'
            previous_sub_path = '{}{}'.format(previous_sub_path, tag)

            self.close_log(previous_sub_path)
                
        if sub_path is None:
            sub_path = ''
        else:
            sub_path = str(sub_path)
            if len(sub_path) < 1 or sub_path[-1] != '/':
                sub_path += '/'
                
        logger_path = '{}{}'.format(sub_path, tag)
        logger_name = '{}-{}_{}'.format(self.name, self.version, logger_path)
        os.makedirs(self.data_path + sub_path, exist_ok=True)
        handler = logging.FileHandler(self.data_path + "{}.txt".format(logger_path))
        logger = logging.getLogger(logger_name)
        if use_formatter:
            handler.setFormatter(self.log_formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
        
        if verbose:
            console_handler = logging.StreamHandler()
            if use_formatter:
                console_handler.setFormatter(self.log_formatter)   
            logger.addHandler(console_handler)
            
        self.handlers[logger_path] = handler
        self.loggers[logger_path] = logger
        
        return self.loggers[logger_path]
    
    def fetch_or_create_logger(self, tag, sub_path=None):
        passed_sub_path = sub_path
        if sub_path is None:
            sub_path = ''
        else:
            sub_path = str(sub_path)
            if sub_path[-1] != '/':
                sub_path += '/'
                
        logger_path = '{}{}'.format(sub_path, tag)
   
        if logger_path in self.loggers.keys():
            return self.loggers[logger_path]
        else:
            return self.setup_logger(tag, passed_sub_path)

    def close_log(self, logger_path=None):
        if logger_path is None:
            logger_paths = list(self.loggers.keys())[:]
            for lg in logger_paths:
                self.loggers.pop(lg)
                self.handlers.pop(lg).close()
        else:
            if logger_path in self.loggers.keys():
                self.loggers.pop(logger_path)
                self.handlers.pop(logger_path).close()
            
    def log_scalar(self, tag, value, step, sub_path=None):
        logger = self.fetch_or_create_logger(tag, sub_path)
        logger.info("{},{},{}".format(step, time.time(), value))
        
    def log_ndarray(self, tag, array, step, sub_path=None):
        logger = self.fetch_or_create_logger(tag, sub_path)
        shape = np.array(array.shape, dtype=float).tobytes()
        value = array.astype(float).tobytes()
        logger.info("{},{},{},{}".format(step, time.time(), shape, value))       

    def log(self, value):
        self.loggers['agent'].info(value)

    def logd(self, tag, value='', level=0):
        lev_val = '-'*level + '>'
        self.loggers['agent_debug'].info("{} {}\t{}".format(lev_val, tag, value))
        
    def get_previous_run_versions(self):
        try:
            files = os.listdir(self.path)
            files = [f for f in files if self.name + '-' in f]
            return files
        except FileNotFoundError as e:
            print("Root path {} doesn't exist.  Creating it...".format(self.path))
            os.makedirs(self.path)
            return []
    
    def save_models(self):
        for model in self.models:
            self.models[model].save_weights(self.model_path + '{}-{}_{}.h5'.format(self.name, self.version, model))
            
    def load_models(self):
        for model in self.models:
            self.models[model].load_weights(self.model_path + '{}-{}_{}.h5'.format(self.name, self.version, model))
