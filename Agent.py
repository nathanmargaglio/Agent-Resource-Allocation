import os
import shutil
import time
import logging
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
    
class Agent:
    def __init__(self, name='agent', version=0, path='./runs/', subagents=[], tmp_run=False,
                set_subagent_data=True, save_run=True, versioning=True, verbose=0, *args, **kargs): 
        # Meta Data
        self.kargs = kargs
        self.tmp_run = tmp_run
        self.tmp_path = "{}tmp-0/".format(path)
        self.set_meta_data(name, version, path, subagents, set_subagent_data)
        self.models = {}
        self.loggers = {}
        self.handlers = {}
        self.jobs = {}
        self.sessions = {}
        
        self.log_formatter = logging.Formatter("%(asctime)s %(message)s")
        self.verbose = verbose
        self.save_run = save_run
        self.versioning = versioning
        self.subagent_versioning = self.versioning
        
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth=True
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print('Cleaning Up:', self)
        self.clean_up()
        
    def clean_up(self):
        self.close_session()
        self.close_job()
        self.close_log()
        del self.models
        self.models = {}
    
    def set_meta_data(self, name=None, version=None, path=None, subagents=None, set_subagent_data=True):
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
            
        if self.tmp_run:
            self.name = 'tmp'
            self.version = 0
            
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

    def create_new_run(self):
        # Creates a new run
        # i.e., gives the agent a unique name (through versioning)
        # and creates a new directory for the training run
        if self.versioning is True and not self.tmp_run:
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

        if self.tmp_run:
            try:
                shutil.rmtree(self.tmp_path)
            except OSError as e:
                print("Error: {} - {}.".format(e.filename, e.strerror))
                
        # Make sure our paths exist
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
       
    def training_loop(self):
        pass
       
    def pretraining_step(self):
        pass
    
    def build_models(self):
        pass

    def _training_job(self):
        self.episode = 0
        self.train_step = 0
        self.episode_step = 0
        self.episode_rewards = []
  
        for i, agent in enumerate(self.subagents):
            agent.create_new_run(self.subagent_versioning)
            
        sess = tf.Session(config=self.tf_config)
        try:
            self.create_new_run()

            self.setup_logger('agent', verbose=self.verbose > 0, use_formatter=True)
            self.log('Run Created: {}'.format(self.run_path))
            self.log('Agen logs created.')

            self.setup_logger('agent_debug', verbose=self.verbose > 1, use_formatter=True)
            self.logd('Debug logs created.')

            self.log('Running Pretraining Step.')
            self.pretraining_step()
            self.log('Building Models.')
            self.build_models()
            self.log('Beginning Training Loop.')
            self.training_loop()
            sess.close()
            self.log('Training Complete!')
        except Exception as e:
            sess.close()
            self.log('Error!')
            self.log("{}".format(e))
            raise e
        
    def run(self, episodes, multiprocess=True):
        self.max_episodes = episodes
        self._training_job()
      
    def setup_logger(self, tag, sub_path=None, verbose=False, level=logging.INFO,
                     close_previous_version=True, use_formatter=False):
        
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
        
    def close_log(self, name=None):
        if name is None:
            names = list(self.loggers.keys())[:]
            for n in names:
                self.loggers.pop(n)
                self.handlers.pop(n).close()
        else:
            if name in self.loggers.keys():
                self.loggers.pop(name)
                self.handlers.pop(name).close()
 
    def close_job(self, name=None):
        if name is None:
            names = list(self.jobs.keys())[:]
            for n in names:
                job = self.jobs.pop(n)
                job.terminate()
                job.join()
        else:
            if name in self.jobs.keys():
                job = self.jobs.pop(name)
                job.terminate()
                job.join()
    
    def log_scalar(self, tag, value, step, sub_path=None):
        logger = self.fetch_or_create_logger(tag, sub_path)
        logger.info("{},{},{}".format(step, time.time(), value))
        
    def log_ndarray(self, tag, array, step, sub_path=None):
        array = np.array(array)
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
            os.makedirs(self.path, exist_ok=True)
            return []
    
    def save_models(self):
        for model in self.models:
            self.models[model].save_weights(self.model_path + '{}-{}_{}.h5'.format(self.name, self.version, model))
            
    def load_models(self):
        for model in self.models:
            self.models[model].load_weights(self.model_path + '{}-{}_{}.h5'.format(self.name, self.version, model))
