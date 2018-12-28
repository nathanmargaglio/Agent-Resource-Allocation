import os
import time
import logging
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, RemoteMonitor, CSVLogger
import tensorflow as tf

class Agent:
    def __init__(self):
        self.log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.log_formatter)
        self.console_handler.setLevel(logging.INFO)
        self.root_logger.addHandler(self.console_handler)
        self.set_meta_data()

    def set_meta_data(self, name=None, path='./runs/'):
        if path[-1] != '/':
            path.append('/')
        if name is None:
            name = 'run'
        self.name = name
        self.path = path
        self.run_path = None
        self.callbacks = []
        self.writer = None
        self.root_logger.handlers = []
        self.root_logger.addHandler(self.console_handler)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,simple_value=value)])
        self.writer.add_summary(summary, step)
        self.root_logger.debug("{} - {} - {}".format(tag, value, step))

    def log(self, value):
        self.root_logger.info(value)

    def get_previous_models(self, name=None, path='./runs/'):
        try:
            files = os.listdir(path)
            if name:
                files = [f for f in files if name + '-' in f]
            return files
        except FileNotFoundError as e:
            print("{} doesn't exist.  Creating it...")
            os.makedirs(path)
            return []

    def create_run_dir(self):
        extra_version = 0
        while self.run_path is None or os.path.exists(self.run_path):
            self.version = len(self.get_previous_models(self.name, self.path)) + extra_version
            self.run_path = "{}{}-{}/".format(self.path, self.name, self.version)
            extra_version += 1

        os.makedirs(self.run_path)

        self.callbacks = self.create_callbacks()
        self.writer = tf.summary.FileWriter(self.run_path)
        self.file_handler = logging.FileHandler(self.run_path + "log.txt")
        self.file_handler.setFormatter(self.log_formatter)
        self.file_handler.setLevel(logging.INFO)
        self.root_logger.addHandler(self.file_handler)
        self.root_logger.info('Run Logs Created')

    def get_callbacks(self):
        return self.callbacks

    def create_callbacks(self):
        callbacks = []

        #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
        #callbacks.append(TensorBoard(self.run_path))

        #keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        #callbacks.append(ModelCheckpoint(model_path + 'checkpoint'))

        #keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        #callbacks.append(EarlyStopping(monitor='', patience=25))

        #keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
        #callbacks.append(RemoteMonitor())

        #keras.callbacks.CSVLogger(filename, separator=',', append=False)
        #callbacks.append(CSVLogger(self.run_path + 'log.csv'))

        return callbacks
