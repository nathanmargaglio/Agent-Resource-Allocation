import os
import time
import logging
from io import BytesIO
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, RemoteMonitor, CSVLogger
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import numpy as np

class Agent:
    def __init__(self):
        self.log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.log_formatter)
        self.console_handler.setLevel(logging.INFO)
        self.root_logger.addHandler(self.console_handler)

        self.name = None
        self.version = 0
        self.path = './runs/'
        self.save_run = True
        self.save_images = True
        self.save_animations = True

    def set_meta_data(self, name=None, version=None, path=None):
        if path is not None:
            if path[-1] != '/':
                path.append('/')
            self.path = path

        if name is not None:
            self.name = name

        if version is not None:
            self.version = version

        self.run_path = "{}{}-{}/".format(self.path, self.name, self.version)
        self.model_path = self.run_path + 'models/'
        self.tb_path = self.run_path + 'tb/'
        self.animation_path = self.run_path + 'animations/'
        for agent in self.agents:
            agent.set_path(self.run_path)

        self.callbacks = []
        self.writer = None
        self.image_writer = None
        self.root_logger.handlers = []
        self.root_logger.addHandler(self.console_handler)

    def log_scalar(self, tag, value, step, writer='meta'):
        if self.save_run:
            writers = {
                    'meta': self.writer,
                    'uni': self.uniform_writer,
                    'rand': self.random_writer
                    }
            for n, w in enumerate(self.sub_writers):
                writers['sub_{}'.format(n)] = w
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,simple_value=value)])
            writers[writer].add_summary(summary, step)

        self.root_logger.info("{} \t {} [{}] \t {}".format(writer, tag, step, value))

    def log(self, value):
        self.root_logger.info(value)

    def log_image(self, tag, image, step):
        if self.save_animations:
            frame_dir = self.animation_path + '/' + tag
            if not os.path.isdir(frame_dir):
                os.makedirs(frame_dir)
                self.frame_count[frame_dir] = 0
            count = self.frame_count[frame_dir]
            plt.imsave(frame_dir + '/{0:03d}.png'.format(count), image)
            self.frame_count[frame_dir] += 1

        if self.save_images:
            s = BytesIO()
            plt.imsave(s, image, format='png')
            img_sum = tf.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=image.shape[0],
                    width=image.shape[1]
                    )
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
            self.image_writer.add_summary(summary, step)

    def close_frame_writers(self):
        for key in self.frame_writers:
            self.frame_writers[key].close()

    def log_plot(self, tag, fig, step):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.log_image(tag, data, step)

    def get_previous_models(self, name=None, path='./runs/'):
        try:
            files = os.listdir(path)
            if name:
                files = [f for f in files if name + '-' in f]
            return files
        except FileNotFoundError as e:
            print("{} doesn't exist.  Creating it...".format(path))
            os.makedirs(path)
            return []

    def create_new_run(self, name='run', path='./runs/'):
        self.name = name
        self.path = path

        extra_version = 0
        print('Creating log directory...')
        while True:
            try:
                self.version = len(self.get_previous_models(self.name, self.path)) + extra_version
                self.run_path = "{}{}-{}/".format(self.path, self.name, self.version)
                os.makedirs(self.run_path)
                break
            except FileExistsError as e:
                extra_version += 1
                if extra_version % 100 == 0:
                    print('\n',e)
            if extra_version % 10 == 0:
                print('.', end=' ')
            if extra_version > 999:
                raise StopIteration('Exceeded 999 runs of the name {}'.format(self.name))

        self.set_meta_data()

        os.makedirs(self.model_path)
        if self.save_animations:
            os.makedirs(self.animation_path)
            self.frame_writers = {}
            self.frame_count = {}

        for i, agent in enumerate(self.agents):
            agent.set_path(self.run_path)
            agent.set_index(i)

        self.callbacks = self.create_callbacks()
        self.writer = tf.summary.FileWriter(self.tb_path + 'meta', filename_suffix='_meta')
        self.uniform_writer = tf.summary.FileWriter(self.tb_path + 'uniform', filename_suffix='_uniform')
        self.random_writer = tf.summary.FileWriter(self.tb_path + 'random', filename_suffix='_random')
        self.sub_writers = []
        for i in range(self.agent_count):
            self.sub_writers.append(tf.summary.FileWriter(self.tb_path + 'sub_{}'.format(i), filename_suffix='_sub'))
        self.image_writer = tf.summary.FileWriter(self.tb_path + 'images', filename_suffix='_images')

        self.file_handler = logging.FileHandler(self.run_path + "log.txt")
        self.file_handler.setFormatter(self.log_formatter)
        self.file_handler.setLevel(logging.INFO)
        self.root_logger.addHandler(self.file_handler)
        self.root_logger.info('Run Logs Created: {}'.format(self.run_path))

    def get_callbacks(self):
        return self.callbacks

    def create_callbacks(self):
        callbacks = []
        return callbacks

    def cleanup(self):
        self.close_frame_writers()
