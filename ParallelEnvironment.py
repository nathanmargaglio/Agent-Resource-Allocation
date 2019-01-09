from multiprocessing import Process, Pipe
import pickle
import cloudpickle
import numpy as np

# Adapted from Stable Baselines
# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/vec_env/subproc_vec_env.py

class CloudpickleWrapper(object):
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break

class ParallelEnvironment:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.processes = [Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                          for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for process in self.processes:
            process.daemon = True  # if the main process crashes, we should not cause things to hang
            process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        self.num_envs = len(env_fns)
        self.observation_space = observation_space
        self.action_space = action_space

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True
        
    def env_method(self, method_name, *method_args, **method_kwargs):
        for remote in self.remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in self.remotes]

    def get_attr(self, attr_name):
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(len(self.remotes))
        elif isinstance(indices, int):
            indices = [indices]
        for remote in [self.remotes[i] for i in indices]:
            remote.send(('set_attr', (attr_name, value)))
        return [remote.recv() for remote in [self.remotes[i] for i in indices]]