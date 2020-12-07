import h5py
import numpy as np
from tqdm import tqdm

from abc import ABCMeta, abstractmethod
from collections import deque
from ..dataset import Transition, TransitionMiniBatch, trace_back_and_clear
from .utility import get_action_size_from_env


class TransitionQueue:
    """ A queue for transition objects.

    This class is a replacement for deque for Transition objects.
    When the last transition of an episode is removed from the buffer,
    the all links between the transition in the same episode will be cleared to
    make GC properly free transitions objects.

    Args:
        maxlen (int): the maximum size of buffer.

    Attributes:
        maxlen (int): the maximum size of buffer.
        buffer (list): buffer for transitions.
        cursor (int): the current cursor pointing to the position to insert.

    """

    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.buffer = []
        self.cursor = 0

    def append(self, transition):
        """ Appends a transition to buffer.

        Args:
            transition (d3rlpy.dataset.Transition): transition.

        """
        assert isinstance(transition, Transition)
        if self.maxlen is None or self.size() < self.maxlen:
            self.buffer.append(transition)
        else:
            if self.buffer[self.cursor].terminal:
                # clear links to correctly free memories
                trace_back_and_clear(self.buffer[self.cursor])
            self.buffer[self.cursor] = transition
            self.cursor += 1
            if self.cursor == self.maxlen:
                self.cursor = 0

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.buffer[index]

    def __iter__(self):
        return iter(self.buffer)

    def size(self):
        """ Returns the size of buffer.

        Returns:
            int: the size of buffer.

        """
        return len(self.buffer)


class Buffer(metaclass=ABCMeta):
    @abstractmethod
    def append(self, observation, action, reward, terminal):
        """ Append observation, action, reward and terminal flag to buffer.

        If the terminal flag is True, Monte-Carlo returns will be computed with
        an entire episode and the whole transitions will be appended.

        Args:
            observation (numpy.ndarray): observation.
            action (numpy.ndarray or int): action.
            reward (float): reward.
            terminal (bool or float): terminal flag.

        """
        pass

    @abstractmethod
    def append_episode(self, episode):
        """ Append Episode object to buffer.

        Args:
            episode (d3rlpy.dataset.Episode): episode.

        """
        pass

    @abstractmethod
    def sample(self, batch_size, n_frames=1, n_steps=1, gamma=0.99):
        """ Returns sampled mini-batch of transitions.

        If observation is image, you can stack arbitrary frames via
        ``n_frames``.

        .. code-block:: python

            buffer.observation_shape == (3, 84, 84)

            # stack 4 frames
            batch = buffer.sample(batch_size=32, n_frames=4)

            batch.observations.shape == (32, 12, 84, 84)

        Args:
            batch_size (int): mini-batch size.
            n_frames (int):
                the number of frames to stack for image observation.
            n_steps (int): the number of steps before the next observation.
            gamma: discount factor used in N-step return calculation.

        Returns:
            d3rlpy.dataset.TransitionMiniBatch: mini-batch.

        """
        pass

    @abstractmethod
    def size(self):
        """ Returns the number of appended elements in buffer.

        Returns:
            int: the number of elements in buffer.

        """
        pass


class ReplayBuffer(Buffer):
    """ Standard Replay Buffer.

    Args:
        maxlen (int): the maximum number of data length.
        env (gym.Env): gym-like environment to extract shape information.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes to
            initialize buffer

    Attributes:
        prev_observation (numpy.ndarray): previously appended observation.
        prev_action (numpy.ndarray or int): previously appended action.
        prev_reward (float): previously appended reward.
        prev_transition (d3rlpy.dataset.Transition):
            previously appended transition.
        transitions (d3rlpy.online.buffers.TransitionQueue):
            queue of transitions.
        observation_shape (tuple): observation shape.
        action_size (int): action size.

    """

    def __init__(self, maxlen, env, episodes=None):
        # temporary cache to hold transitions for an entire episode
        self.prev_observation = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_transition = None

        self.transitions = TransitionQueue(maxlen=maxlen)

        # extract shape information
        self.observation_shape = env.observation_space.shape
        self.action_size = get_action_size_from_env(env)

        # add initial transitions
        if episodes:
            for episode in episodes:
                self.append_episode(episode)

    def append(self, observation, action, reward, terminal):
        # validation
        assert observation.shape == self.observation_shape
        if isinstance(action, np.ndarray):
            assert action.shape[0] == self.action_size
        else:
            action = int(action)
            assert action < self.action_size

        # create Transition object
        if self.prev_observation is not None:
            if isinstance(terminal, bool):
                terminal = 1.0 if terminal else 0.0

            transition = Transition(observation_shape=self.observation_shape,
                                    action_size=self.action_size,
                                    observation=self.prev_observation,
                                    action=self.prev_action,
                                    reward=self.prev_reward,
                                    next_observation=observation,
                                    next_action=action,
                                    next_reward=reward,
                                    terminal=terminal,
                                    prev_transition=self.prev_transition)

            if self.prev_transition:
                self.prev_transition.next_transition = transition

            self.transitions.append(transition)

            self.prev_transition = transition

        self.prev_observation = observation
        self.prev_action = action
        self.prev_reward = reward

        if terminal:
            self.prev_observation = None
            self.prev_action = None
            self.prev_reward = None
            self.prev_transition = None

    def append_episode(self, episode):
        assert episode.get_observation_shape() == self.observation_shape
        assert episode.get_action_size() == self.action_size
        for transition in episode.transitions:
            self.transitions.append(transition)

    def sample(self, batch_size, n_frames=1, n_steps=1, gamma=0.99):
        indices = np.random.randint(self.size(), size=batch_size)
        transitions = [self.transitions[index] for index in indices]
        return TransitionMiniBatch(transitions, n_frames, n_steps, gamma)

    def size(self):
        return len(self.transitions)

    def __len__(self):
        return self.size()

    def _create_numpy_arrays(self):
        observations = []
        actions = []
        rewards = []
        terminals = []
        ep_terminals = []
        is_new_transition = True
        pbar = tqdm(desc="Saving buffer data", total=len(self.transitions))
        for i, transition in enumerate(self.transitions):
            pbar.update(i)
            if transition.prev_transition is None:
                terminals.append(False)
                ep_terminals.append(False)
            observations.append(transition.observation)
            actions.append(transition.action)
            rewards.append(transition.reward)
            terminals.append(transition.terminal)
            if transition.next_transition is None:
                observations.append(transition.next_observation)
                actions.append(transition.next_action)
                rewards.append(transition.next_reward)
                ep_terminals.append(True)
            else:
                ep_terminals.append(False)
        pbar.close()
        observations = np.stack(observations, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)
        terminals = np.stack(terminals, axis=0)
        ep_terminals = np.stack(ep_terminals, axis=0)
        return observations, actions, rewards, terminals, ep_terminals

    def dump_data(self, fname):
        """ Saves dataset as HDF5.

        Args:
            fname (str): file path.

        """
        observations, actions, rewards, terminals, ep_terminals = self._create_numpy_arrays()
        with h5py.File(fname, 'w') as f:
            f.create_dataset('observations', data=observations)
            f.create_dataset('actions', data=actions)
            f.create_dataset('rewards', data=rewards)
            f.create_dataset('terminals', data=terminals)
            f.create_dataset('ep_terminals', data=ep_terminals)
            f.flush()

    def load_data(self, fname):
        """ Loads dataset from HDF5.

        Args:
            fname (str): file path.

        """
        with h5py.File(fname, 'r') as f:
            observations = f['observations'][()]
            actions = f['actions'][()]
            rewards = f['rewards'][()]
            terminals = f['terminals'][()]
            ep_terminals = f['ep_terminals'][()]

        self._to_transitions(observations, actions, rewards, terminals, ep_terminals)

    def _to_transitions(self, states, actions, rewards, terminals, ep_terminals):
        transitons = []
        num_data = states.shape[0]
        prev_transition = None
        i = 0
        pbar = tqdm(desc="Loading buffer dataset", total=num_data)
        while True:
            pbar.update(i)
            observation = states[i]
            action = actions[i]
            reward = rewards[i]
            next_observation = states[i + 1]
            next_action = actions[i + 1]
            next_reward = rewards[i + 1]
            terminal = terminals[i + 1]
            ep_terminal = ep_terminals[i + 1]
            transition = Transition(observation_shape=observation.shape,
                                    action_size=action.size,
                                    observation=observation,
                                    action=action,
                                    reward=reward,
                                    next_observation=next_observation,
                                    next_action=next_action,
                                    next_reward=next_reward,
                                    terminal=terminal,
                                    prev_transition=prev_transition)

            # set pointer to the next transition
            if prev_transition:
                prev_transition.next_transition = transition

            if ep_terminal:
                prev_transition = None
                i += 2
            else:
                prev_transition = transition
                i += 1

            self.transitions.append(transition)

            if i >= num_data - 1:
                break
        pbar.close()
