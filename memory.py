import numpy as np


class ReplayBuffer:
    """Save (o_t, a_t, o_t+1, r_t).
    """
    _ATTRIBUTES_TO_SAVE = [
        '_observation_size', '_action_size',
        '_observations', '_actions', '_next_observations', '_rewards', '_terminates',
        '_idx', 'step', 'capacity', 'full',
    ]

    def __init__(self, capacity: int, observation_size: tuple, action_size: tuple):
        self._observation_size = observation_size
        self._action_size = action_size

        self._observations = np.empty((capacity, *observation_size), dtype=np.float32)
        self._next_observations = np.empty((capacity, *observation_size), dtype=np.float32)
        self._actions = np.empty((capacity, *action_size), dtype=np.float32)
        self._terminates = np.empty((capacity, 1), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)

        self._idx = 0
        self.step = 0
        self.capacity = capacity
        self.full = False

    def reset(self):
        self._idx = 0
        self.step = 0
        self.full = False

    def push(self, o: np.ndarray, a: np.ndarray, r: float, o_next: np.ndarray, t: bool):
        assert o.shape == self._observation_size, o.shape
        assert a.shape == self._action_size, a.shape
        assert o_next.shape == self._observation_size, o_next.shape
        assert isinstance(r, float), r
        assert isinstance(t, bool), t

        np.copyto(self._observations[self._idx], o)
        np.copyto(self._next_observations[self._idx], o_next)
        np.copyto(self._actions[self._idx], a)
        self._rewards[self._idx][0] = r
        self._terminates[self._idx][0] = float(t)

        self._idx += 1
        self.step += 1
        self.full |= (self._idx >= self.capacity)
        self._idx %= self.capacity

    def __len__(self):
        return self.capacity if self.full else self._idx

    def sample(self, batch_size, replace=True):
        """Return a tuple of batch data.
        batch_s, batch_a, batch_r, batch_s', batch_t
        """
        assert batch_size <= len(self), "specified bath size {} is larger than replay buffer size {}".format(batch_size, len(self))
        indexes = np.random.choice(len(self), batch_size, replace=replace)
        observations = self._observations[indexes]
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        next_observations = self._next_observations[indexes]
        terminates = self._terminates[indexes]
        return observations, actions, rewards, next_observations, terminates

    def state_dict(self):
        return {k: getattr(self, k) for k in self._ATTRIBUTES_TO_SAVE}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)         
    