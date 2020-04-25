import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np


EPSILON = 1e-5


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class NN(nn.Module):
    """Fully connencted neural network.
    """
    def __init__(self, in_size, out_size, hidden_layer_num, hidden_layer_size, activation_func, output_func=None, learn_output_scale=False):
        super().__init__()
        self._layers = [nn.Linear(in_size, hidden_layer_size)]
        self._layers += [nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(hidden_layer_num-1)] 
        self._layers += [nn.Linear(hidden_layer_size, out_size)]
        self._layers = nn.ModuleList(self._layers)

        self._activation_func = activation_func
        self._output_func = output_func
        self._output_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True) if learn_output_scale else None

        self.apply(weights_init_)

    def forward(self, x):
        for l in self._layers[:-1]:
            x = self._activation_func(l(x))
        x = self._layers[-1](x)
        if self._output_func:
            x = self._output_func(x)
        if self._output_scale:
            x = x * self._output_scale
        return x
    

class BijectiveTransform(nn.Module):
    """Implementation fo bijective transformation in Real NVP.
    """
    def __init__(self, v_size, layer_num, scale_net_hidden_layer_num=1, scale_net_hidden_layer_size=256, 
            translate_net_hidden_layer_num=1, translate_net_hidden_layer_size=256, condition_vector_size=0):
        """
        Parames
        ---
        condition_vector_size: size of an additional vector which is concatenated to hidden variables.
            z = f([x ; cond])
        """
        super().__init__()
        self._v_size = v_size
        self._condition_vector_size = condition_vector_size
        self._layer_num = layer_num
        m = torch.cat([torch.ones(v_size//2), torch.zeros(v_size - v_size//2)])
        self.register_buffer("_masks", torch.stack([m.clone() if i%2==0 else 1. - m.clone() for i in range(self._layer_num)]))
        self._s = nn.ModuleList([NN(v_size + condition_vector_size, v_size, scale_net_hidden_layer_num, scale_net_hidden_layer_size, torch.relu, torch.tanh, True) for _ in range(layer_num)])
        self._t = nn.ModuleList([NN(v_size + condition_vector_size, v_size, translate_net_hidden_layer_num, translate_net_hidden_layer_size, torch.relu) for _ in range(layer_num)])
        self._prior = MultivariateNormal(torch.zeros(v_size), torch.eye(v_size))  # N(0, 1)

    def _calc_log_determinant(self, s):
        """log det(diag( exp( s(x_{1:d}) ) )"""
        return s.sum(dim=1)

    def infer(self, x, cond=None):
        """Inference z = f(x)

        Return
        ---
        z, log_det_J
        where log_det_J = log det(\frac{\partial f}{\partial x})
        """
        batch_size = x.size(0)
        assert x.size() == torch.Size([batch_size, self._v_size]), x.size()
        if self._condition_vector_size > 0:
            cond.size() == torch.Size([batch_size, self._condition_vector_size]), cond.size()

        log_det_J = 0
        z = x
        for i in range(self._layer_num):
            mask = self._masks[i]
            z_ = torch.cat([mask * z, cond], dim=1) if cond is not None else mask * z
            s = self._s[i](z_) * (1. - mask)
            t = self._t[i](z_) * (1. - mask)
            z = mask * z + (1. - mask) * z * s.exp() + t
            log_det_J += self._calc_log_determinant(s)
        return z, log_det_J

    def generate(self, z, cond=None):
        """Generation x = f^-1(z)

        Return
        ---
        x, log_det_inv_J
        where log_det_inv_J = log det(\frac{\partial f^{-1}}{\partial z})
        """
        batch_size = z.size(0)
        assert z.size() == torch.Size([batch_size, self._v_size]), z.size()
        if self._condition_vector_size > 0:
            cond.size() == torch.Size([batch_size, self._condition_vector_size]), cond.size()

        log_det_inv_J = 0
        x = z
        for i in reversed(range(self._layer_num)):
            mask = self._masks[i]
            x_ = torch.cat([mask * x, cond], dim=1) if cond is not None else mask * x
            s = self._s[i](x_) * (1. - mask)
            t = self._t[i](x_) * (1. - mask)
            x = mask * x + ((1. - mask) * x - t) * (-s).exp()
            log_det_inv_J += self._calc_log_determinant(-s)
        return x, log_det_inv_J

    def calc_log_likelihood(self, x_batch, cond_batch=None):
        """Maxmiize log p(x) = log p(f(x)) + log | det(\frac{\partial f}{\partial x}) |
        """
        batch_size = x_batch.size(0)
        assert x_batch.size() == torch.Size([batch_size, self._v_size])
        if self._condition_vector_size > 0:
            cond_batch.size() == torch.Size([batch_size, self._condition_vector_size]), cond_batch.size()

        z, log_det_J = self.infer(x_batch, cond_batch)
        log_pz = self._prior.log_prob(z)
        log_px = (log_pz + log_det_J).mean()
        return log_px

    def sample(self, sample_size, cond=None):
        if self._condition_vector_size > 0:
            cond.size() == torch.Size([sample_size, self._condition_vector_size]), cond.size()
        z = self._prior.sample_n(sample_size)
        x, _ = self.generate(z, cond)
        return x.detach()


class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_layer_num=2, hidden_layer_size=128):
        super().__init__()
        self._qnet_1 = NN(obs_size + action_size, 1, hidden_layer_num, hidden_layer_size, torch.relu)
        self._qnet_2 = NN(obs_size + action_size, 1, hidden_layer_num, hidden_layer_size, torch.relu)
        self.apply(weights_init_)

    def forward(self, obs, action):
        xu = torch.cat([obs, action], dim=1)
        q1 = self._qnet_1(xu)
        q2 = self._qnet_2(xu)
        return q1, q2


class Policy(nn.Module):
    def __init__(self, obs_size, action_size, obs_embedding_hidden_layer_num=2, obs_embedding_hidden_layer_size=128, action_low=None, action_high=None):
        super().__init__()
        self._observation_size = obs_size
        self._action_size = action_size
        self._action_low, self._action_high = action_low, action_high

        if self._action_high is not None:
            self._action_scaler = Scaler(self._action_low, self._action_high)
        else:
            # Limit to default std
            # DEFAULT_STD = 5.0
            # self._action_scaler = Scaler(np.ones(self._action_size) * -DEFAULT_STD, np.ones(self._action_size) * DEFAULT_STD)
            self._action_scaler = None

        obs_embedding_size = 2 * action_size
        self._f = BijectiveTransform(action_size, 2, 1, action_size, 1, action_size, obs_embedding_size) # following the configurations in the original paper
        self._embed_nn = NN(obs_size, obs_embedding_size, obs_embedding_hidden_layer_num, obs_embedding_hidden_layer_size, torch.relu)

    def forward(self, hh, obs):
        """Return lower latent variable and log_det_J.
        h <- f(hh, obs)
        log p(h)  = log p(hh) - log_det_J
        log_det_J = log det df/dhh = - log det df^{-1}/dh
        """
        N = obs.size(0)
        assert hh.size() == torch.Size([N, self._action_size]), "expected {}, got {}".format((N, self._action_size), hh.size())
        assert obs.size() == torch.Size([N, self._observation_size]), "expected {}, got {}".format((N, self._observation_size), obs.size())

        log_det = 0
        obs_embedding = self._embed_nn(obs)
        h, log_det_inv_J = self._f.generate(hh, obs_embedding)  # h' <- f^{-1}(h)
        log_det += log_det_inv_J
        # Convert range if needed
        if self._action_scaler:
            h, log_det_scale = self._action_scaler(h)  # h <- scale(h')
            log_det += log_det_scale

        return h, log_det

    def inverse(self, h, obs):
        """Just for check.
        """
        N = obs.size(0)
        assert obs.size() == torch.Size([N, self._observation_size]), obs.size()
        assert h.size() == torch.Size([N, self._action_size]), hh.size()

        log_det = 0
        if self._action_scaler:
            h, log_det_inv_scale = self._action_scaler.inverse(h)
            log_det += (- log_det_inv_scale)  # u <- f^{-1}(a), p(u) = p(a) det df/du = p(u) ( det df^{-1}/da )^{-1}

        obs_embedding = self._embed_nn(obs)
        hh, log_det_J = self._f.infer(h, obs_embedding)
        log_det += (- log_det_J)  # h <- f(a), p(h) = p(a) det df^{-1}/dh = p(a) ( det df/da )^{-1}
        return hh, log_det



class Scaler(nn.Module):
    """Convert unlimited action into a variable with range.
    a = f(u) where $u \in [-inf, inf]$ -> $a \in [low, high]$
    """
    def __init__(self, action_space_low: np.ndarray, action_space_high: np.ndarray):
        super().__init__()
        # a = w tanh(u) + m
        self.register_buffer("_m", torch.FloatTensor((action_space_low + action_space_high) / 2.))
        self.register_buffer("_w", torch.FloatTensor((action_space_high - action_space_low) / 2.))
        assert (self._w >= 0).all(), self._w

    def forward(self, x):
        """Convert x -> a
        Return
        ----
        a
        log_det_J = log det diag{ w * (1 - tanh^2(x)) }
        """
        u = torch.tanh(x)
        a = self._m + self._w * u
        log_det_J = torch.log(self._w * (1. - u.pow(2)) + EPSILON).sum(dim=1)
        return a, log_det_J

    def inverse(self, a):
        """Convert a -> x
        Return
        ---
        a
        log_det_inv_J = - log det diag{ w * (1 - tanh^2(x)) }
        """
        u = (a - self._m) / self._w
        x = _arctanh(u)
        log_det_inv_J = - torch.log(self._w * (1. - u.pow(2)) + EPSILON).sum(dim=1)
        return x, log_det_inv_J



def _arctanh(y):
    assert -1 < y < 1, y
    return .5 * (torch.log(y+1) - torch.log(1-y))
        