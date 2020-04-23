import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np

from ..agent import Agent
from .model import BijectiveTransform, QNetwork, Policy
from .utils import soft_update, hard_update



class LSPHierarchicalRL(Agent):
    """Latent Space Polices for Hierarchical Reinforcement Learning.
    """
    _ATTRIBUTES_TO_SAVE = [
        "_subpolicies", "_policy_optim", 
        "_critic", "_critic_target", "_critic_optim",
        "_log_alpha", "_alpha_optim",
    ]

    def __init__(self, observation_space, action_space, subpolicy_coupling_layer_num=2,
            prior=None, gamma=0.99, target_smoothing_rate=0.01, alpha=0.2, target_update_interval=1,
            critic_hidden_layer_num=2, critic_hidden_layer_size=128, learning_rate=3.0*10e-4, device='cpu'):
        """
        Params
        ---
        reward_funcs: List of reward functions
        """
        self._device = torch.device(device)
        self._observation_size = observation_space.shape[0]
        self._action_size = action_space.shape[0]
        self._gamma = gamma
        self._init_alpha = alpha
        self._target_smoothing_rate = target_smoothing_rate
        self._target_update_interval = target_update_interval
        self._learning_rate = learning_rate

        self._prior = prior if prior is not None else MultivariateNormal(torch.zeros(self._action_size), torch.eye(self._action_size))

        self._subpolicy_coupling_layer_num = subpolicy_coupling_layer_num

        self._critic_hidden_layer_num = critic_hidden_layer_num
        self._critic_hidden_layer_size = critic_hidden_layer_size

        # Prepare a policy
        subpolicy = Policy(self._observation_size, self._action_size, 2, 128, action_low=action_space.low, action_high=action_space.high).to(self._device)
        self._subpolicies = [subpolicy]
        self._policy_optim = Adam(self._subpolicies[-1].parameters(), lr=self._learning_rate)

        # Prepare critic and its target
        self._critic = QNetwork(self._observation_size, self._action_size, critic_hidden_layer_num, critic_hidden_layer_size).to(self._device)
        self._critic_optim = Adam(self._critic.parameters(), lr=self._learning_rate)
        self._critic_target = QNetwork(self._observation_size, self._action_size, critic_hidden_layer_num, critic_hidden_layer_size).to(self._device)
        hard_update(self._critic_target, self._critic)

        # Prepare alpha
        self._alpha = self._init_alpha
        self._target_entropy = - self._action_size
        self._log_alpha = torch.ones(1, requires_grad=True, device=self._device)
        self._alpha_optim = Adam([self._log_alpha], lr=self._learning_rate)

    def select_action(self, obs: np.ndarray, eval=False, random=False):
        """Select an abstract action using a subpolicy in the higher hierarchy based on observation.
        post_process_action method should be called before passing the returned action to an environment.
        """
        assert obs.shape == (self._observation_size,), "expected {}, got {}".format((self._observation_size,), obs.shape)
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            h, _ = self._select_latent_and_log_prob(obs, eval=eval, skip_subpolicy=random)
        return h.squeeze(0).detach().cpu().numpy()

    def post_process_action(self, obs: np.ndarray, h: np.ndarray):
        """Get an actual action from a latent variable.
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
        h = torch.FloatTensor(h).unsqueeze(0).to(self._device)
        with torch.no_grad():
            for sp in reversed(self._subpolicies[:-1]):
                h, _ = sp(h, obs)  # deterministic calculation
        return h.squeeze(0).detach().cpu().numpy()
        
    def _select_latent_and_log_prob(self, obs: torch.FloatTensor, eval=False, skip_subpolicy=False):
        """Select latent variable using a subpolicy being trained and return it with its log prob.
        Params
        ---
        obs: ([batch_size, observation_size])

        Return
        ---
        h_l: ([batch_size, action_size])
        log_p: ([batch_size])
        """
        N = obs.size(0)
        assert obs.size() == torch.Size([N, self._observation_size])

        # Sample latent variable
        if not eval:
            hh = self._prior.sample((N, ))
            log_p_hh = self._prior.log_prob(hh).to(self._device)
            hh = hh.to(self._device)
        else:
            hh = torch.zeros(N, self._action_size).to(self._device)  # deterministic if eval
            log_p_hh = 0

        if skip_subpolicy:
            return hh, log_p_hh

        # Select action in latent space
        h, log_det_J = self._subpolicies[-1](hh, obs)  # h <- sp(hh, obs)
        log_p = log_p_hh - log_det_J  # log p(h) = log p(hh) + log det d(sp^{-1})/dh = log p(hh) - log det d(sp)/d(hh)
        return h, log_p

    def update_parameters(self, batch_data, total_updates):
        """Update parameters in a subpolicy and Q-net using soft actor-critic algorithm.

        Return
        ---
        qf1_loss, qf2_loss, policy_loss, alpha_loss, self._alpha
        """
        observations, hs, rewards, next_observations, terminations = batch_data
        observations = torch.FloatTensor(observations).to(self._device)
        hs = torch.FloatTensor(hs).to(self._device)
        rewards = torch.FloatTensor(rewards).to(self._device)
        next_observations = torch.FloatTensor(next_observations).to(self._device)
        terminations = torch.FloatTensor(terminations).to(self._device)

        with torch.no_grad():
            next_obs_hs, next_obs_hs_log_ps = self._select_latent_and_log_prob(next_observations)
            qf1_next_target, qf2_next_target = self._critic_target(next_observations, next_obs_hs)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_obs_hs_log_ps.unsqueeze(1)
            next_q_values = rewards + (1 - terminations) * self._gamma * (min_qf_next_target)

        # Calculate critic loss
        qf1, qf2 = self._critic(observations, hs)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_values) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_values) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        # Calculate subpolicy loss
        hs_, log_ps_ = self._select_latent_and_log_prob(observations)
        qf1_pi, qf2_pi = self._critic(observations, hs_)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = - (min_qf_pi - self._alpha * log_ps_).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        print("avg. min_qf_pi", min_qf_pi.mean().item(), "avg. log ps", log_ps_.mean().item())

        self._critic_optim.zero_grad()
        qf1_loss.backward()
        self._critic_optim.step()

        self._critic_optim.zero_grad()
        qf2_loss.backward()
        self._critic_optim.step()

        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        # Calculate alpha loss
        alpha_loss = (self._log_alpha * (- log_ps_.detach() - self._target_entropy)).mean()

        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        self._alpha = self._log_alpha.exp().detach().item()

        if total_updates % self._target_update_interval == 0:
            soft_update(self._critic_target, self._critic, self._target_smoothing_rate)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), self._alpha

    def insert_subpolicy(self):
        """Fix current subpolicies and insert a new higher subpolicy.
        A critic, its target, alpha_log, and their optimizers are also refreshed.
        """
        # Fix parameters
        for p in self._subpolicies[-1].parameters():
            p.requires_grad_(False)

        # Insert a new subpolicy
        new_subpolicy = Policy(self._observation_size, self._action_size, 2, 128).to(self._device)
        self._subpolicies.append(new_subpolicy)
        self._policy_optim = Adam(self._subpolicies[-1].parameters(), lr=self._learning_rate)

        # Prepare critic and its target
        self._critic = QNetwork(self._observation_size, self._action_size, self._critic_hidden_layer_num, self._critic_hidden_layer_size).to(self._device)
        self._critic_optim = Adam(self._critic.parameters(), lr=self._learning_rate)
        self._critic_target = QNetwork(self._observation_size, self._action_size, self._critic_hidden_layer_num, self._critic_hidden_layer_size).to(self._device)
        hard_update(self._critic_target, self._critic)

        # Prepare alpha
        self._alpha = self._init_alpha
        self._log_alpha = torch.ones(1, requires_grad=True, device=self._device)
        self._alpha_optim = Adam([self._log_alpha], lr=self._learning_rate)

    def state_dict(self):
        state_dict = {}
        for k in self._ATTRIBUTES_TO_SAVE:
            if k == "_subpolicies":
                state_dict[k] = [sp.state_dict() for sp in getattr(self, "_subpolicies")]
            elif hasattr(getattr(self, k), 'state_dict'):
                state_dict[k] = getattr(self, k).state_dict()
            else:
                state_dict[k] = getattr(self, k)
        return state_dict

    def load_state_dict(self, state_dict):
        self._subpolicies = []
        for p_state_dict in state_dict['_subpolicies']:
            p = Policy(self._observation_size, self._action_size, 2, 128).to(self._device)
            p.load_state_dict(p_state_dict)
            self._subpolicies.append(p)
        self._policy_optim = Adam(self._subpolicies[-1].parameters(), lr=self._learning_rate)    
        self._policy_optim.load_state_dict(state_dict['_policy_optim'])

        for k, v in state_dict.items():
            if k == "_subpolicies" or k == "_policy_optim":
                pass
            elif hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)         
    