"""Soft actor-critic algorithm.
https://arxiv.org/abs/1801.01290
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .model import GaussianPolicy, QNetwork
from .utils import soft_update, hard_update
from ..agent import Agent



class SAC(Agent):
    _ATTRIBUTES_TO_SAVE = [
        'gamma', 'tau',
        'target_update_interval', 'automatic_entropy_tuning',
        'critic', 'critic_optim', 'critic_target',
        'policy', 'policy_optim',
        'target_entropy', 'alpha', 'log_alpha', 'alpha_optim',
    ]

    def __init__(
            self, observation_space, action_space, device, gamma=0.99, tau=0.005, alpha=0.2, hidden_size=256,
            target_update_interval=1, automatic_entropy_tuning=True, learning_rate=3.0e-4):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device(device) 

        self.critic = QNetwork(observation_space.shape[0], action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = QNetwork(observation_space.shape[0], action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=learning_rate)
        else:
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None

        self.policy = GaussianPolicy(observation_space.shape[0], action_space.shape[0], hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        self._observation_shape = observation_space.shape
        self._action_shape = action_space.shape

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_data, total_updates):
        """
        Params
        ---
        batch_data:
            state_batch
            action_batch
            reward_batch
            next_state_batch
            term_batch: 1 if episode terminate
        """
        state_batch, action_batch, reward_batch, next_state_batch, term_batch = batch_data

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        term_batch = torch.FloatTensor(term_batch).to(self.device)

        batch_size = state_batch.size(0)
        assert state_batch.size() == torch.Size([batch_size, *self._observation_shape]), state_batch.size()
        assert next_state_batch.size() == torch.Size([batch_size, *self._observation_shape]), next_state_batch.size()
        assert action_batch.size() == torch.Size([batch_size, *self._action_shape]), action_batch.size()
        assert reward_batch.size() == torch.Size([batch_size, 1]), reward_batch.size()
        assert term_batch.size() == torch.Size([batch_size, 1]), term_batch.size()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - term_batch) * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if total_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def state_dict(self):
        return {
            k: getattr(self, k).state_dict() if hasattr(getattr(self, k), 'state_dict') else getattr(self, k)
            for k in self._ATTRIBUTES_TO_SAVE
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)         
    