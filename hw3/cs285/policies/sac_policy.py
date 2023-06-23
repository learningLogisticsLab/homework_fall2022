from cs285.policies.MLP_policy import MLPPolicy
from torch import distributions
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

from cs285.infrastructure.sac_utils import SquashedNormal

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return self.log_alpha.exp()

    def get_action(self, obs, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]


        dis = self.forward(obs)
        action = None
        if sample:
            action = ptu.to_numpy(dis.sample())
        else:
            action =  ptu.to_numpy(dis.mean)
        return action
    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            mean = self.mean_net(observation)
      
            std = torch.exp(torch.clamp(self.logstd, self.log_std_bounds[0], self.log_std_bounds[1]))

            action_distribution =  SquashedNormal(mean, std)
            return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        obs = ptu.from_numpy(obs)

        action = self.get_action(obs)
        dis = self.forward(obs)
        action = dis.sample()


        q1, q2 = critic(obs, action)
        q = torch.min(q1, q2)

        log_probs = dis.log_prob(action).sum(axis=1)
 
        loss = -torch.mean(q.detach() - self.alpha.detach() * log_probs)
        


        # update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for p in self.mean_net.parameters():
            if torch.isnan(p).any():
                print("mean net after update  has nan")
        
        # update temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.log_alpha * (-log_probs.detach() + self.target_entropy).mean())
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return loss.item(), alpha_loss, self.alpha