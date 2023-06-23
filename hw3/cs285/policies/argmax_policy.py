import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3: # adds batch dimension if needed
            observation = obs
        else:
            observation = obs[None]
        
        ## return the action that maxinmizes the Q-value 
        # at the current observation as the output

        qvals = self.critic.qa_values(observation)
        action = np.argmax(qvals, axis=1)
        return action.squeeze()