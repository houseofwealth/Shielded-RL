from copy import deepcopy
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
# from ipdb import set_trace

# ---------------------------------------------------------------------------
# ActionSelector: abstract base class shared by all action selectors.
# ---------------------------------------------------------------------------
class ActionSelector:
    def __init__(self, config, policy, env):
        self.env          = env.envs[0]   # NOTE: assumes VecEnv with exactly 1 env inside
        self.policy       = policy
        self.num_chances  = config['num_shield_chances']
        self.config       = config
        self.n_agent_fails = 0

    def getActionForEachAgent(self, single_obs):
        raise NotImplementedError('Subclasses must implement getActionForEachAgent')

    '''obs is a dict of key→tensor pairs.  Returns a new dict where every
    tensor is repeated num_chances times along dim 0.'''
    def replicateObsNumChancesTimes(self, obs):
        if not isinstance(obs, dict):
            raise NotImplementedError('Can only handle obs that are dicts!')
        return {key: value.repeat(self.num_chances, 1) for key, value in obs.items()}


# WaterTankActionSelector moved to WaterTankActionSelector.py
# class WaterTankActionSelector(ActionSelector): ...
