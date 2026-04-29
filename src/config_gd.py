from GameOfDronesEnv import GameOfDronesEnv
#below is what learning alg to use and which kind of policy it uses
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy #MIP is alias for MultiInputActorCriticPolicy
# from ray import tune
import numpy as np

# USE_PARALLEL = False
# AGENT_SIZE = 0
# print('AGENT_SIZE', AGENT_SIZE) 
# STEP_SIZE = 1        
#                              
""" 20% of worksapce size"""
# LObs = -4
# RObs = 4
# CObs = -4
# FObs = 4
# BObs = 3
# TObs = 8

DEFAULT_CONFIG = {
    'policy_class': MultiInputPolicy,
    'env_class': GameOfDronesEnv,
    'action_selector_class': None,  # set below after DEFAULT_CONFIG to avoid circular import

    # ---hyperparams---
    'learning_rate': 3e-4,  #feb experiments with 5e-5,     #the higher, the faster the learning but can be unstable or overshoot the optimal soln
    # 'batch_size': was this higher? But im sticking with PPO default of 64
    'gamma': 0.99,                        #discount factor for future rewards
    'ent_coef': 0,   #Feb experiments used 0.05, #was 0 then 0.1 (worsened) It prevents the policy from changing too drastically in one update, ensuring stable learning
    #'target_kl': 0.05,                   
    # 'max_episode_length': 100,
    'max_learning_steps': int(500E3),     #total for trining run
    'buffer_size': int(30E3),             #rollout buffer size

    # ---environment setup---
    #PRED_SIZE for determining if a collision happened
    #MIN_SEP is a stronger constraint that ensures a min sep is mainted beteween preds eg drones shouldnt get too close
    #KILL_RADIUS is the distance at which a predator is considered to have caught the prey.

    'env_config': {        
        'use_shield': True,
        'num_shield_chances': 100,

        'initial_prey_pos' : np.array([0, 10.0]),         #chg for 3-D, also Other option is 'random'
        'initial_prey_velocity': np.array([0.0, -0.5]),   #chg for 3-D, also Other option is 'random'
        #TBD: do we want to limit prey and pred speeds to account for terminal velcocities?
        # 'max_velocity': 10 / STEP_SIZE,
        'base_position':        np.array([0.0, 0.0]),
        'num_preds':            2,
        'num_dims':             2,
        'PRED_SIZE':            0, # 0.1 for shield, 1 for no shield
        'PREY_SIZE':            0.1, #<-- why not make this AGENT_SIZE as well?
        'KILL_RADIUS':          1.0,
        'STEP_SIZE':            1,
        'MAX_EPISODE_STEPS':    20,
        'max_acceleration':     10,                           
        'workspace_size':       10,
        'GEOFENCING' :          False,
        'DOING_OBSTACLES':      False, 
        'DOING_BOUNDED':        False,
        'STEPS_BOUND':          3,
        # --- 20% of worksapce size---
        'LObs' :                -4,
        'RObs' :                4,
        'CObs' :                -4,
        'FObs' :                4,
        'BObs' :                3,
        'TObs' :                8,

        'DOING_SEP':            True, # enable pred-pred separation shield (OKDist)
        'MAX_SEP':              -1,    # predators must stay within this distance of each other (L-inf); -1 to disable
        'MIN_SEP':              2,   # predators must stay at least this far apart (collision avoidance); -1 to disable
    },
}

# Import deferred to after DEFAULT_CONFIG is defined so that model_1pt.py
# can successfully import DEFAULT_CONFIG when DronesActionSelector loads it.
from src.DronesActionSelector import DronesActionSelector  # noqa: E402
DEFAULT_CONFIG['action_selector_class'] = DronesActionSelector
