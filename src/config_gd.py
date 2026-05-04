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
    #was previously using the classname reference in here directly but then that required this config to import the file with that actual class which caused circular imports elsewhere since that one needed the model file..
    'action_selector_class': 'src.DronesActionSelector:DronesActionSelector',

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
        'num_preds':            3,
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

        'DOING_SEP':            False, # enable pred-pred separation shield (OKDist)
        'MAX_SEP':              -1,    # predators must stay within this distance of each other (L-inf); -1 to disable
        'MIN_SEP':              2,     # predators must stay at least this far apart (collision avoidance); -1 to disable

        # --- smart prey / adversarial tracking shield (model_gd_smart_prey) ---
        'TRACKING_PREY':        True, # enable smart prey: adversarial prey acceleration + OKTrack shield
        'A_PREY_MAX':           7,     # max prey acceleration per axis (half of A_MAX)
        'MAX_TRACK_DIST':       10,    # predator must stay within this L-inf distance of prey; -1 to disable
        'MIN_TRACK_DIST':       -1,    # predator-prey min separation; -1 to disable.
                                        # NOTE: >0 not supported — post-clip repair analogous to DOING_SEP would be needed first
    },
}
