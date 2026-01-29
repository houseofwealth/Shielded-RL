from env import PredPreyEnv
#below is what learning alg to use and which kind of policy it uses
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy #MIP is alias for MultiInputActorCriticPolicy
# from ray import tune
import numpy as np

# USE_PARALLEL = False
AGENT_SIZE = 0.1
print('AGENT_SIZE', AGENT_SIZE)
STEP_SIZE = 1                                     
""" 20% of worksapce size"""
LObs = -4
RObs = 4
CObs = -4
FObs = 4
BObs = 3
TObs = 8

DEFAULT_CONFIG = {
    'policy_class': MultiInputPolicy,
    'learning_rate': 5e-5, #1e-4, #was 3e-4,     #the higher, the faster the learning but can be unstable or overshoot the optimal soln

    # 'batch_size': was this higher? But im sticking with PPO default of 64
    'gamma': 0.99,                        #discount factor for future rewards

    # 'ent_coef' currently defaul of 0, its entropy term in loss fun, the higher this is, The action distribution becomes flatter (more spread out ie less deterministic), meaning the agent is more likely to explore a wider range of actions.

    #Controls the maximum divergence (KL divergence) between the old and new policies during training. It prevents the policy from changing too drastically in one update, ensuring stable learning
    #'target_kl': 0.05,                   

    'max_episode_length': 100,
    'max_learning_steps': int(1000E3),     #total for trining run
    'buffer_size': int(30E3),             #rollout buffer size
    # moved -> env_config b/c env now needs it 'use_shield': False,
    'num_shield_chances': 100,
    'env_class': PredPreyEnv,
    'env_config': {        
        'use_shield': False,
        'initial_prey_pos' : np.array([0, 10.0]),         #chg for 3-D, also Other option is 'random'
        'initial_prey_velocity': np.array([0.0, -0.5]),   #chg for 3-D, also Other option is 'random'
        #TBD: do we want to limit prey and pred speeds to account for terminal velcocities?
        # 'max_velocity': 10 / STEP_SIZE,
        'base_position':        np.array([0.0, 0.0]),
        'num_preds':            1,
        'num_dims':             2,
        'AT_TARGET_RADIUS':     0, # 0.1 for shield, 1 for no shield
        'MAX_EPISODE_STEPS':    20,
        'MAX_ACCEPTABLE_RADIUS':AGENT_SIZE,
        'STEP_SIZE':            STEP_SIZE,
        'max_acceleration':     10,                           
        'workspace_size':       10,
        'GEOFENCING' :          True,
        'DOING_OBSTACLES':      False, 
        'DOING_BOUNDED':        False, 
        'STEPS_BOUND':          3,
        'LObs' :                LObs,
        'RObs' :                RObs,
        'CObs' :                CObs,
        'FObs' :                FObs,
        'BObs' :                BObs,
        'TObs' :                TObs
    },
}
