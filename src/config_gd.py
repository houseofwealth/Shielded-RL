from GameOfDronesEnv import GameOfDronesEnv
#below is what learning alg to use and which kind of policy it uses
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy #MIP is alias for MultiInputActorCriticPolicy
# from ray import tune
import numpy as np

# USE_PARALLEL = False
AGENT_SIZE = 0
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
    'learning_rate': 3e-4,  #feb experiments with 5e-5,     #the higher, the faster the learning but can be unstable or overshoot the optimal soln

    # 'batch_size': was this higher? But im sticking with PPO default of 64
    'gamma': 0.99,                        #discount factor for future rewards

    'ent_coef': 0,   #Feb experiments used 0.05, #was 0 then 0.1 (worsened) It prevents the policy from changing too drastically in one update, ensuring stable learning
    #'target_kl': 0.05,                   

    # 'max_episode_length': 100,
    'max_learning_steps': int(500E3), #int(1000E3),     #total for trining run
    'buffer_size': int(30E3),             #rollout buffer size
    # moved -> env_config b/c env now needs it 'use_shield': False,
    'num_shield_chances': 100,
    'env_class': GameOfDronesEnv,
    'action_selector_class': None,  # set below after DEFAULT_CONFIG to avoid circular import
    'env_config': {        
        'use_shield': True,
        'initial_prey_pos' : np.array([0, 10.0]),         #chg for 3-D, also Other option is 'random'
        'initial_prey_velocity': np.array([0.0, -0.5]),   #chg for 3-D, also Other option is 'random'
        #TBD: do we want to limit prey and pred speeds to account for terminal velcocities?
        # 'max_velocity': 10 / STEP_SIZE,
        'base_position':        np.array([0.0, 0.0]),
        'num_preds':            1,
        'num_dims':             2,
        'PRED_SIZE':            AGENT_SIZE, # how close preds get to each other before colliding, 0.1 for shield, 1 for no shield
        'PREY_SIZE':            0.1,  #how close it needs to get to be at BASE <-- why not make this AGENT_SIZE as well?
        'KILL_RADIUS':          1.0,  #how close pred has to get to count as KILL
        'STEP_SIZE':            STEP_SIZE,
        'MAX_EPISODE_STEPS':    20,
        'max_acceleration':     10,                           
        'workspace_size':       10,
        'GEOFENCING' :          False,
        'DOING_OBSTACLES':      False, 
        'DOING_BOUNDED':        True,
        'STEPS_BOUND':          3,
        'LObs' :                LObs,
        'RObs' :                RObs,
        'CObs' :                CObs,
        'FObs' :                FObs,
        'BObs' :                BObs,
        'TObs' :                TObs,
        'SHIELD_MARGIN':        1e-5,   #when correction happens placing the corected point right on the polytope boundary can lead to violations b/c of roundoff errors/ This margin causes the safe point to be a bit inside the polytope
        'SHIELD_PENALTY':       0.01,    # reward for a step where the shield intervened, replacing the task reward; placeholder, to be tuned

        # 'CHECKING_COLLISIONS':  True, # whether shield should check for collisions (not whether collisions should be fatal)

        # --- Multi-agent (MA) settings below are commented out: not used by the single-agent code ---
        # --- min max separation between preds ---
        # 'DOING_SEP':            False, # enable pred-pred separation 
        # 'MIN_SEP':              1, #0.1,     # predators must stay at least this far apart; -1 to disable. Set small non zero value for collision avoidance
        # 'MAX_SEP':              15, #-1,    # predators must stay within this distance of each other (L-inf); -1 to disable

        # --- smart prey / adversarial tracking shield (model_gd_smart_prey) ---
        # 'TRACKING_PREY':        True, # enable smart prey: adversarial prey acceleration + OKTrack shield
        # 'A_PREY_MAX':           5,     # max prey acceleration per axis (7 is max for \E \A)
        # 'MAX_TRACK_DIST':       10,    # predator must stay within this L-inf distance of prey; -1 to disable
        # 'MIN_TRACK_DIST':       -1,    # predator-prey min separation; -1 to disable.
                                        # NOTE: >0 not supported — post-clip repair analogous to DOING_SEP would be needed first
    },
}

# Import deferred to after DEFAULT_CONFIG is defined so that model_1pt.py
# can successfully import DEFAULT_CONFIG when DronesActionSelector loads it.
from src.DronesActionSelector import DronesActionSelector  # noqa: E402
DEFAULT_CONFIG['action_selector_class'] = DronesActionSelector
