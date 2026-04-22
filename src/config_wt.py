from WaterTankEnv import WaterTankEnv
from stable_baselines3.ppo.policies import MultiInputPolicy
from src.WaterTankActionSelector import WaterTankActionSelector

DEFAULT_CONFIG = {
    'policy_class':             MultiInputPolicy,
    'learning_rate':            3e-4,
    'gamma':                    0.99,
    'max_learning_steps':       int(500E3),
    'buffer_size':              int(30E3),
    'num_shield_chances':       -1,
    'env_class':                WaterTankEnv,
    'action_selector_class':    WaterTankActionSelector,
    'env_config': {
        'use_shield':           False,  
        'INITIAL_TANK_LEVEL':   50.0,   # starting tank level
        'TANK_CAPACITY':        100.0,  # episode ends if exceeded
        'INN_MIN':              1.0,    # inflow lower bound  (from model envPred)
        'INN_MAX':              2.0,    # inflow upper bound
        'OUT_MIN':              0.0,    # outflow lower bound
        'OUT_MAX':              1.0,    # outflow upper bound
        'MAX_EPISODE_STEPS':    100,
        'TEMPORAL_VIOLATION_PENALTY':   1.0,  # reward penalty when t==2 or s==2 (shield-forced state)
    },
}
