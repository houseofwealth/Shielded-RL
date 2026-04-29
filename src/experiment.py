from config_gd import DEFAULT_CONFIG  # Game of Drones
# from config_wt import DEFAULT_CONFIG   # Water Tank
from copy import deepcopy
from src.PPOLearner import PPOLearner
import numpy as np
import os
import argparse

'''
TODO:
- fix the use of DEFAULT_CONFIG in model_1pt to use the actual current config
- resolve confusion b/w agent_size and distance from base 
- BLshield needs redoig after that 
'''

#this is where tensorboard will look for displaying results
def mkResultsLoc(config):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../prototype/src
    results_loc = os.path.join(script_dir, '../experiments/results', str(config['experiment_id']))
    return os.path.abspath(results_loc)

def saveConfig(config, results_loc=None):
    if results_loc is None:
        results_loc = mkResultsLoc(config)
    fname = results_loc + '/config.txt'
    os.makedirs(os.path.dirname(fname), exist_ok=True)  # Ensure the directory exists
    print('Saving config to: ' + fname)
    with open(fname, 'w') as f:
        print(config, file=f)

''' Create a learner. SB3 calls this the PPO "model" or algorithm but technically the model is the A-C networks '''
def mkPPOLearner(config, results_loc):
    # do we need to adjust lr schedule at all? See SB3 notes
    learner = config['model_class'](
        #maybe a bunch of hyperparams need setting but am using defaults for most of them
        config,
        tensorboard_log=results_loc,
        # target_kl=0.05              
        target_kl= config.get('target_kl', 0.05)  # Default to 0.05 if not in config        
    )
    return learner

def loadLearner(config, experiment_id):
    fname = '../experiments/models/' + str(experiment_id)
    print('Loading learner from: ' + os.path.abspath(fname))
    env = config['env_class'](config)
    # PPO.load() is used instead of PPOLearner.load() because PPOLearner.__init__ expects
    # 'config' not 'policy', so SB3's load() classmethod would fail on PPOLearner directly.
    # We then patch the class back to PPOLearner and re-attach PPOLearner-specific state.
    from stable_baselines3 import PPO
    learner = PPO.load(fname, env=env)
    learner.__class__ = PPOLearner
    learner.use_shield = config['env_config']['use_shield']
    action_selector_class = config['action_selector_class']
    learner.action_selector = action_selector_class(config, learner.policy, learner.env)
    return learner

def learn(config, learner):
    experiment_id = config['experiment_id']  # Get from config after mkResultsLoc sets it
    learner.learn(config )

    fname = '../experiments/models/' + str(experiment_id)
    print('Saving learner to: ' + os.path.abspath(fname))
    learner.save(fname)
    print('Learning complete')


'''
create the learner with or without shielding and call learn
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'run'], default='train')
    parser.add_argument('--id', type=int, default=None, help='experiment_id of saved model (required for --mode run)')
    args = parser.parse_args()

    config = deepcopy(DEFAULT_CONFIG) #why is this being copied?
    config['model_class'] = PPOLearner

    if args.mode == 'run':
        assert args.id is not None, 'Must provide --id <experiment_id> for run mode'
        learner = loadLearner(config, args.id)
        learner.run()
    else:
        # Each experiment gets and id and a save location
        experiment_id = np.random.randint(9E9)
        config['experiment_id'] = experiment_id
        print('use_shield', config['env_config']['use_shield'])           
        breakpoint()
        results_loc = mkResultsLoc(config)
        saveConfig(config, results_loc)  # Now config has experiment_id
        learner = mkPPOLearner(config, results_loc)
        learn(config, learner)
        # learner.run()
