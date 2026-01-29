from config import DEFAULT_CONFIG
from copy import deepcopy
from src.PPOLearner import PPOLearner
import numpy as np
import os

'''
TODO:
- fix the use of DEFAULT_CONFIG in model_1pt to use the actual current config
- resolve confusion b/w agent_size and distance from base 
- BLshield needs redoig after that 
'''

#this is where tensorboard will look for displaying results
def mkResultsLoc(config):
    results_loc = '../experiments/results/' + str(config['experiment_id'])
    results_loc = os.path.abspath(results_loc)
    return results_loc

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
    config = deepcopy(DEFAULT_CONFIG) #why is this being copied?
    config['model_class'] = PPOLearner
    # Each experiment gets and id and a save location
    experiment_id = np.random.randint(9E9)
    config['experiment_id'] = experiment_id
    breakpoint()
    print('use_shield', config['env_config']['use_shield'])           
    results_loc = mkResultsLoc(config)
    saveConfig(config, results_loc)  # Now config has experiment_id
    learner = mkPPOLearner(config, results_loc)
    learn(config, learner)
