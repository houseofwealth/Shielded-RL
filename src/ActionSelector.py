from copy import deepcopy
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
# from ipdb import set_trace
from .shields.mr_models.model_1pt import *
from .shields.builder.utils import *

class ActionSelector:
    '''
    The CustomPPO alg (the "agent") calls on this to find an action for the RL that passes the shield (not ideal design, but easiest to just replace calls to policy() that the base class PPO does with calls to the ActionSelector).
    '''

    def __init__(self, config, policy, env):
        # No! instantiates new env: self.env = config['env'](config['env_config'])
        self.env = env.envs[0]  #NOTE: assumes vec env with only 1 env inside
        #not correct: it gives the class not the object: config['policy']
        self.policy = policy 
        self.num_chances = config['num_shield_chances']
        # if self.env.num_preds > 1:
        #     raise NotImplementedError('Multiple predators not implemented')
        self.config = config
        self.n_agent_fails = 0  #tracks # times agent failed to find ok action

    
    '''returns triple: action_per_pred, value, log_prob_per_action, of which action_per_pred is a numpy array of vectors of length num_preds <-- OR is it an array of length num_preds*num_dims? b/c i think (eg [1.0,2.0,30, 0.1,0.2,0.3] for 2 preds in 3D)'''
    def getActionForEachPred(self, single_obs):   #obs is a tensor 
        # single_obs = deepcopy(obs)
        obss = self.replicateObsNumChancesTimes(single_obs)
        (action_per_pred, value, log_prob_per_action, failed_to_find_ok_action) = \
            self.getPolicyAction(obss, single_obs)
        if failed_to_find_ok_action:
            (action_per_pred, value, log_prob_per_action) = \
                self.getRandomAction(single_obs) 
                # self.getRandomAction(deepcopy(single_obs)) #TBD: remove this 2nd deep copy?

        assert len(action_per_pred) == self.env.num_dims, 'action is wrong size!'
        return action_per_pred, value, log_prob_per_action
    

    '''this along with all the other "action" methods below return an action per pred'''
    def getPolicyAction(self, replicated_obs, single_obs):
        #actionss b/c its [num_reds, num_chances] array
        actionss, chosen_action_index = self.getOKPolicyAction(replicated_obs,single_obs)
        failed_to_find_ok_action = False
        if chosen_action_index < 0:
            chosen_action_index = 0
            failed_to_find_ok_action = True
        
        #action_per_pred is 1-D array of length num_preds*num_dims
        action_per_pred = actionss[chosen_action_index] #was actions[chosen_action, :]
        # print('chosen_action_index', chosen_action_index, ' action_per_pred', action_per_pred)

        assert len(action_per_pred) == self.env.num_preds * self.env.num_dims, 'action is wrong size! ' + str(action_per_pred)

        #reshape b/c evaluat_actoins expects a batched tensor. if there were multiple envs m it would be an m element array
        action_per_pred_as_1_element_2D_array = th.tensor(action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)).to(self.policy.device)

        #If it were to take this action in this state, what is its probability and what is the state's value? NB: its called evaluate_actions plural b/c the action choices are tpyically batched up and called within train() in SB3. We need to call it here b/c we sampled multiple actions and then selected one, so need to get the value and log prob for that one action.
        values, log_probs, _ = \
            self.policy.evaluate_actions(single_obs, 
                                         action_per_pred_as_1_element_2D_array
            )
        return action_per_pred, values, log_probs, failed_to_find_ok_action
    

    def getOKPolicyAction(self, replicated_obs, single_obs):
        # if self.config['use_shield']:
        if self.config['env_config']['use_shield']:
            """ 
                - policy is repeatedly sampled as many times as the length of replicated_obs --> array num_chances long, but each array element is num_preds*num_dims array (eg [1.0,2.0,30, 0.1,0.2,0.3] for 2 preds in 3D) b/c each pred has its own policy (decentralized MARL). hence actionss!
                - obs_as_tensor take the original dict of numpy arrays and replaces it with dict of tensors on the given device!
                - values and log_probs are discarded here b/c they are computed again in caller. 
            """
            obs_tensor = obs_as_tensor(replicated_obs, self.policy.device)
            #------- this is where it samples the policy -----------
            #Normally, the values and log_probs returned here are directly used, however as we get a bunch of actions and then select one from that bunch, it is necessary to call evaluate_actions() in getPolicyAction above to get the actual values and log probs for the chosen action
            actionss, values, log_probs = self.policy(obs_tensor)
            #-------------------------------------------------------
            actionss = actionss.cpu().numpy()
            chosen_action_index = self.selectOKAction(actionss, single_obs)
            return actionss, chosen_action_index
        else:
            #even though its passed a dict with values being single array, policy still returns a 2-D array, [[-1.2, 3.4]]
            actionss, values, log_probs = self.policy(single_obs)
            # breakpoint()
            actionss = actionss.cpu().numpy()    
            #actionss is pytorch tensor, convert to numpy array
            return actionss, 0 

    def selectOKAction(self, actions, single_obs):
        #actionss is really actionss b/c each elt is an action per pred, but code below is only writtenfor single pred. *TBDL fix for multiple preds*
        assert(self.env.num_preds == 1), 'Multiple predators not implemented'

        # Get the positions that would result from the actions
        num_dims = self.env.num_dims
        current_position = single_obs['pred_posz'].cpu().numpy()
        current_velocity = single_obs['agent_velocities'][:, :num_dims].cpu().numpy()
        # print('current_position', current_position,'current_velocity', current_velocity)
        current_state = np.concatenate((current_position, current_velocity), axis=1).squeeze().tolist()
        current_state_z3 = list(map(toZ3Type, current_state))

        # valid_actions = []
        chosen_action_index = -1
        prey_pos = self.env.prey.position
        prey_pos = prey_pos.tolist()
        prey_vel = self.env.prey.velocity.tolist()
        prey_st = prey_pos + prey_vel

        if self.env.start_of_episode:
          self.env.start_of_episode = False
          se = solnExists(current_state, prey_st, self.env.STEPS_BOUND)
          if se: print('solnExists in', current_state, prey_st, self.env.STEPS_BOUND)
          else: print('***WARNING: no solution from', current_state, prey_st, self.env.STEPS_BOUND)

        for num, action in enumerate(actions): 
            agent_action = action.tolist()
            agent_acceleration = self.env.actionToAcceleration(agent_action)
            
            res = OK(
                agent_acceleration,
                current_state,
                prey_st,
                self.env.STEPS_BOUND - self.env.n_steps_to_bound,
                )
            if res:
                # if num>0 and res == True: print('accepted', agent_acceleration, current_state)
                if num > 0:     #if num=0 then shield wasn't needed
                    self.env.predators[0].shield_was_used_in_step = True
                #this weird test for when OK returns the action action the agent should pick
                if res != True and len(res)==self.env.num_dims: 
                    self.env.predators[0].shield_was_used_in_step = True
                    # print('res acc', res)
                    res = self.acclerationToAction(res) #np.array(res)).tolist()
                    actions[0] = res
                    #**TBD what about multiple preds?
                    chosen_action_index = 0   
                else:
                    # print('action', num, 'is ok')
                    chosen_action_index = num
                break
            # else: 
                # print('agent_acceleration', agent_acceleration, "failed in", current_state)
        return chosen_action_index    
    

    #actions are normalized values
    def acclerationToAction(self, acc):
        action = [i / self.env.max_acceleration for i in acc]
        # maybe assert (self.action_space.low <= elt for elt in action).all() and .. 
        # SN: tried assert self.action_space.log <= action, etc but didnt like that either
        # assert all(self.action_space.low <= elt and elt <= self.action_space.high for elt in action), "provided acceleration value outside action space limits"
        return action


    def getRandomAction(self, single_obs):
        # Get the actions
        actionss = self.sampleFromUniformDistrib(single_obs)
        chosen_action_index = self.selectOKAction(actionss, single_obs)  

        if chosen_action_index < 0: #ue wasnt able to find an ok action by random selection, so just pick the first action - kind of silly need to fix
            chosen_action_index = 0
        # if (num == self.num_chances - 1):
            self.n_agent_fails += 1
            if self.n_agent_fails%100 == 0:
                print('n_agent_fails', self.n_agent_fails)

        action_per_pred = actionss[chosen_action_index]

        values, log_probs, _ = \
            self.policy.evaluate_actions(
                single_obs,
                # 1st dim is just one b.c have only one env
                th.tensor(action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)).to(self.policy.device))
        return action_per_pred, values, log_probs
    

    '''Draw <num_chances> random actions 1 per pred from uniform distribution'''
    def sampleFromUniformDistrib(self, single_obs):
        low = self.env.action_space.low
        high = self.env.action_space.high
        actionss = np.random.uniform(
            low, high, size=(self.num_chances, self.env.num_preds * self.env.num_dims))
        return actionss
    

    '''obs is a dict of key,tensor pairs, the keys are 'agent_velocities' 'acheived_goal' etc. return a dict in which the tensor for each key is repeated num_chances times along dim 0, so the value has shape [num_chances,tensor]'''
    def replicateObsNumChancesTimes(self, obs):
        if not isinstance(obs, dict):
            raise NotImplementedError('Can only handle obs that are dicts!')
        # Create a new dictionary with repeated tensors
        return {key: value.repeat(self.num_chances, 1) for key, value in obs.items()}
