from copy import deepcopy
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from .ActionSelector import ActionSelector
from .shields.mr_models.model_gd import OK, solnExists
from .shields.mr_models.model_gd_dist import OKDist
from .shields.builder.utils import *


class DronesActionSelector(ActionSelector):
    '''
    Game-of-Drones specific action selector.
    The CustomPPO alg (the "agent") calls on this to find an action for the RL
    that passes the shield (not ideal design, but easiest to just replace calls
    to policy() that the base class PPO does with calls to the ActionSelector).
    TODO: refactor into a cleaner GoD subclass; shield-check logic is currently
    entangled with policy sampling.
    '''

    def __init__(self, config, policy, env):
        super().__init__(config, policy, env)
        # if self.env.num_preds > 1:
        #     raise NotImplementedError('Multiple predators not implemented')


    '''returns triple: action_per_pred, value, log_prob_per_action, of which action_per_pred is a numpy array of vectors of length num_preds <-- OR is it an array of length num_preds*num_dims? b/c i think (eg [1.0,2.0,3.0, 0.1,0.2,0.3] for 2 preds in 3D)'''
    def getActionForEachAgent(self, single_obs):   #obs is a tensor 
        # single_obs = deepcopy(obs)
        obss = self.replicateObsNumChancesTimes(single_obs)
        is_random_action = False
        (action_per_pred, value, log_prob_per_action, failed_to_find_ok_action) = \
            self.getPolicyAction(obss, single_obs)
        if failed_to_find_ok_action:
            (action_per_pred, value, log_prob_per_action) = \
                self.getRandomAction(single_obs) 
            is_random_action = True
            print('had to pick rand action', action_per_pred)
                # self.getRandomAction(deepcopy(single_obs)) #TBD: remove this 2nd deep copy?

        assert len(action_per_pred) == self.env.num_preds * self.env.num_dims, 'action is wrong size!'
        return action_per_pred, value, log_prob_per_action, is_random_action


    '''this along with all the other "action" methods below return an action per pred'''
    def getPolicyAction(self, replicated_obs, single_obs):
        #actionss b/c its [num_reds, num_chances] array
        actionss, chosen_action_index = self.getOKPolicyAction(replicated_obs, single_obs)
        failed_to_find_ok_action = False
        if chosen_action_index < 0:
            chosen_action_index = 0
            failed_to_find_ok_action = True
        
        #action_per_pred is 1-D array of length num_preds*num_dims
        action_per_pred = actionss[chosen_action_index] #was actions[chosen_action, :]
        # print('chosen_action_index', chosen_action_index, ' action_per_pred', action_per_pred)

        assert len(action_per_pred) == self.env.num_preds * self.env.num_dims, 'action is wrong size! ' + str(action_per_pred)

        #reshape b/c evaluate_actions expects a batched tensor. if there were multiple envs m it would be an m element array
        action_per_pred_as_1_element_2D_array = th.tensor(action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)).to(self.policy.device)

        #If it were to take this action in this state, what is its probability and what is the state's value? NB: its called evaluate_actions plural b/c the action choices are typically batched up and called within train() in SB3. We need to call it here b/c we sampled multiple actions and then selected one, so need to get the value and log prob for that one action.
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
                - obs_as_tensor takes the original dict of numpy arrays and replaces it with dict of tensors on the given device!
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
            #even though its passed a dict with values being single array, policy still returns a 2-D array, eg 3 preds
            # [ [ 0.3, -0.7,  0.1,  0.9, -0.4,  0.2],   chance 0: joint action for all 3 preds
            #   [-0.1,  0.5,  0.8, -0.3,  0.6, -0.9]...] #chance 2
            actionss, values, log_probs = self.policy(single_obs)
            # breakpoint()
            actionss = actionss.cpu().numpy()    
            #actionss is pytorch tensor, convert to numpy array
            return actionss, 0 

    def selectOKAction(self, actions, single_obs):
        #actionss is really actionss b/c each elt is an action per pred, but code below is only written for single pred. *TBD: fix for multiple preds*
        num_dims = self.env.num_dims
        num_preds = self.env.num_preds

        # Get the positions that would result from the actions
        pred_states = []
        for pred_idx in range(num_preds):
            current_position = single_obs['pred_posz'][:, pred_idx * num_dims:(pred_idx + 1) * num_dims].cpu().numpy()
            current_velocity = single_obs['agent_velocities'][:, pred_idx * num_dims:(pred_idx + 1) * num_dims].cpu().numpy()
            # print('current_position', current_position,'current_velocity', current_velocity)
            current_state = np.concatenate((current_position, current_velocity), axis=1).squeeze().tolist()
            pred_states.append(current_state)

        # valid_actions = []
        chosen_action_index = -1
        prey_pos = self.env.prey.position
        prey_pos = prey_pos.tolist()
        prey_vel = self.env.prey.velocity.tolist()
        prey_st = prey_pos + prey_vel

        if self.env.start_of_episode:
          self.env.start_of_episode = False
          for pred_idx, current_state in enumerate(pred_states):
            se = solnExists(current_state, prey_st, self.env.STEPS_BOUND)
            if se: print(f'solnExists pred {pred_idx}', current_state, prey_st, self.env.STEPS_BOUND)
            else: print(f'***WARNING: no solution from pred {pred_idx}', current_state, prey_st, self.env.STEPS_BOUND)

        steps_remaining = self.env.STEPS_BOUND - self.env.n_steps_to_bound

        for num, action in enumerate(actions): 
            agent_action = action.tolist()
            all_ok = True
            replaced_accs = {}  # pred_idx -> replacement acceleration
            # agent_acceleration = self.env.actionToAcceleration(agent_action)  # old: single pred
            joint_acc = self.env.actionToAcceleration(agent_action)  # clip against full joint action_space

            pred_accs = []
            for pred_idx in range(num_preds):
                # pred_action = agent_action[pred_idx * num_dims:(pred_idx + 1) * num_dims]  # old: passed slice to actionToAcceleration, but that clips against full action_space shape so fails for num_preds>1
                # pred_acc = self.env.actionToAcceleration(pred_action)  # old
                pred_acc = joint_acc[pred_idx * num_dims:(pred_idx + 1) * num_dims]
                pred_accs.append(pred_acc)
                res = OK(pred_acc, pred_states[pred_idx], prey_st, steps_remaining)
                if not res:
                    all_ok = False
                    break
                #this weird test for when OK returns the action the agent should pick
                if res != True and len(res) == num_dims:
                    replaced_accs[pred_idx] = res

            # Pairwise separation check: only pred i=0 is held responsible for each pair (i,j)
            if all_ok and self.env.DOING_SEP and num_preds > 1:
                for i in range(num_preds):
                    for j in range(i + 1, num_preds):
                        accel_i = replaced_accs.get(i, pred_accs[i])
                        accel_j = replaced_accs.get(j, pred_accs[j])
                        if not OKDist(accel_i, pred_states[i], accel_j, pred_states[j]):
                            all_ok = False
                            break
                    if not all_ok:
                        break

            if all_ok:
                # if num>0 and res == True: print('accepted', agent_acceleration, current_state)
                shield_needed = num > 0 or replaced_accs
                if shield_needed:
                    for pred in self.env.predators:
                        pred.shield_was_used_in_step = True
                if replaced_accs:
                    # print('res acc', res)
                    mixed_action = list(agent_action)
                    for pred_idx, accel in replaced_accs.items():
                        override_action = self.acclerationToAction(accel)
                        mixed_action[pred_idx * num_dims:(pred_idx + 1) * num_dims] = override_action
                    actions[0] = mixed_action
                    #**TBD what about multiple preds?
                    chosen_action_index = 0   
                else:
                    # print('action', num, 'is ok')
                    chosen_action_index = num
                break
            # else: 
                # print('agent_acc', agent_acc, "failed in", current_state)
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

        if chosen_action_index < 0: #was unable to find an ok action by random selection, so just pick the first action - kind of silly, need to fix
            chosen_action_index = 0
        # if (num == self.num_chances - 1):
            self.n_agent_fails += 1
            if self.n_agent_fails % 100 == 0:
                print('n_agent_fails', self.n_agent_fails)

        action_per_pred = actionss[chosen_action_index]

        values, log_probs, _ = \
            self.policy.evaluate_actions(
                single_obs,
                # 1st dim is just one b/c have only one env
                th.tensor(action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)).to(self.policy.device))
        return action_per_pred, values, log_probs


    '''Draw <num_chances> random actions 1 per pred from uniform distribution'''
    def sampleFromUniformDistrib(self, single_obs):
        low = self.env.action_space.low
        high = self.env.action_space.high
        actionss = np.random.uniform(
            low, high, size=(self.num_chances, self.env.num_preds * self.env.num_dims))
        return actionss
