from copy import deepcopy
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from .ActionSelector import ActionSelector
from .shielding.mr_models.model_gd import OK, solnExists, closestSafePoint
from .shielding.builder.utils import *


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
        self.proposed_action = None
        self.last_shield_failed = False
        self.env.no_safe_point_counter = 0
        self.shield_rate_decisions = 0
        self.shield_use_count = 0
        self.shield_correction_count = 0
        self.no_safe_point_count = 0
        if self.env.num_preds > 1:
            raise NotImplementedError('Multiple predators not implemented')


    '''returns triple: action_per_pred, value, log_prob_per_action, of which action_per_pred is a numpy array of vectors of length num_preds <-- OR is it an array of length num_preds*num_dims? b/c i think (eg [1.0,2.0,30, 0.1,0.2,0.3] for 2 preds in 3D)'''
    def getActionForEachAgent(self, single_obs):   #obs is a tensor 
        # single_obs = deepcopy(obs)
        # obss = self.replicateObsNumChancesTimes(single_obs)
        is_random_action = False
        (action_per_pred, value, log_prob_per_action) = \
            self.getPolicyAction(single_obs)
        is_random_action = self.last_shield_failed
        self.shield_rate_decisions += 1
        self.shield_use_count += int(any(
            getattr(pred, 'shield_was_used_in_step', False) for pred in self.env.predators))
        self.shield_correction_count += int(any(
            getattr(pred, 'sheild_corrected', False) for pred in self.env.predators))
        self.no_safe_point_count += int(self.last_shield_failed)
        # if failed_to_find_ok_action:
        #     (action_per_pred, value, log_prob_per_action) = \
        #         self.getRandomAction(single_obs) 
        #     is_random_action = True
        #     self.proposed_action = action_per_pred
            # print('had to pick rand action', action_per_pred)
                # self.getRandomAction(deepcopy(single_obs)) #TBD: remove this 2nd deep copy?

        assert len(action_per_pred) == self.env.num_preds * self.env.num_dims, 'action is wrong size!'
        return action_per_pred, value, log_prob_per_action, is_random_action


    '''this along with all the other "action" methods below return an action per pred'''
    def getPolicyAction(self, single_obs):
        #actions b/c its [num_reds, num_chances] array
        # actions, chosen_action_index = self.getOKPolicyAction(replicated_obs, single_obs)
        # actions, chosen_action_index, proposed_action, values, log_probs = self.getOKPolicyAction(replicated_obs, single_obs)
        # actions, action_per_pred, proposed_action, values, log_probs = self.getOKPolicyAction( single_obs)
        self.last_shield_failed = False
        action_per_pred, proposed_action, values, log_probs = self.getOKPolicyAction(single_obs)
        # failed_to_find_ok_action = False
        # Soemtimes the shield cant find an ok action due to FP errors: assert chosen_action_index >= 0, 'error: shield failed to find a safe action'
        # if chosen_action_index < 0:
        #     chosen_action_index = 0 # just use the original
        #     failed_to_find_ok_action = True
        
        #action_per_pred is 1-D array of length num_preds*num_dims
        # action_per_pred = actions[chosen_action_index] #was actions[chosen_action, :]
        # print('chosen_action_index', chosen_action_index, ' action_per_pred', action_per_pred)

        assert len(action_per_pred) == self.env.num_preds * self.env.num_dims, 'action is wrong size! ' + str(action_per_pred)

        #reshape b/c evaluate_actions expects a batched tensor. if there were multiple envs m it would be an m element array
        # action_per_pred_as_1_element_2D_array = th.tensor(action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)).to(self.policy.device)

        #If it were to take this action in this state, what is its probability and what is the state's value? NB: its called evaluate_actions plural b/c the action choices are typically batched up and called within train() in SB3. We need to call it here b/c we sampled multiple actions and then selected one, so need to get the value and log prob for that one action.
        # values and log_probs now come from the single policy() call, as log pi(proposal), so evaluate_actions is disabled
        # values, log_probs, _ = \
            # self.policy.evaluate_actions(single_obs, 
                                         # action_per_pred_as_1_element_2D_array
            # )
        self.proposed_action = proposed_action
        return action_per_pred, values, log_probs #, failed_to_find_ok_action

    '''this is not neccearily totally "OK". Due to roundoff errors, shield may still fail to find a safe action, in wnich case it uses actionss[0]'''
    def getOKPolicyAction(self,  single_obs):
        # if self.config['use_shield']:
        if self.config['env_config']['use_shield']:
            # return self.selectOKAction(actions, single_obs)
            return self.getShieldedAction(single_obs)
            # return actions, chosen_action_index
        else:
            return self.getUnshieldedAction(single_obs)

    def getUnshieldedAction(self, single_obs):
            #actions b/c its one action for each pred (even though its returns a single array eg [ 0.3, -0.7,  0.1,  0.9, -0.4,  0.2] for 3 preds. Actually it returns [[ 0.3, -0.7,  0.1,  0.9, -0.4,  0.2]] b/c its a vecrorized env hence the need for actions[0] below)
            actions, values, log_probs = self.policy(single_obs)    #sample the policy
            # breakpoint()
            #actions is pytorch tensor, convert to numpy array
            actions = actions.cpu().numpy()    
            #actions[0] b/c its a vectorized env see above
            return actions[0], actions[0].copy(), values, log_probs

    # was def getShieldedAction(self, actionss, single_obs):
    def getShieldedAction(self, single_obs):
        """ 
            - policy is repeatedly sampled as many times as the length of replicated_obs --> array num_chances long, but each array element is num_preds*num_dims array (eg [1.0,2.0,30, 0.1,0.2,0.3] for 2 preds in 3D) hence actions == action_per_pred x #preds repeated num_chances times!
            - obs_as_tensor takes the original dict of numpy arrays and replaces it with dict of tensors on the given device!
            - values and log_probs are discarded here b/c they are computed again in caller. 
        """
        replicated_obs = self.replicateObsNumChancesTimes(single_obs)
        #------- this is where it samples the policy -----------
        actionss, values, log_probs = self.policy(replicated_obs)
        actionss = actionss.cpu().numpy().astype(float)
        proposed_action = actionss[0].copy()
        # chosen_action_index = self.selectOKAction(actionss, single_obs)

        num_preds = self.env.num_preds

        pred_sts = []
        for pred_idx in range(num_preds):
            pred_sts.append(self.extractPosAndVelforPred(single_obs, pred_idx))
        prey_st = self.env.prey.position.tolist() + self.env.prey.velocity.tolist()

        """Look for an OK action that also doesnt need correcting"""        
        steps_remaining = self.env.STEPS_BOUND - self.env.n_steps_to_bound
        for sample_index, action in enumerate(actionss):
            acceleration = self.env.actionToAcceleration(action)
            ok_result = OK(self.config, acceleration, pred_sts[0], prey_st, steps_remaining)
            if isinstance(ok_result, list):
                forced_action = self.acclerationToAction(ok_result)
                self.env.predators[0].shield_was_used_in_step = True
                return (forced_action, action.copy(), values[sample_index:sample_index + 1],
                        log_probs[sample_index:sample_index + 1])
            if ok_result is not True:
                continue
            #calling cSP b/c an OK action right at the boudnary also needs correcting, otherwise single vs double float problem causes shield failure
            found, safe_acceleration = closestSafePoint(
                self.config, acceleration, pred_sts[0], prey_st, steps_remaining)
            if not found or not np.array_equal(acceleration, safe_acceleration):
                continue
            if sample_index > 0:
                self.env.predators[0].shield_was_used_in_step = True
            return (action.copy(), action.copy(), values[sample_index:sample_index + 1],
                    log_probs[sample_index:sample_index + 1])

        '''if it gets here then no truly safe action found'''
        # proposed_acceleration = actionss[0].astype(float) * self.env.max_acceleration
        # clipped first: what the env would execute unshielded, so only real shield corrections count
        proposed_acceleration = self.env.actionToAcceleration(actionss[0].astype(float))
        found, safe_acceleration = closestSafePoint(
            self.config, proposed_acceleration, pred_sts[0], prey_st, steps_remaining)
        if not found:
            # BEGIN TEMP DIAGNOSTIC: remove after investigating no-safe-projection warnings.
            viability = solnExists(self.config, pred_sts[0], prey_st, steps_remaining)
            if isinstance(viability, (list, tuple)):
                viable_within_remaining = any(
                    viability[:max(0, steps_remaining + 1)])
            else:
                viable_within_remaining = viability
            print(
                f'***WARNING: no safe projection; proposed_acc={proposed_acceleration}, '
                f'state={pred_sts[0]}, prey={prey_st}, steps_remaining={steps_remaining}, '
                f'solnExists={viability}, viable_within_remaining={viable_within_remaining}, '
                f'DOING_BOUNDED={self.config["env_config"]["DOING_BOUNDED"]}, '
                f'KILL_RADIUS={self.config["env_config"]["KILL_RADIUS"]}, '
                f'SHIELD_MARGIN={self.config["env_config"].get("SHIELD_MARGIN", 0.0)}')
            # END TEMP DIAGNOSTIC
            self.env.no_safe_point_counter += 1
            self.last_shield_failed = True
            
        #bounded last step needs special handling b/c pred very unlikely to find the exact vector by sampling alone
        if (found and not (self.config['env_config']['DOING_BOUNDED'] and steps_remaining == 1)
                and not np.array_equal(proposed_acceleration, safe_acceleration)):
            actionss[0] = self.acclerationToAction(safe_acceleration)
            self.env.predators[0].shield_was_used_in_step = True
            self.env.predators[0].sheild_corrected = True
        elif found and self.config['env_config']['DOING_BOUNDED'] and steps_remaining == 1:
            actionss[0] = self.acclerationToAction(safe_acceleration)
            self.env.predators[0].shield_was_used_in_step = True

        # old 100-sample selection loop, disabled
        '''
        # valid_actions = []
        chosen_action_index = -1
        prey_pos = self.env.prey.position
        prey_pos = prey_pos.tolist()
        prey_vel = self.env.prey.velocity.tolist()
        prey_st = prey_pos + prey_vel

        if self.env.start_of_episode:
          self.env.start_of_episode = False
          for pred_idx, current_state in enumerate(pred_sts):
            se = solnExists(current_state, prey_st, self.env.STEPS_BOUND)
            if se: print(f'solnExists pred {pred_idx}', current_state, prey_st, self.env.STEPS_BOUND)
            else: print(f'***WARNING: no solution from pred {pred_idx}', current_state, prey_st, self.env.STEPS_BOUND)

        steps_remaining = self.env.STEPS_BOUND - self.env.n_steps_to_bound

        for num, action in enumerate(actions): 
            agent_action = action.tolist()
            all_ok = True
            override_accels = {}  # pred_idx -> replacement acceleration
            # agent_acceleration = self.env.actionToAcceleration(agent_action)  # old: single pred
            joint_acceleration = self.env.actionToAcceleration(agent_action)  # clip against full joint action_space

            for pred_idx in range(num_preds):
                # pred_action = agent_action[pred_idx * num_dims:(pred_idx + 1) * num_dims]  # old: passed slice to actionToAcceleration, but that clips against full action_space shape so fails for num_preds>1
                # pred_accel = self.env.actionToAcceleration(pred_action)  # old
                pred_accel = joint_acceleration[pred_idx * num_dims:(pred_idx + 1) * num_dims]
                res = OK(pred_accel, pred_sts[pred_idx], prey_st, steps_remaining)
                if not res:
                    all_ok = False
                    break
                #this weird test for when OK returns the action the agent should pick
                if res != True and len(res) == num_dims:
                    override_accels[pred_idx] = res

            if all_ok:
                # if num>0 and res == True: print('accepted', agent_acceleration, current_state)
                shield_needed = num > 0 or bool(override_accels)
                if shield_needed:
                    for pred in self.env.predators:
                        pred.shield_was_used_in_step = True
                if override_accels:
                    # print('res acc', res)
                    mixed_action = list(agent_action)
                    for pred_idx, accel in override_accels.items():
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
                # print('agent_acceleration', agent_acceleration, "failed in", current_state)
        return chosen_action_index
        '''    
        return actionss[0], proposed_action, values[:1], log_probs[:1]


    def extractPosAndVelforPred(self, single_obs, pred_idx):
        num_dims = self.env.num_dims
        current_position = single_obs['pred_posz'][:, pred_idx * num_dims:(pred_idx + 1) * num_dims].cpu().numpy()
        current_velocity = single_obs['agent_velocities'][:, pred_idx * num_dims:(pred_idx + 1) * num_dims].cpu().numpy()
        # print('current_position', current_position,'current_velocity', current_velocity)
        return np.concatenate((current_position, current_velocity), axis=1).squeeze().tolist()

    #actions are normalized values
    def acclerationToAction(self, acc):
        action = [i / self.env.max_acceleration for i in acc]
        # maybe assert (self.action_space.low <= elt for elt in action).all() and .. 
        # SN: tried assert self.action_space.log <= action, etc but didnt like that either
        # assert all(self.action_space.low <= elt and elt <= self.action_space.high for elt in action), "provided acceleration value outside action space limits"
        return action


    '''Draw <num_chances> random actions 1 per pred from uniform distribution'''
    def sampleFromUniformDistrib(self, single_obs):
        low = self.env.action_space.low
        high = self.env.action_space.high
        actions = np.random.uniform(
            low, high, size=(self.num_chances, self.env.num_preds * self.env.num_dims))
        return actions


    '''
                GRAVEYARD
    def getRandomAction(self, single_obs):
        # Get the actions
        actions = self.sampleFromUniformDistrib(single_obs)
        chosen_action_index = self.selectOKAction(actions, single_obs)  

        if chosen_action_index < 0: #was unable to find an ok action by random selection, so just pick the first action - kind of silly, need to fix
            chosen_action_index = 0
        # if (num == self.num_chances - 1):
            self.n_agent_fails += 1
            if self.n_agent_fails % 100 == 0:
                print('n_agent_fails', self.n_agent_fails)

        action_per_pred = actions[chosen_action_index]

        values, log_probs, _ = \
            self.policy.evaluate_actions(
                single_obs,
                # 1st dim is just one b/c have only one env
                th.tensor(action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)).to(self.policy.device))
        return action_per_pred, values, log_probs
    '''

