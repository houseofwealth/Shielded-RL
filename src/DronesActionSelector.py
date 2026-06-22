from copy import deepcopy
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from .ActionSelector import ActionSelector
from .shields.mr_models.model_gd import OK, solnExists
from .shields.mr_models.model_gd_dist import OKDist, solnExistsDist
from .shields.mr_models.model_gd_smart_prey import OKTrack, solnExistsTrack
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
        # Default-on here to avoid touching config files unless you want to override.
        self.allow_clip_bypass_in_selector = self.config['env_config'].get('allow_clip_bypass_in_selector', True)
        self.clip_bypass_used = 0
        # if self.env.num_preds > 1:
        #     raise NotImplementedError('Multiple predators not implemented')


    '''returns triple: action_per_pred, value, log_prob_per_action, of which action_per_pred is a numpy array of vectors of length num_preds <-- OR is it an array of length num_preds*num_dims? b/c i think (eg [1.0,2.0,3.0, 0.1,0.2,0.3] for 2 preds in 3D)'''
    def getActionForEachAgent(self, single_obs):   #obs is a tensor 
        if self.config['env_config']['use_shield'] and self.env.start_of_episode:
            single_obs = self.ensureValidInitState(single_obs)

        self.effective_obs = self.tensorObsToNumpyObs(single_obs)
        obss = self.replicateObsNumChancesTimes(single_obs)
        is_random_action = False
        (action_per_pred, value, log_prob_per_action, failed_to_find_ok_action, policy_batch) = \
            self.getPolicyAction(obss, single_obs)
        if failed_to_find_ok_action:
            # All candidates in the policy batch failed the shield. Rather than doing a fresh
            # uniform resample (getRandomAction), pick a random candidate from the already-sampled
            # policy batch so the executed action stays closer to on-policy.
            action_per_pred = policy_batch[np.random.randint(len(policy_batch))]
            action_per_pred_tensor = th.tensor(
                action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)
            ).to(self.policy.device)
            value, log_prob_per_action, _ = self.policy.evaluate_actions(single_obs, action_per_pred_tensor)
            is_random_action = True
            self.n_agent_fails += 1
            if self.n_agent_fails % 1 == 0:
                print('n_agent_fails', self.n_agent_fails)

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
        return action_per_pred, values, log_probs, failed_to_find_ok_action, actionss


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

        pred_states = self.getCurrentPredStates(single_obs, num_preds, num_dims)

        # valid_actions = []
        chosen_action_index = -1
        prey_pos = self.env.prey.position
        prey_pos = prey_pos.tolist()
        prey_vel = self.env.prey.velocity.tolist()
        prey_st = prey_pos + prey_vel

        steps_remaining = self.env.STEPS_BOUND - self.env.n_steps_to_bound

        # First prefer a sampled joint action that already passes all enabled shields.
        for num, action in enumerate(actions):
            joint_acc = self.env.actionToAcceleration(action.tolist())
            unary_guards_passed, guarded_accels, any_accel_overridden = self.checkUnaryGuards(
                joint_acc, pred_states, prey_st, steps_remaining, num_preds, num_dims)

            if not unary_guards_passed:
                continue    #onto next action choice, this one failed

            if not self.checkBinaryGuards(guarded_accels, pred_states, num_preds):
                continue    #onto next action choice, this one failed 

            if num > 0 or any_accel_overridden:
                for pred in self.env.predators:
                    pred.shield_was_used_in_step = True

            if not any_accel_overridden:
                return num

            actions[0] = self.composeNewJointAction(action, guarded_accels, num_preds, num_dims)
            return 0

        # Fallback: if it got to here, scan all samples for each predator separately and compose a joint action.
        # chosen_per_pred[pred_idx] = (acc_to_use, is_replaced, sample_index)
        # This reduces effective joint rejection rate from p^num_preds to p.
        # its a kind of diagonalization
        print('*Failed to find an acceptable action tuple - trying each pred separately')
        chosen_per_pred = self.findActionForEachPredSeparately(
            actions, pred_states, prey_st, steps_remaining, num_preds, num_dims)
        if chosen_per_pred is None:
            return -1

        # Did the shield have to do anything?
        sample_indices  = [chosen_per_pred[i][2] for i in range(num_preds)]
        any_replaced    = any(chosen_per_pred[i][1] for i in range(num_preds))
        all_same_sample = len(set(sample_indices)) == 1

        shield_needed = any_replaced or not all_same_sample or sample_indices[0] > 0
        if shield_needed:
            for pred in self.env.predators:
                pred.shield_was_used_in_step = True

        if not any_replaced and all_same_sample:
            # All predators passed from the same joint sample with no overrides — use it directly
            chosen_action_index = sample_indices[0]
        else:
            # Compose a joint action from the per-predator winners
            guarded_accels = [chosen_per_pred[pred_idx][0] for pred_idx in range(num_preds)]
            actions[0] = self.composeNewJointAction(actions[0], guarded_accels, num_preds, num_dims)
            chosen_action_index = 0

        return chosen_action_index    


    def ensureValidInitState(self, single_obs):
        if not self.env.start_of_episode:
            return single_obs

        max_init_resamples = self.config['env_config']['max_init_resamples']
        for _ in range(max_init_resamples):
            pred_states = self.getCurrentPredStates(single_obs, self.env.num_preds, self.env.num_dims)
            prey_st = self.env.prey.position.tolist() + self.env.prey.velocity.tolist()

            if self.checkInvInitState(pred_states, prey_st, self.env.num_preds):
                self.env.start_of_episode = False
                return single_obs

            raw_obs = self.env.resamplePredatorsForShieldInit()
            single_obs = self.envObsToTensor(raw_obs)

        raise RuntimeError('Unable to sample shield-feasible initial predator state')

    def envObsToTensor(self, obs):
        return obs_as_tensor(self.batchEnvObs(obs), self.policy.device)

    def batchEnvObs(self, obs):
        return {
            key: np.expand_dims(value, axis=0)
            for key, value in obs.items()
        }

    def tensorObsToNumpyObs(self, obs):
        return {
            key: value.detach().cpu().numpy()
            for key, value in obs.items()
        }


    def getCurrentPredStates(self, single_obs, num_preds, num_dims):
        pred_states = []
        for pred_idx in range(num_preds):
            current_position = single_obs['pred_posz'][:, pred_idx * num_dims:(pred_idx + 1) * num_dims].cpu().numpy()
            current_velocity = single_obs['agent_velocities'][:, pred_idx * num_dims:(pred_idx + 1) * num_dims].cpu().numpy()
            # print('current_position', current_position,'current_velocity', current_velocity)
            current_state = np.concatenate((current_position, current_velocity), axis=1).squeeze().tolist()
            pred_states.append(current_state)
        return pred_states

    def checkInvInitState(self, pred_states, prey_st, num_preds):
        init_state_ok = True
        for pred_idx, current_state in enumerate(pred_states):
            se = solnExists(current_state, prey_st, self.env.STEPS_BOUND)
            if se: print(f'solnExists pred {pred_idx}', current_state, prey_st, self.env.STEPS_BOUND)
            else:
                print(f'***WARNING: no solution from pred {pred_idx}', current_state, prey_st, self.env.STEPS_BOUND)
                init_state_ok = False
            if self.env.TRACKING_PREY:
                se_t = solnExistsTrack(current_state, prey_st)
                if se_t: print(f'solnExistsTrack pred {pred_idx}', current_state, prey_st)
                else:
                    print(f'***WARNING: no tracking soln from pred {pred_idx}', current_state, prey_st)
                    init_state_ok = False
        if self.env.DOING_SEP and num_preds > 1:
            for i in range(num_preds):
                for j in range(i + 1, num_preds):
                    se_d = solnExistsDist(pred_states[i], pred_states[j])
                    if se_d: print(f'solnExistsDist pair {i},{j}', pred_states[i], pred_states[j])
                    else:
                        print(f'***WARNING: no separation soln from pair {i},{j}', pred_states[i], pred_states[j])
                        init_state_ok = False
        return init_state_ok

    def checkUnaryGuards(self, joint_acc, pred_states, prey_st, steps_remaining, num_preds, num_dims):
        guarded_accels = []
        # any_accel_overridden is needed b/c OK can sometimes return a specific action value rather than just a Bool, in which case a new modified tuple has to be constructed
        any_accel_overridden = False
        unary_guards_passed = True

        for pred_idx in range(num_preds):
            pred_acc = joint_acc[pred_idx * num_dims:(pred_idx + 1) * num_dims]

            res = OK(pred_acc, pred_states[pred_idx], prey_st, steps_remaining)
            if not res:
                unary_guards_passed = False
                break

            if res != True and len(res) == num_dims:
                acc_to_use = res
                any_accel_overridden = True
            else:
                acc_to_use = pred_acc

            if self.env.TRACKING_PREY:
                if not OKTrack(acc_to_use, pred_states[pred_idx], prey_st):
                    unary_guards_passed = False
                    break

            guarded_accels.append(acc_to_use)

        return unary_guards_passed, guarded_accels, any_accel_overridden

    def __clippedNextStateSatisfiesInvariant(self, actual_next_state, prey_st, num_preds):
        pred_positions = actual_next_state['pred_positions']
        pred_velocities = actual_next_state['pred_velocities']

        if self.env.DOING_SEP and num_preds > 1:
            for i in range(num_preds):
                for j in range(i + 1, num_preds):
                    actual_st_i = pred_positions[i].tolist() + pred_velocities[i].tolist()
                    actual_st_j = pred_positions[j].tolist() + pred_velocities[j].tolist()
                    if not solnExistsDist(actual_st_i, actual_st_j):
                        return False

        if self.env.TRACKING_PREY:
            for i in range(num_preds):
                actual_st_i = pred_positions[i].tolist() + pred_velocities[i].tolist()
                if not solnExistsTrack(actual_st_i, prey_st):
                    return False

        return True

    def checkBinaryGuards(self, guarded_accels, pred_states, num_preds):
        joint_acc = np.asarray(guarded_accels, dtype=float).reshape(num_preds * self.env.num_dims)
        transition_state = self.env.getCalculatedAndActualNextState(
            joint_acceleration=joint_acc,
            pred_states=pred_states,
        )
        actual_next_state = transition_state['actual_next_state']
        which_preds_clipped = transition_state['which_preds_clipped']

        if np.any(which_preds_clipped) and not self.allow_clip_bypass_in_selector:
            return False

        if np.any(which_preds_clipped):
            self.clip_bypass_used += 1

            # Clip-involved moves are validated against the invariant on actual post-clip states. That is, will the clipped position ensure a future move always possible
            prey_st = self.env.prey.position.tolist() + self.env.prey.velocity.tolist()
            if not self.__clippedNextStateSatisfiesInvariant(actual_next_state, prey_st, num_preds):
                return False
            return True

        # no clipping, standard okdist check.
        if self.env.DOING_SEP and num_preds > 1:
            for i in range(num_preds):
                for j in range(i + 1, num_preds):
                    if not OKDist(guarded_accels[i], pred_states[i], guarded_accels[j], pred_states[j]):
                        return False

        return True

    '''in bounded reachability case at the last step shield returns the reqd action'''
    def composeNewJointAction(self, action, guarded_accels, num_preds, num_dims):
        composed = list(action.tolist())
        for pred_idx in range(num_preds):
            acc = guarded_accels[pred_idx]
            composed[pred_idx * num_dims:(pred_idx + 1) * num_dims] = self.acclerationToAction(acc)
        return composed

    def findActionForEachPredSeparately(self, actions, pred_states, prey_st, steps_remaining, num_preds, num_dims):
        chosen_per_pred = {}

        for pred_idx in range(num_preds):
            for num, action in enumerate(actions):
                joint_acc = self.env.actionToAcceleration(action.tolist())
                pred_acc = joint_acc[pred_idx * num_dims:(pred_idx + 1) * num_dims]

                res = OK(pred_acc, pred_states[pred_idx], prey_st, steps_remaining)
                if not res:
                    continue

                # OK may return a replacement acceleration instead of True
                if res != True and len(res) == num_dims:
                    acc_to_use = res
                    is_replaced = True
                else:
                    acc_to_use = pred_acc
                    is_replaced = False

                if self.env.TRACKING_PREY:
                    if not OKTrack(acc_to_use, pred_states[pred_idx], prey_st):
                        continue

                chosen_per_pred[pred_idx] = (acc_to_use, is_replaced, num)
                break

        # If any predator exhausted all samples, signal failure
        if len(chosen_per_pred) < num_preds:
            return None

        guarded_accels = [chosen_per_pred[pred_idx][0] for pred_idx in range(num_preds)]
        if not self.checkBinaryGuards(guarded_accels, pred_states, num_preds):
            return None

        return chosen_per_pred


    #actions are normalized values
    def acclerationToAction(self, acc):
        action = [i / self.env.max_acceleration for i in acc]
        # maybe assert (self.action_space.low <= elt for elt in action).all() and .. 
        # SN: tried assert self.action_space.log <= action, etc but didnt like that either
        # assert all(self.action_space.low <= elt and elt <= self.action_space.high for elt in action), "provided acceleration value outside action space limits"
        return action


    '''UNUSED?
    def getRandomAction(self, single_obs):
        # Get the actions
        actionss = self.sampleFromUniformDistrib(single_obs)
        chosen_action_index = self.selectOKAction(actionss, single_obs)  

        if chosen_action_index < 0: #was unable to find an ok action by random selection, so just pick the first action - kind of silly, need to fix
            chosen_action_index = 0
        # if (num == self.num_chances - 1):
            self.n_agent_fails += 1
            if self.n_agent_fails % 1 == 0:
                print('n_agent_fails', self.n_agent_fails)

        action_per_pred = actionss[chosen_action_index]

        values, log_probs, _ = \
            self.policy.evaluate_actions(
                single_obs,
                # 1st dim is just one b/c have only one env
                th.tensor(action_per_pred.reshape(1, self.env.num_preds * self.env.num_dims)).to(self.policy.device))
        return action_per_pred, values, log_probs
    '''

    '''Draw <num_chances> random actions 1 per pred from uniform distribution'''
    def sampleFromUniformDistrib(self, single_obs):
        low = self.env.action_space.low
        high = self.env.action_space.high
        actionss = np.random.uniform(
            low, high, size=(self.num_chances, self.env.num_preds * self.env.num_dims))
        return actionss
