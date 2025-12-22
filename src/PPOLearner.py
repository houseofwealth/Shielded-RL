from stable_baselines3 import PPO
from gymnasium import spaces
import time
from stable_baselines3.common.utils import safe_mean, obs_as_tensor
import numpy as np
import torch as th
from src.ActionSelector import ActionSelector

class PPOLearner(PPO):
    def __init__(
        self,
        config,
        tensorboard_log=None,
        target_kl=None,
        ):  
        # this instantiates env, policy, and other member vars from parent class OnPolicyAlgorithm
        super().__init__(
                        config['policy_class'],
                        env = config['env_class'](config['env_config']),
                        tensorboard_log=tensorboard_log,
                        learning_rate=config['learning_rate'],
                        target_kl=target_kl
                        #device=config['device']
        )
        self.use_shield = config['use_shield']
        #policy member var gets initialized by the parent class (PPO) constructor..
        self.action_selector = ActionSelector(config, self.policy, self.env)

    '''Overrides parent method in PPO (which just calls its OPA's learn method this is same as that with couple of changes).
    learning = collect rollouts (episode runs) and then train the NNs on them'''
    def learn(
        self,
        config,
        callback=None,
        log_interval=1,
        tb_log_name="OnPolicyAlgorithm",
        reset_num_timesteps=True,
        progress_bar=False
        ):

        iteration = 0

        #setup_learn may modify both the given max_learning_steps and the callback. It also initializes some member vars such as _last_obs
        max_learning_steps, callback = \
            self._setup_learn(  config['max_learning_steps'],
                                callback,
                                reset_num_timesteps,
                                tb_log_name,
                                progress_bar )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        #num_timesteps is incremented in collect_rollouts every time an env step is taken. Its incremented by the # of envs. env has its own total_timesteps var so each env can seperately keep track
        while self.num_timesteps < max_learning_steps:
            #*any reason to change buffer size?  b/c SB3 code doesnt have it
            # self.rollout_buffer.buffer_size = original_buffer_size

            print('\nstarting new rollout buffer after num_timesteps =', self.num_timesteps)      
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1      #just tracks # rollouts completed 
            self._update_current_progress_remaining(self.num_timesteps, max_learning_steps)
            self.log(iteration, log_interval)
            self.train()

        callback.on_training_end()
        return self


    '''Overrides method in OnPolicyAlgorithm (parent of PPO) code is almost literally the same. Only difference is that it calls getActionForEachPred() instead of policy()'''
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps,
    ):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        tot_rew = 0
        while n_steps < n_rollout_steps: #and not all_predators_dead
            """ use_sde is false
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.policy.reset_noise(env.num_envs) """


            # base method in OPA simply calls policy() to get action, but we need to pass the action to the shield first. action_selector needs the obs but since that's probably a numpy array, it needs to get converted to a tensor on the correct device first. Since action selection is just inferencing, turn off gradient calcs
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                action_per_pred, values, log_probs = self.action_selector.getActionForEachPred(obs_tensor)

            clipped_actions = action_per_pred
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(action_per_pred, self.action_space.low, self.action_space.high)

            clipped_actions_for_vec_env = np.asarray([clipped_actions])

            '''StableBaselines.step() which calls actual env.step() turns rew and done and info into lists for the purposes of supporting vectorized envs (VecEnvs) which run multiple envs in parallel. 
            
            Also this is just 4 return values even though step() returns a list of 5 values, b/c its trying to be cnosistent with the old Gym API. To use the new Gymnasium API (5-value return), replace the wrapper with a Gymnasium-compatible one: 
            from gymnasium.vector import SyncVectorEnv

            def make_env():
                return MyEnv(config)

            env = SyncVectorEnv([make_env for _ in range(num_envs)])            
            This ensures that env.step() returns 5 values.'''
            new_obs, rewards, dones, infos = env.step(clipped_actions_for_vec_env)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                action_per_pred = action_per_pred.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                action_per_pred,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # "runs" the V network to obtain current value estimates for the state"
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        #determines what the GAE should be from the rewards. Note - no updates yet, that happens in train()
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True
    

    '''this info sent to tensorboard. The / convention just creates categories on tensorboard'''
    def log(self, iteration, log_interval):
        if iteration % log_interval == 0:
            # fps = int(self.num_timesteps / (time.time() - self.start_time))
            # Calculate mean reward for all envs
            mean_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            mean_length = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
            # self.logger.record("time/iterations", self._n_updates, exclude="tensorboard")
            # self.logger.record("time/fps", fps, exclude="tensorboard")
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)
            self.logger.dump(self.num_timesteps)
