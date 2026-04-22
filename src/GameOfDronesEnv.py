from gymnasium import spaces, Env
import numpy as np
from agents.predator import Predator
from agents.prey import Prey

'''Modeled loosly on MPE2 env in github. Private methods start with __ to indicate they should only be called within env. Public methods (__init__, reset, step) are called by SB3 training loop and actionToAcceleration is called by the ActionSelector in PPOLearner.'''

class GameOfDronesEnv(Env):
    # ------------------------------------------------------------------
    # Public interface  (called by SB3 training loop)
    # ------------------------------------------------------------------
    def __init__(self, config):

        assert isinstance(config, dict), 'Config should be a dict!'
        env_config = config['env_config']
        assert isinstance(env_config, dict), 'env_config in config should be a dict!'
        # Load config into a member dict - this allows entries in the config to be accessed as if they were fields of env. any changes to these dict entries doesnt impact config (as expected)
        for key, value in env_config.items():
            self.__dict__[key] = value

        self.predators = []
        for _ in range(self.num_preds):
            self.predators.append(Predator(num_dims=self.num_dims, 
                                            workspace_size=self.workspace_size,
                                            doing_obstacles=self.DOING_OBSTACLES,
                                            doing_geofence = self.GEOFENCING,
                                            LObs=self.LObs, RObs=self.RObs, TObs=self.TObs, BObs=self.BObs))

        self.prey = Prey(num_dims=self.num_dims, workspace_size=self.workspace_size)
       
        self.__mkSpaces()             #action space and obs space
        # self.reset()              #not needed, training loop in SB3 calls it before each episode
        self.hitGeoFenceOrObs = False
        self.n_hit_geofence_or_obs = 0
        self.n_prey_caught = 0
        self.n_tot_rew = 0
        self.n_bound_exceeded = 0
        self.n_truncated = 0
        self.n_p_e = 0
        self.prev_potential = 0.0
        self.gamma = config['gamma']


    '''called at the start of each episode to reset env state'''
    def reset(self, options=None, seed=0):
        self.__resetCountsAndFlags()
        self.__initPreyPosAndVel()
        self.__initPredPoszAndVels()

        obs, _ = self.__getObservation()
        # cant put this here b/c of cyclic dependency b/c shield <- model_1pt <- config <- env so set a flag instead    
        # se = solnExistsPy(self.predators[0].position + self.predators[0].velocity, 
        #                   self.prey.position + self.prey.velocity, 
        #                   self.env.STEPS_BOUND)
        
        self.start_of_episode = True
        return obs, {}               #SB3 expects these return values 
      

    def step(self, action): #SB3 code won't allow this b/c its called internally and not expering extra params is_random_action=False):
        # print('env_action', action)
        # breakpoint()
        self.num_episode_steps += 1
        # self.num_steps_in_this_env += 1
        acceleration = self.actionToAcceleration(action)

        for pred_num, predator in enumerate(self.predators):
            if predator.is_live:
                start_idx = pred_num * self.num_dims
                end_idx = (pred_num + 1) * self.num_dims
                predator.move(acceleration[start_idx:end_idx], self.STEP_SIZE)
                predator.clipPosToWSBoundary()    
        
        #prey has no acceleration!
        self.prey.move(0, self.STEP_SIZE)   #prey moves w/ constant velocity
        self.prey.clipPosToWSBoundary()

        obs, task_failed = self.__getObservation()
        if self.DOING_BOUNDED: 
            bound_exceeded = self.n_steps_to_bound > (self.STEPS_BOUND - 1) #wy was this 2??
            if bound_exceeded: 
                self.n_bound_exceeded += 1
                if self.n_bound_exceeded%100 == 0: print('n_bound_exceeded', self.n_bound_exceeded)
            task_failed = task_failed or bound_exceeded
        
        truncated = self.num_episode_steps >= self.MAX_EPISODE_STEPS
        prey_escaped = self.prey.at(self.base_position, self.PREY_SIZE)
        if prey_escaped: 
            self.n_p_e += 1
            if self.n_p_e%100 == 0: print('n_prey_escaped', self.n_p_e)
        if truncated: 
            self.n_truncated += 1
            # [seems to be same as n_prey_escaped] if self.n_truncated%10 == 0: print('n_truncated', self.n_truncated)

        '''terminated = natural end of episode (task_failed, pred caught prey, or prey escaped); truncated = cut short by time limit.
        Kept separate so VecEnv can set TimeLimit.truncated in infos, enabling correct GAE bootstrapping in collect_rollouts.
        task_failed if preds collided (w/ each other, geofence, bound exceeded, etc)'''
        prey_caught = self.__aPredCaughtPrey()
        terminated = task_failed or \
                prey_caught or \
                prey_escaped
            
        # reward = self.getReward(obs['pred_posz'], obs['target_pos'], done)
        # pass terminated or truncated so __getReward still fires terminal rewards on timeout (preserves existing behaviour)
        reward = self.__getReward(terminated or truncated, task_failed)
        self.n_tot_rew += reward
        self.n_steps_to_bound += 1
        # SB3 wrapper on step() requires all these 5 things
        return obs, reward, terminated, truncated, {'task_failed': task_failed, 'prey_caught': prey_caught, 'prey_escaped': prey_escaped}


    def printSummary(self, n_runs):
        print(f'\n--- Summary over {n_runs} runs ---')
        print(f'Prey caught:          {self.n_prey_caught}')
        print(f'Prey escaped:         {self.n_p_e}')
        print(f'Truncated (timeout):  {self.n_truncated}')
        print(f'Geofence/obs hits:    {self.n_hit_geofence_or_obs}')
        print(f'Bound exceeded:       {self.n_bound_exceeded}')

    def actionToAcceleration(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action * self.max_acceleration


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def __resetCountsAndFlags(self):
        self.num_episode_steps = 0
        self.n_steps_to_bound = 0

    def __initPreyPosAndVel(self):
        self.prey.is_live = True
        if isinstance(self.initial_prey_pos, str) and self.initial_prey_pos == 'random':
            pos = self.target_pos_space.sample()
            pos[-1] = 0  # Set last dimension to ground b/c workspace starts at ground
            self.prey.position = pos
        else:
            self.prey.position = self.initial_prey_pos

        if  isinstance(self.initial_prey_velocity, str) and self.initial_prey_velocity == 'random':
            self.prey.velocity = self.agent_velocities_space.sample()
        else:
            self.prey.velocity = self.initial_prey_velocity


    def __initPredPoszAndVels(self):
        preds_collided = True; pred_landed_outside_ws = True; pred_landed_in_obs = True
        while preds_collided or pred_landed_outside_ws or pred_landed_in_obs:
            #reset all the flags at the start of each iteration in case a flag gets unset but another remains true in a given iteration
            preds_collided = True; pred_landed_outside_ws = True; pred_landed_in_obs = True
            positions = self.pred_posz_space.sample()  # produces for say a 2 pred 3d space a 6 element vector [-2, 8, -1, 4, -3, 10]
            for pred_num, predator in enumerate(self.predators):
                start = pred_num * self.num_dims
                end = (pred_num + 1) * self.num_dims
                #slice into the positions array to get the pos for this pred
                #TBD: this all needs to go in a Preadtor reset method
                predator.position = positions[start:end]
                predator.velocity = np.zeros(self.num_dims)
                predator.hit_geofence = False
                predator.is_live = True
                predator.shield_was_used_in_step = False

            # check it didnt place preds on top of each other or outside WS or inside obstacle
            collisions = self.__findAndKillCollidedPreds()
            if not np.any(collisions):
                preds_collided = False
            if not any(predator.landedOutsideWS() for predator in self.predators):
                pred_landed_outside_ws = False
            if not any(predator.hitObs() for predator in self.predators):
                pred_landed_in_obs = False

    def __mkSpaces(self):
        total_num_preds = self.num_preds #+ self.num_fake_preds
        self.action_space = spaces.Box(-1, 1, shape=(self.num_dims * self.num_preds,))

        # agent_velocities: pred velocities + prey velocity
        self.agent_velocities_space = spaces.Box(
            -self.workspace_size , self.workspace_size , shape=(total_num_preds * self.num_dims + self.num_dims,)
        )

        # pred_posz: predz positions - eg for 2 preds in 3D would be a 6-D hyperrectangle, ranging from -10 to 10 in each dim
        self.pred_posz_space = spaces.Box(
            -self.workspace_size , self.workspace_size , shape=(total_num_preds * self.num_dims,))
        
        # target_pos: prey position
        self.target_pos_space = spaces.Box(
            -self.workspace_size , self.workspace_size , shape=(self.num_dims,)
        )

        obs_spaces = {
            'agent_velocities': self.agent_velocities_space,
            'pred_posz': self.pred_posz_space,
            'target_pos': self.target_pos_space,
            # 'dist_to_walls': self.dist_to_walls_space,
        }
        
        # Add "radar", ie distance to walls for each predator
        # For 2D, this is 4 values per predator (dist to left, right, bottom, top walls)
        # Note: currently only used for unshielded agent
        if False: #not self.use_shield and self.GEOFENCING:
            self.dist_to_walls_space = spaces.Box(
                0, 2 * self.workspace_size, shape=(self.num_preds * self.num_dims * 2,)
        )
            obs_spaces['dist_to_walls'] = self.dist_to_walls_space

        # Add distance to obstacle faces for each predator (same pattern as dist_to_walls)
        # For 2D: 4 values per predator (dist to LObs, RObs, BObs, TObs)
        # if not self.use_shield and self.DOING_OBSTACLES:
        #     self.dist_to_obs_space = spaces.Box(
        #         -2 * self.workspace_size, 2 * self.workspace_size, 
        #         shape=(self.num_preds * self.num_dims * 2,)
        #     )
        #     obs_spaces['dist_to_obs'] = self.dist_to_obs_space

        self.observation_space = spaces.Dict(obs_spaces)
    

    '''returns new system state and whether preds collided with each other, geofence, etc'''
    def __getObservation(self, is_random_action=False):
        _ = self.__findAndKillCollidedPreds()
        task_failed = self.__APredHitGeoFenceOrObs()
        if task_failed:
            # breakpoint()
            # if not is_random_action and self.use_shield:
                # print('***Error: hit geofence or obs even though action wasnt random')
            self.n_hit_geofence_or_obs += 1
            if self.n_hit_geofence_or_obs % 10 == 0: 
                print('# times hit geofence or obs', self.n_hit_geofence_or_obs)

        pred_posz = np.concatenate([predator.position for predator in self.predators])
        pred_velocities = np.concatenate([predator.velocity for predator in self.predators])
        agent_vels = np.concatenate([pred_velocities, self.prey.velocity])

        # # Calculate distance to walls for each predator
        # dist_to_walls = []
        # for predator in self.predators:
        #     pos = predator.position
        #     # Assumes workspace is centered at 0, from -workspace_size to +workspace_size
        #     # Order: dist to +x, -x, +y, -y, ... for each dimension
        #     for dim in range(self.num_dims):
        #         dist_to_walls.append(self.workspace_size - pos[dim])
        #         dist_to_walls.append(pos[dim] - (-self.workspace_size))
        
        obs = { 'agent_velocities': agent_vels,
                'pred_posz': pred_posz,
                'target_pos': self.prey.position,
                # 'dist_to_walls': np.array(dist_to_walls),
        }

        #these two provide the unsheided agent with "radar" to sense where the feofence and obstacles are.
        # if not self.use_shield and self.DOING_OBSTACLES:
        #     obs['dist_to_obs'] = np.concatenate([self.__mkDistToObstacleObs(p) for p in self.predators])
        # if False: #not self.use_shield and self.GEOFENCING:
        #     obs['dist_to_walls'] = np.concatenate([self.__mkDistToWallsObs(p) for p in self.predators])  
        # Distance to obstacle faces for each predator (same pattern as dist_to_walls)
        # Positive = outside obstacle on that side, negative = inside/past that face
        # In 2D: dist to LObs (from left), RObs (from right), BObs (from below), TObs (from above)
        # if self.DOING_OBSTACLES:
        #     dist_to_obs = []
        #     obs_bounds = [self.LObs, self.RObs, self.BObs, self.TObs]  # [x_lo, x_hi, y_lo, y_hi]
        #     for predator in self.predators:
        #         pos = predator.position
        #         # x dimension: distance to left and right faces of obstacle
        #         dist_to_obs.append(self.LObs - pos[0])   # negative if pred is left of obstacle (safe side)
        #         dist_to_obs.append(pos[0] - self.RObs)   # negative if pred is right of obstacle (safe side)
        #         # y dimension: distance to bottom and top faces of obstacle
        #         dist_to_obs.append(self.BObs - pos[1])   # negative if pred is below obstacle (safe side)
        #         dist_to_obs.append(pos[1] - self.TObs)   # negative if pred is above obstacle (safe side)
        #     obs['dist_to_obs'] = np.array(dist_to_obs, dtype=np.float32)

        return obs, task_failed
    
    def __mkDistToWallsObs(self, predator):
        pos = predator.position
        dist_to_walls = []
        # Assumes workspace is centered at 0, from -workspace_size to +workspace_size
        # Order: dist to +x, -x, +y, -y, ... for each dimension
        for dim in range(self.num_dims):
            dist_to_walls.append(self.workspace_size - pos[dim])  # distance to +ve wall
            dist_to_walls.append(pos[dim] - (-self.workspace_size))  # distance to -ve wall
        return np.array(dist_to_walls, dtype=np.float32)
    
    def __mkDistToObstacleObs(self, predator):
        # Distance to obstacle faces for each predator (same pattern as dist_to_walls)
        # Positive = outside obstacle on that side, negative = inside/past that face
        # In 2D: dist to LObs (from left), RObs (from right), BObs (from below), TObs (from above)
        pos = predator.position
        dist_to_obs = []
        # x dimension: distance to left and right faces of obstacle
        dist_to_obs.append(self.LObs - pos[0])   # negative if pred is left of obstacle (safe side)
        dist_to_obs.append(pos[0] - self.RObs)   # negative if pred is right of obstacle (safe side)
        # y dimension: distance to bottom and top faces of obstacle
        dist_to_obs.append(self.BObs - pos[1])   # negative if pred is below obstacle (safe side)
        dist_to_obs.append(pos[1] - self.TObs)   # negative if pred is above obstacle (safe side)
        return np.array(dist_to_obs, dtype=np.float32)
    
    '''check if any preds collided with each other and mark as dead'''
    def __findAndKillCollidedPreds(self):
        #Not an automatic fail since other preds may still be alive
        who_collided = self.__FindMutualPredCollisions()
        self.__killCollidedPredsAndResetThem(who_collided)
        return who_collided

  
    """Check if any 2 predators collide with each other. Not an automatic fail since other preds may still be alive """
    def __FindMutualPredCollisions(self):
      who_collided = np.zeros(self.num_preds, dtype=bool)
      live_preds = [pred for pred in self.predators if pred.is_live]
      for pred1_idx, pred1 in enumerate(live_preds):
          for pred2_relative_idx, pred2 in enumerate(live_preds[pred1_idx+1:]):
              if np.linalg.norm(pred1.position - pred2.position) < self.PRED_SIZE:
                  who_collided[pred1_idx] = True
                  who_collided[pred1_idx + pred2_relative_idx  + 1] = True
      return who_collided
    
    
    '''OTOH, if any pred hit geofence or obstacle, its game over'''
    def __APredHitGeoFenceOrObs(self):
      return any(predator.hitGeoFence() or predator.hitObs() for predator in self.predators)


    # set their is_live to false, position to [-1] * num_dims, velocity to [0] * num_dims
    def __killCollidedPredsAndResetThem(self, who_collided):
      for i, pred in enumerate(self.predators):
          if who_collided[i]:
              pred.is_live = False
              #reset pred position after collision - why is this needed?
              pred.position = np.array([-1]*self.num_dims)
              pred.velocity = np.zeros(self.num_dims)


    def __getReward(self, done, task_failed):
        shield_used = False
        if self.use_shield:
            shield_used = any(getattr(pred, 'shield_was_used_in_step', False) for pred in self.predators)

        if done:
            if self.__aPredCaughtPrey():
                reward = 1
            else:
                # reward = -1
                #*TBD: why is this just 1st predator. Need to figure out how to do this for multiagant
                # if self.predators[0].hitGeoFence() or self.predators[0].hitObs(): #hit_geofence would have been set in step() after clipping
                if task_failed:
                    reward = -1
                else:
                    reward = -1
                    # # reward proportional to how close the predator got to the prey
                    # # predator_pos = self.predators[0].position  # Assuming single predator
                    # prey_pos = self.prey.position
                    # normalized_distance = self.predators[0].normalizedDistanceTo(prey_pos)
                    # # reward = -normalized_distance  # Negative reward proportional to normalized distance
        else: #episode not finished, no biscuit for you yet. *TBD?: small reward for surviving or getting close to prey?
            # 1. simplest: reward = 0 # does slightly better?
            # 2. then tried:
            # # Small constant time penalty to encourage efficiency
            # reward = -0.01
            # # Add a larger penalty if the shield had to intervene
            # if shield_used:
            #     reward -= 0.1
            # 3.
            if True: #self.use_shield:
                reward = 0
            else:
                #PBR scheme from Mn at al, 1999
                prey_pos = self.prey.position
                #this will be between 0 (its as far from prey as poss) and 1 (its at prey)
                normalized_distance = self.predators[0].normalizedDistanceTo(prey_pos, min_sep=self.KILL_RADIUS)
                potential = 1 - normalized_distance
                reward = self.gamma * potential - self.prev_potential  # only potential increase gets +ve reward, to encourage getting closer to prey
                self.prev_potential = potential

        if self.use_shield and shield_used:
            for pred in self.predators:
                pred.shield_was_used_in_step = False

        return reward
    

    '''Who should be responsible for determining caught - Predator or Env? Decided it was Env's responsibility as here it is Env that makes determinations by taking all context into account'''
    def __aPredCaughtPrey(self):
        for predator in self.predators:
            # if self.entityCollidedAt(predator, self.prey.position):
            if predator.at(self.prey.position, self.KILL_RADIUS):
                self.prey.is_live = False
                self.n_prey_caught += 1
                if self.n_prey_caught%100 == 0: print('n_prey_caught',self.n_prey_caught)
                return True
        return False
