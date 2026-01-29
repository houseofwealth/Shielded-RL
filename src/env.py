from gymnasium import spaces, Env
import numpy as np
from agents.predator import Predator
from agents.prey import Prey

'''Modeled loosly on MPE2 env in github'''

class PredPreyEnv(Env):
    def __init__(self, config):

        assert isinstance(config, dict), 'Config should be a dict!'

        # Load config into a member dict - this allows entries in the config to be accessed as if they were fields of env. any changes to these dict entries doesnt impact config (as expected)
        for key, value in config.items():
            self.__dict__[key] = value

        self.predators = []
        for _ in range(self.num_preds):
            self.predators.append(Predator(num_dims=self.num_dims, 
                                            workspace_size=self.workspace_size,
                                            doing_obstacles=self.DOING_OBSTACLES,
                                            doing_geofence = self.GEOFENCING,
                                            LObs=self.LObs, RObs=self.RObs, TObs=self.TObs, BObs=self.BObs))

        self.prey = Prey(num_dims=self.num_dims, workspace_size=self.workspace_size)
       
        self.mkSpaces()             #action space and obs space
        # self.reset()              #not needed, training loop in SB3 calls it before each episode
        self.hitGeoFenceOrObs = False
        self.n_hit_geofence_or_obs = 0
        self.n_prey_caught = 0
        self.n_tot_rew = 0
        self.n_bound_exceeded = 0
        self.n_truncated = 0
        self.n_p_e = 0

    def mkSpaces(self):
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
        
        self.observation_space = spaces.Dict({
            'agent_velocities': self.agent_velocities_space,
            'pred_posz': self.pred_posz_space,
            'target_pos': self.target_pos_space,
        })

    '''called at the start of each episode to reset env state'''
    def reset(self, options=None, seed=0):
        self.resetCountsAndFlags()
        self.initPreyPosAndVel()
        self.initPredPoszAndVels()

        obs, _ = self.getObservation()
        # cant put this here b/c of cyclic dependency b/c shield <- model_1pt <- config <- env so set a flag instead    
        # se = solnExistsPy(self.predators[0].position + self.predators[0].velocity, 
        #                   self.prey.position + self.prey.velocity, 
        #                   self.env.STEPS_BOUND)
        
        self.start_of_episode = True
        return obs, {}               #SB3 expects these return values 

    def resetCountsAndFlags(self):
        self.num_episode_steps = 0
        self.n_steps_to_bound = 0


    def initPreyPosAndVel(self):
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


    def initPredPoszAndVels(self):
        preds_collided = True; pred_landed_outside_ws = True
        while preds_collided or pred_landed_outside_ws:
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

            # check it didnt place preds on top of each other or outside WS
            collisions = self.findAndKillCollidedPreds()
            if not np.any(collisions):
                preds_collided = False
            if not any(predator.landedOutsideWS() for predator in self.predators):
                pred_landed_outside_ws = False
      

    def step(self, action):
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

        obs, task_failed = self.getObservation()
        if self.DOING_BOUNDED: 
            bound_exceeded = self.n_steps_to_bound > (self.STEPS_BOUND - 1) #wy was this 2??
            if bound_exceeded: 
                self.n_bound_exceeded += 1
                if self.n_bound_exceeded%100 == 0: print('n_bound_exceeded', self.n_bound_exceeded)
            task_failed = task_failed or bound_exceeded
        
        truncated = self.num_episode_steps >= self.MAX_EPISODE_STEPS
        prey_escaped = self.prey.at(self.base_position, self.MAX_ACCEPTABLE_RADIUS)
        if prey_escaped: 
            self.n_p_e += 1
            if self.n_p_e%10 == 0: print('n_prey_escaped', self.n_p_e)
        if truncated: 
            self.n_truncated += 1
            # [seems to be same as n_prey_escaped] if self.n_truncated%10 == 0: print('n_truncated', self.n_truncated)

        '''done just means end the episode here, task_failed if preds collided (w/ each other, geofence, bound exceeded, etc), it got truncated if it took too long'''
        done =  task_failed or \
                self.aPredCaughtPrey() or \
                truncated or \
                prey_escaped
            
        # reward = self.getReward(obs['pred_posz'], obs['target_pos'], done)
        reward = self.getReward(done)
        self.n_tot_rew += reward
        self.n_steps_to_bound += 1
        # SB3 wrapper on step() requires all these 5 things
        return obs, reward, done, truncated, {'task_failed': task_failed}
    

    '''returns new system state and whether preds collided with each other, geofence, etc'''
    def getObservation(self):
        _ = self.findAndKillCollidedPreds()
        task_failed = self.APredHitGeoFenceOrObs()
        if task_failed:
            # breakpoint()
            self.n_hit_geofence_or_obs += 1
            if self.n_hit_geofence_or_obs % 100 == 0: 
                print('# times hit geofence or obs', self.n_hit_geofence_or_obs)

        pred_posz = np.concatenate([predator.position for predator in self.predators])
        pred_velocities = np.concatenate([predator.velocity for predator in self.predators])
        agent_vels = np.concatenate([pred_velocities, self.prey.velocity])

        obs = { 'agent_velocities': agent_vels,
                'pred_posz': pred_posz,
                'target_pos': self.prey.position,
        }
        return obs, task_failed
    
            
    '''check if any preds collided with each other and mark as dead'''
    def findAndKillCollidedPreds(self):
        #Not an automatic fail since other preds may still be alive
        who_collided = self.FindMutualPredCollisions()
        self.killCollidedPredsAndResetThem(who_collided)
        return who_collided

  
    """Check if any 2 predators collide with each other. Not an automatic fail since other preds may still be alive """
    def FindMutualPredCollisions(self):
      who_collided = np.zeros(self.num_preds, dtype=bool)
      live_preds = [pred for pred in self.predators if pred.is_live]
      for pred1_idx, pred1 in enumerate(live_preds):
          for pred2_relative_idx, pred2 in enumerate(live_preds[pred1_idx+1:]):
              if np.linalg.norm(pred1.position - pred2.position) < self.AT_TARGET_RADIUS:
                  who_collided[pred1_idx] = True
                  who_collided[pred1_idx + pred2_relative_idx  + 1] = True
      return who_collided
    
    # def APredHitGeoFenceOrObs(self):
    #   for pred in self.predators:
    #     if pred.hitGeoFence() or pred.hitObs():  #hit_geofence would have been set in step() after clipping
    #         # breakpoint()
    #         self.n_hit_geofence_or_obs += 1
    #         if self.n_hit_geofence_or_obs % 100 == 0: print('# times hit geofence or obs', self.n_hit_geofence_or_obs)
    #         return True      #returning true ends the episode by setting done
    #   return False
    
    '''OTOH, if any pred hit geofence or obstacle, its game over'''
    def APredHitGeoFenceOrObs(self):
      return any(predator.hitGeoFence() or predator.hitObs() for predator in self.predators)


    # set their is_live to false, position to [-1] * num_dims, velocity to [0] * num_dims
    def killCollidedPredsAndResetThem(self, who_collided):
      for i, pred in enumerate(self.predators):
          if who_collided[i]:
              pred.is_live = False
              #reset pred position after collision - why is this needed?
              pred.position = np.array([-1]*self.num_dims)
              pred.velocity = np.zeros(self.num_dims)


    def getReward(self, done):
        shield_used = False
        if self.use_shield:
            shield_used = self.predators[0].shield_was_used_in_step if hasattr(self.predators[0], 'shield_was_used_in_step') else False

        if done:
            if self.aPredCaughtPrey():
                reward = 1
            else:
                # reward = -1
                if self.predators[0].hitGeoFence() or self.predators[0].hitObs():  #hit_geofence would have been set in step() after clipping
                    reward = -1
                else:
                    # reward proportional to how close the predator got to the prey
                    # predator_pos = self.predators[0].position  # Assuming single predator
                    prey_pos = self.prey.position
                    # distance = np.linalg.norm(predator_pos - prey_pos)
                    # max_distance = np.sqrt(self.workspace_size**2 * self.num_dims)
                    # normalized_distance = distance / max_distance
                    # prey_pos = self.prey.position
                    normalized_distance = self.predators[0].normalizedDistanceTo(prey_pos)
                    reward = -normalized_distance  # Negative reward proportional to normalized distance
        else: #episode not finished, no biscuit for you yet. *TBD?: small reward for surviving or getting close to prey?
            # reward = 0 # does slightly better?
            # # Small constant time penalty to encourage efficiency
            reward = -0.01
            # # Add a larger penalty if the shield had to intervene
            # if shield_used:
            #     reward -= 0.1

        if self.use_shield and shield_used:
            self.predators[0].shield_was_used_in_step = False

        return reward


    def actionToAcceleration(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action * self.max_acceleration
    

    '''Who should be responsible for determining caught - Predator or Env? Decided it was Env's responsibility as here it is Env that makes determinations by taking all context into account'''
    def aPredCaughtPrey(self):
        for predator in self.predators:
            # if self.entityCollidedAt(predator, self.prey.position):
            if predator.at(self.prey.position, self.MAX_ACCEPTABLE_RADIUS):
                self.prey.is_live = False
                self.n_prey_caught += 1
                if self.n_prey_caught%100 == 0: print('n_prey_caught',self.n_prey_caught)
                return True
        return False
