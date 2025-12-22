from abc import ABC
import numpy as np
from ipdb import set_trace

class BaseAgent(ABC):
    def __init__(self, num_dims, workspace_size, doing_obstacles=False, doing_geofence=False,
                 LObs=0, RObs=0, TObs=0, BObs=0):
        # self._max_velocity = max_velocity
        self._workspace_size = workspace_size
        self.num_dims = num_dims
        self._position = np.zeros(num_dims)
        self._velocity = np.zeros(num_dims)
        self._is_live = True
        self.hit_geofence = False
        # self.n_hit_geofence_or_obs = 0
        self.doing_obstacles = doing_obstacles
        self.doing_geofence = doing_geofence
        self.LObs = LObs
        self.RObs = RObs
        self.TObs = TObs
        self.BObs = BObs


    # @property
    # def max_velocity(self):
    #     return self._max_velocity

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, new_position):
        self._position = new_position
    
    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, new_velocity):
        self._velocity = new_velocity
    
    @property
    def is_live(self):
        return self._is_live
    
    @is_live.setter
    def is_live(self, new_value:bool):
        self._is_live = new_value


    '''clips to the workspace, in addition sets a flag if the workspace boundary was hit (needed for geofencing)'''
    def clipPosToWSBoundary(self, isreset=False):
        all_but_last = self._position[:-1]
        last = self._position[-1]

        #WS is always -ws_size .. +ws_size except in last dim where its 0..ws_size, which is why last pos needs special handling see next block
        # violations = (all_but_last < -self._workspace_size) | (all_but_last > self._workspace_size)
        # last_violation = (last < 0) or (last > self._workspace_size)

        #NOTE: hack fix it
        # if self.__class__.__name__ == "Predator" and (np.any(violations) or last_violation):
        if self.__class__.__name__ == "Predator" and self.landedOutsideWS():
            # breakpoint()
            self.hit_geofence = True

        # account for last dim having different range 0..ws_size 
        clipped_all_but_last = np.clip(all_but_last, -self._workspace_size, self._workspace_size)
        clipped_last = np.clip(last, 0, self._workspace_size)
        self._position = np.concatenate([clipped_all_but_last, [clipped_last]])
    
    def landedOutsideWS(self):
        #checks if any of the elements of the position falls outside the workspace boundary. WS is always -ws_size .. +ws_size except in last dim where its 0..ws_size, which is why last pos needs special handling
        last_element = self._position[self.num_dims - 1]
        out_of_bounds =  any( i < -self._workspace_size or i > self._workspace_size \
                             for i in self._position[0:(self.num_dims - 1)]) or \
                        last_element < 0 or \
                        last_element > self._workspace_size
        return out_of_bounds


    #TBD: fix for 3D
    def hitObs(self): 
        last_element = self._position[self.num_dims - 1]
        #checks if any of the elements of the position falls in the obstacle area
        if self.doing_obstacles: 
            hit_obstacle = \
              any(self.LObs <= i and i <= self.RObs for i in self._position[0:(self.num_dims - 1)]) and \
              self.BObs <= last_element and\
              last_element <= self.TObs
        else: 
            hit_obstacle = False
        return (self.doing_obstacles and hit_obstacle)


    #NOTE: ***assumes that hit_geofence flag has been set by someone, eg clipPosToWSBoundary()
    def hitGeoFence(self): 
        return self.doing_geofence and self.hit_geofence
        
    def at(self, pos, min_sep):
        distance = np.linalg.norm(self._position - pos)
        return distance < min_sep

    def normalizedDistanceTo(self, pos):
        distance = np.linalg.norm(self._position - pos)
        max_distance = np.sqrt(self._workspace_size**2 * self.num_dims)
        normalized_distance = distance / max_distance
        return normalized_distance
    
    def move(self, acceleration, step_size):
        if not self.is_live:
            # breakpoint()
            return
        #TODO used to cap the velocity to reppresent terminal velocity
        delta_v = acceleration * step_size
        self.old_position = self._position.copy()
        self.old_velocity = self._velocity.copy()

        self._position = self._position + (self._velocity * step_size) + (delta_v/2 * step_size)
        self._velocity = self._velocity + (acceleration * step_size)
