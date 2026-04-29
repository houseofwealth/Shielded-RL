from abc import ABC
import numpy as np
from ipdb import set_trace

class BaseAgent(ABC):
    def __init__(self, num_dims, workspace_size, doing_obstacles=False, doing_geofence=False,
                 LObs=0, RObs=0, TObs=0, BObs=0):
        # self._max_velocity = max_velocity
        self._workspace_size = workspace_size
        self.num_dims = num_dims
        self.old_position = np.zeros(num_dims)
        self.position = np.zeros(num_dims)
        self.velocity = np.zeros(num_dims)
        self._is_live = True
        self.hit_geofence = False
        # self.n_hit_geofence_or_obs = 0
        self.doing_obstacles = doing_obstacles
        self.doing_geofence = doing_geofence
        self.LObs = LObs
        self.RObs = RObs
        self.TObs = TObs
        self.BObs = BObs
        

    """ OLD coordinate-wise clip (retained for reference) ---
    def clipPosToWSBoundary(self, isreset=False):
        if self.__class__.__name__ == "Predator" and self.landedOutsideWS():
            self.hit_geofence = True
        all_but_last = self.position[:-1]
        last = self.position[-1]
        clipped_all_but_last = np.clip(all_but_last, -self._workspace_size, self._workspace_size)
        clipped_last = np.clip(last, 0, self._workspace_size)
        new_position = np.concatenate([clipped_all_but_last, [clipped_last]])
        clipped_mask = (new_position != self.position)
        if np.any(clipped_mask):
            self.velocity = self.velocity.copy()
            self.velocity[clipped_mask] = 0.0
        self.position = new_position
    """

    '''Clips position to workspace boundary using ray-boundary intersection: finds the first wall the agent would hit along its displacement vector and stops it there. This preserves each agent's unique boundary contact point (unlike old np.clip which collapses any out-of-bounds agent to the nearest corner regardless of direction, causing spurious co-location of multiple agents). Also sets hit_geofence flag and zeroes velocity in the hit dimension(s).'''
    def clipPosToWSBoundary(self, isreset=False):
        if self.__class__.__name__ == "Predator" and self.landedOutsideWS():
            self.hit_geofence = True

        displacement = self.position - self.old_position

        # t=1 means full step with no wall hit; t<1 means wall is hit before end of step
        t_hit = 1.0
        hit_dims = np.zeros(self.num_dims, dtype=bool)

        for d in range(self.num_dims):
            lo = 0.0 if d == self.num_dims - 1 else -self._workspace_size
            hi = self._workspace_size
            if abs(displacement[d]) < 1e-12:
                continue
            t = (hi - self.old_position[d]) / displacement[d] if displacement[d] > 0 \
                else (lo - self.old_position[d]) / displacement[d]
            t = max(t, 0.0)  # clamp: if already outside, snap immediately
            if t < t_hit - 1e-12:       # strictly earlier wall hit
                t_hit = t
                hit_dims[:] = False
                hit_dims[d] = True
            elif t < t_hit + 1e-12:     # same t: hit a corner, both dims
                hit_dims[d] = True

        if t_hit < 1.0:
            self.position = self.old_position + t_hit * displacement
            # Zero velocity in hit dimension(s): wall absorbs outward momentum
            self.velocity = self.velocity.copy()
            self.velocity[hit_dims] = 0.0
    
    def landedOutsideWS(self):
        #checks if any of the elements of the position falls outside the workspace boundary. WS is always -ws_size .. +ws_size except in last dim where its 0..ws_size, which is why last pos needs special handling
        last_element = self.position[self.num_dims - 1]
        out_of_bounds =  any( i < -self._workspace_size or i > self._workspace_size \
                             for i in self.position[0:(self.num_dims - 1)]) or \
                        last_element < 0 or \
                        last_element > self._workspace_size
        return out_of_bounds


    #TBD: fix for 3D
    def hitObs(self): 
        last_element = self.position[self.num_dims - 1]
        #checks if any of the elements of the position falls in the obstacle area
        if self.doing_obstacles: 
            hit_obstacle = \
              any(self.LObs <= i and i <= self.RObs for i in self.position[0:(self.num_dims - 1)]) and \
              self.BObs <= last_element and\
              last_element <= self.TObs
        else: 
            hit_obstacle = False
        return (self.doing_obstacles and hit_obstacle)


    #NOTE: ***assumes that hit_geofence flag has been set by someone, eg clipPosToWSBoundary()
    def hitGeoFence(self): 
        return self.doing_geofence and self.hit_geofence
        
    def at(self, pos, min_sep):
        # distance = np.linalg.norm(self.position - pos)
        # return distance < min_sep
        for alpha in np.linspace(0, 1, 1000):
            interpolated = (1 - alpha) * self.old_position + alpha * self.position
            if np.linalg.norm(interpolated - pos) < min_sep:
                return True
        return False

    def normalizedDistanceTo(self, pos, min_sep=0):
        distance = np.linalg.norm(self.position - pos)
        if distance < min_sep:  
            distance = 0
        # max_distance = np.sqrt(self._workspace_size**2 * self.num_dims)
        assert self.num_dims == 2, "only handling 2D for now"  # only 2D for now
        #its a rectangular workspace not square
        max_distance = np.sqrt((2 * self._workspace_size)**2 + self._workspace_size**2)  
        normalized_distance = distance / max_distance
        return normalized_distance
    
    def move(self, acceleration, step_size):
        if not self.is_live:
            # breakpoint()
            return
        #TODO used to cap the velocity to reppresent terminal velocity
        delta_v = acceleration * step_size
        self.old_position = self.position.copy()
        self.old_velocity = self.velocity.copy()

        self.position = self.position + (self.velocity * step_size) + (delta_v/2 * step_size)
        self.velocity = self.velocity + (acceleration * step_size)
