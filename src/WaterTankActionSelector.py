import numpy as np
import torch as th
from .ActionSelector import ActionSelector

XX_XXX = True   # True  → 4-counter (o,c,p,d) XX+XXX model
                # False → 2-counter (t,s)      XX-only model
if XX_XXX:
    from .shields.mr_models.model_water_tank_XX_XXX import *
else:
    from .shields.mr_models.model_water_tank import *
# from .shields.builder.utils import *

# Shield guard constants (from buildShield output, NUM_NEXTS=2)
# _OPEN_TANK_MAX  = 98.0   # openAction  guard: tank <= 98
# _CLOSE_TANK_MIN =  1.0   # closeAction guard: tank >= 1


# ---------------------------------------------------------------------------
# WaterTankActionSelector: for the binary (open/close) water-tank shield.
# Shield state (t,s counters  or  o,c,p,d counters  depending on XX_XXX) lives here, NOT in the environment.
# ---------------------------------------------------------------------------
class WaterTankActionSelector(ActionSelector):

    def __init__(self, config, policy, env):
        super().__init__(config, policy, env)
        if XX_XXX:
            self.o = 0   # XX  open  counter
            self.c = 0   # XX  close counter
            self.p = 0   # XXX open  counter
            self.d = 0   # XXX close counter
        else:
            self.t = 0   # consecutive-open  counter
            self.s = 0   # consecutive-close counter

    def getActionForEachAgent(self, single_obs):
        # Reset shield counters at episode boundaries.
        if self.env.n_episode_steps == 0:
            if XX_XXX:
                self.o = 0
                self.c = 0
                self.p = 0
                self.d = 0
            else:
                self.t = 0
                self.s = 0

        # Sample the policy's preferred action.  For Discrete(2) the policy
        # returns a (1,) tensor containing 0 or 1.
        actions_tensor, values, log_probs = self.policy(single_obs)
        action = int(actions_tensor.cpu().item())

        if self.config['env_config']['use_shield']:
            open_valve = bool(action == 1)
            # Both calls use the original counters; only commit once the final action is known.
            if XX_XXX:
                ok, new_o, new_p, new_c, new_d = OK(open_valve, self.env.tank_level, self.o, self.p, self.c, self.d)
            else:
                ok, new_t, new_s = OK(open_valve, self.env.tank_level, self.t, self.s)
            if not ok:
                if XX_XXX:
                    ok, new_o, new_p, new_c, new_d = OK(not open_valve, self.env.tank_level, self.o, self.p, self.c, self.d)
                else:
                    ok, new_t, new_s = OK(not open_valve, self.env.tank_level, self.t, self.s)
                if not ok:
                    # Both actions blocked — shield invariant violation, should never happen.
                    if XX_XXX:
                        raise RuntimeError(
                            f'WaterTankActionSelector: both actions blocked! '
                            f'o={self.o} c={self.c} p={self.p} d={self.d} tank={self.env.tank_level:.2f}')
                    else:
                        raise RuntimeError(
                            f'WaterTankActionSelector: both actions blocked! '
                            f't={self.t} s={self.s} tank={self.env.tank_level:.2f}')
                # The shield vetoes the preferred action — flip it.
                action = 1 - action
                # Re-evaluate value + log_prob for the (possibly changed) action
                # since evaluate_actions expects shape (n_envs=1, 1) for Discrete
                action_tensor_2d = th.tensor([[action]], dtype=th.long).to(self.policy.device)
                values, log_probs, _ = self.policy.evaluate_actions(single_obs, action_tensor_2d)
            if XX_XXX:
                self.o = new_o   # commit only after the final action is confirmed
                self.c = new_c
                self.p = new_p
                self.d = new_d
            else:
                self.t = new_t  # commit only after the final action is confirmed
                self.s = new_s

        # Advance shield counters based on the action that will actually be executed.
        # self.updateShieldCounters(bool(action == 1))

        action_np = np.array([action])
        return action_np, values, log_probs

"""    def isActionOK(self, open_valve):
        '''Return True iff the synthesised shield permits this action.
        Guards from buildShield output (NUM_NEXTS=2):
          openAction  precond: t != 2  AND  tank <= 98
          closeAction precond: s != 2  AND  tank >= 1
        '''
        if open_valve:
            return self.t != 2 and self.env.tank_level <= _OPEN_TANK_MAX
        else:
            return self.s != 2 and self.env.tank_level >= _CLOSE_TANK_MIN

        def updateShieldCounters(self, open_valve):
        '''Advance t and s per the synthesised transition rules.'''
        if open_valve:
            if   self.t == 0: self.t = 1   # first open
            elif self.t == 1: self.t = 0   # second open -> reset
            # t==2 cannot occur (isActionOK blocks open when t==2)
            if   self.s == 0: self.s = 0
            elif self.s == 1: self.s = 2   # open after one close -> s=2
            elif self.s == 2: self.s = 0   # forced open at s==2  -> reset
        else:
            if   self.t == 0: self.t = 0
            elif self.t == 1: self.t = 2   # close after one open -> t=2
            elif self.t == 2: self.t = 0   # forced close at t==2 -> reset
            if   self.s == 0: self.s = 1   # first close
            elif self.s == 1: self.s = 0   # second close -> reset
            # s==2 cannot occur (isActionOK blocks close when s==2) 
"""
