from gymnasium import spaces, Env
import numpy as np

'''Water Tank environment, modelled on the water tank example from
"Safe Reinforcement Learning via Shielding" (Alshiekh et al., 2018).
See src/shields/mr_models/model_water_tank.py for the shield model this pairs with.

State:  tank_level  (float in [0, TANK_CAPACITY])

Action: Discrete(2)
  0 = close valve  (openValve = False)
  1 = open valve   (openValve = True)

Transition (non-deterministic -- the environment is the "other player"):
  Each step the environment samples:
    inn ~ Uniform(INN_MIN, INN_MAX)   # inflow  (adversarial env variable)
    out ~ Uniform(OUT_MIN, OUT_MAX)   # outflow (adversarial env variable)
  Then:
    open:  tank_level += inn - out
    close: tank_level -= out

  Matches model_water_tank.py:
    openAction:  tankX == tank + inn - out,  1 <= inn <= 2,  0 <= out <= 1
    closeAction: tankX == tank - out,        1 <= inn <= 2,  0 <= out <= 1

Episode ends when tank_level leaves [0, TANK_CAPACITY], or max steps reached.
Reward: -1 on unsafe termination, 0 each surviving step.
'''


class WaterTankEnv(Env):

    def __init__(self, config):
        assert isinstance(config, dict), 'Config should be a dict!'
        env_config = config['env_config']
        assert isinstance(env_config, dict), 'env_config in config should be a dict!'
        for key, value in env_config.items():
            self.__dict__[key] = value

        self.__mkSpaces()

        # lifetime counters
        self.n_unsafe    = 0
        self.n_truncated = 0

    # ------------------------------------------------------------------
    # Public interface  (called by SB3 training loop)
    # ------------------------------------------------------------------

    def reset(self, options=None, seed=0):
        self.__resetEpisodeCounters()
        self.tank_level = float(self.INITIAL_TANK_LEVEL)
        return self.__getObservation(), {}

    def step(self, action):
        self.n_episode_steps += 1
        open_valve = bool(int(action) == 1)

        # environment (adversary) picks inflow and outflow
        inn = np.random.uniform(self.INN_MIN, self.INN_MAX)
        out = np.random.uniform(self.OUT_MIN, self.OUT_MAX)

        if open_valve:
            self.tank_level += inn - out
        else:
            self.tank_level -= out

        unsafe    = self.tank_level < 0 or self.tank_level > self.TANK_CAPACITY
        truncated = self.n_episode_steps >= self.MAX_EPISODE_STEPS
        done      = unsafe or truncated

        if unsafe:
            self.n_unsafe += 1
            if self.n_unsafe % 100 == 0:
                print(f'n_unsafe: {self.n_unsafe}')
        if truncated:
            self.n_truncated += 1

        reward = self.__getReward(unsafe)
        obs    = self.__getObservation()
        return obs, reward, done, truncated, {'unsafe': unsafe}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def __mkSpaces(self):
        self.action_space = spaces.Discrete(2)  # 0=close, 1=open

        self.observation_space = spaces.Dict({
            'tank_level': spaces.Box(
                low=0.0, high=float(self.TANK_CAPACITY),
                shape=(1,), dtype=np.float32),
        })

    def __resetEpisodeCounters(self):
        self.n_episode_steps = 0

    def __getObservation(self):
        return {
            'tank_level': np.array([self.tank_level], dtype=np.float32),
        }

    def __getReward(self, unsafe):
        if unsafe:
            return -1.0
        return 0.0
