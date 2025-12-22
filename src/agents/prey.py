import numpy as np
from .agent import BaseAgent

class Prey(BaseAgent):
    def __init__(self, num_dims, workspace_size):
        super().__init__(num_dims=num_dims, workspace_size=workspace_size)
