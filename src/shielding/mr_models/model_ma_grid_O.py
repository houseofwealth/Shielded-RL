# NOTE: you can run buildShield and this converges after several > 5 mins. It struggles on account of the huge disjunction you have for the next possible state. Its might be possible to optimize this a bit with a function/array that answers yes/no for a given pair, ie a characteristic function. But that's a bit of a hack and i decided noot to pursue it b/c i realize now wy Elsayed-Aly et al. chose this problem even though it has no interesting winning region to compute, rather they wanted to illustrate their factored shielding approach. Its not how i want to proceed right now so i am not pursing this example for now


# Discretized grid-world MR model: (Fig. 7, Elsayed-Aly et al., AAMAS 2021,
# arXiv:2101.11196v2). Floor-cell layout reconstructed by eye from the published
# figure (not pixel-exact) 
#
# Two agents move on the shared grid (5 actions each: stay/up/down/left/right) and
# must never occupy the same cell at the same time -- this mirrors their Section 4
# "centralized shielding" example (a single shield monitors the joint action of both
# agents, so this is CTCE)
#
# NOTE: only static collision is modeled (their d=0 case). 
# collision during a mutual swap (their d=+-1 case) is not yet encoded -- add later if needed.
#
# To run: swap the import in buildShield.py to
#   from ..mr_models.model_isr_grid import *
# then run buildShield.py (e.g. `time python3 buildShield.py`) to get synthesis time.

from z3    import *
from types import SimpleNamespace

''' ---------------- ISR MAP FLOOR CELLS (x=col, y=row), y=0 is bottom row --------------'''
ISR_MAP = \
[
           (1,9),        (3,9),        (5,9),        (7,9),
    (0,8), (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8), (8,8),
           (1,7),                                    (7,7),
    (0,6), (1,6),                                           (8,6), 
           (1,5),                                    (7,5),
    (0,4), (1,4),                                           (8,4),
           (1,3),        (3,3),                      (7,3),
    (0,2), (1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2),
                  (2,1), (3,1),        (5,1),        (7,1),
                  (2,0), (3,0),
]

# Figure 7 markers:
ISR_AGENT1_START = (1,3); ISR_AGENT1_TARGET = (3,0)
ISR_AGENT2_START = (0,2); ISR_AGENT2_TARGET = (3,3)

'''--- other maps from their examples go here---'''
'''---'''

MAP = ISR_MAP
# generic aliases so callers (e.g. the env) don't need to know which map is active
AGENT1_START = ISR_AGENT1_START; AGENT1_TARGET = ISR_AGENT1_TARGET
AGENT2_START = ISR_AGENT2_START; AGENT2_TARGET = ISR_AGENT2_TARGET
MOVES = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]   # stay, up, down, left, right

'''-------------------------- Z3 VARS -------------------------------------'''
x1, y1, x1X, y1X = Ints('x1 y1 x1X y1X')
x2, y2, x2X, y2X = Ints('x2 y2 x2X y2X')
dx1, dy1, dx2, dy2 = Ints('dx1 dy1 dx2 dy2')

state     = [x1, y1, x2, y2]
stateX    = [x1X, y1X, x2X, y2X]
subst     = list(zip(state, stateX))
inv_subst = list(zip(stateX, state))

global_props = []  # globals() is used by Python

singleton = SimpleNamespace(
             name          = 'main',
             vars          = state,
             postVars      = stateX,
             subst         = subst,
             invariant     = True,
             stateInvDelta = True,
             )

def validCell(x, y):
    return Or([And(x == cx, y == cy) for (cx, cy) in MAP])

def moveOk(dx, dy):
    return Or([And(dx == mx, dy == my) for (mx, my) in MOVES])

# Single joint action: both agents move simultaneously (centralized shield picks
# the joint (move1, move2) pair, same structure as their centralized shielding).
action = SimpleNamespace(
          name        = '2 agent move',
          actionPred  = And(x1X == x1 + dx1, y1X == y1 + dy1,
                             x2X == x2 + dx2, y2X == y2 + dy2),
          envVars     = [],
          envPred     = True,
          controlVars = [dx1, dy1, dx2, dy2],
          controlPred = And(moveOk(dx1, dy1), moveOk(dx2, dy2)),
          precNode    = singleton,
          postNode    = singleton,
          )
transitions = [action]

model = SimpleNamespace(
    name        = 'multi-agent grid',
    initNode    = singleton,
    nodes       = [singleton],
    transitions = transitions,
    )

initProps = []

# Safety properties: both agents stay on valid floor cells, and never share a cell.
safetyProps = [validCell(x1, y1), validCell(x2, y2), Not(And(x1 == x2, y1 == y2))]

'''
Final Model - fixpoint at iteration 2
Node: main
  Vars: [x1, y1, x2, y2]
  Node Invariant: And(Or(And(y1 == 1, x1 == 5),
       And(y1 == 4, x1 == 0),
       And(y1 == 3, x1 == 7),
       And(y1 == 2, x1 == 1),
       And(y1 == 5, x1 == 1),
       And(y1 == 0, x1 == 3),
       And(y1 == 3, x1 == 3),
       And(y1 == 2, x1 == 6),
       And(y1 == 8, x1 == 3),
       And(y1 == 1, x1 == 3),
       And(y1 == 0, x1 == 2),
       And(y1 == 2, x1 == 2),
       And(y1 == 2, x1 == 7),
       And(y1 == 9, x1 == 1),
       And(y1 == 4, x1 == 8),
       And(y1 == 6, x1 == 0),
       And(y1 == 8, x1 == 0),
       And(y1 == 2, x1 == 0),
       And(y1 == 7, x1 == 7),
       And(y1 == 2, x1 == 4),
       And(y1 == 9, x1 == 5),
       And(y1 == 6, x1 == 8),
       And(y1 == 8, x1 == 8),
       And(y1 == 8, x1 == 5),
       And(y1 == 6, x1 == 1),
       And(y1 == 1, x1 == 2),
       And(y1 == 3, x1 == 1),
       And(y1 == 5, x1 == 7),
       And(y1 == 8, x1 == 2),
       And(y1 == 8, x1 == 1),
       And(y1 == 8, x1 == 7),
       And(y1 == 9, x1 == 7),
       And(y1 == 2, x1 == 5),
       And(y1 == 8, x1 == 6),
       And(y1 == 8, x1 == 4),
       And(y1 == 2, x1 == 8),
       And(y1 == 4, x1 == 1),
       And(y1 == 2, x1 == 3),
       And(y1 == 1, x1 == 7),
       And(y1 == 9, x1 == 3),
       And(y1 == 7, x1 == 1)),
    Or(And(x2 == 1, y2 == 9),
       And(x2 == 3, y2 == 9),
       And(x2 == 5, y2 == 9),
       And(x2 == 7, y2 == 9),
       And(x2 == 0, y2 == 8),
       And(x2 == 1, y2 == 8),
       And(x2 == 2, y2 == 8),
       And(x2 == 3, y2 == 8),
       And(x2 == 4, y2 == 8),
       And(x2 == 5, y2 == 8),
       And(x2 == 6, y2 == 8),
       And(x2 == 7, y2 == 8),
       And(x2 == 8, y2 == 8),
       And(x2 == 1, y2 == 7),
       And(x2 == 7, y2 == 7),
       And(x2 == 0, y2 == 6),
       And(x2 == 1, y2 == 6),
       And(x2 == 8, y2 == 6),
       And(x2 == 1, y2 == 5),
       And(x2 == 7, y2 == 5),
       And(x2 == 0, y2 == 4),
       And(x2 == 1, y2 == 4),
       And(x2 == 8, y2 == 4),
       And(x2 == 1, y2 == 3),
       And(x2 == 3, y2 == 3),
       And(x2 == 7, y2 == 3),
       And(x2 == 0, y2 == 2),
       And(x2 == 1, y2 == 2),
       And(x2 == 2, y2 == 2),
       And(x2 == 3, y2 == 2),
       And(x2 == 4, y2 == 2),
       And(x2 == 5, y2 == 2),
       And(x2 == 6, y2 == 2),
       And(x2 == 7, y2 == 2),
       And(x2 == 8, y2 == 2),
       And(x2 == 2, y2 == 1),
       And(x2 == 3, y2 == 1),
       And(x2 == 5, y2 == 1),
       And(x2 == 7, y2 == 1),
       And(x2 == 2, y2 == 0),
       And(x2 == 3, y2 == 0)),
    Not(And(x1 == x2, y1 == y2)))
Transition: 2 agent move
   0   Or(And(dy1 == 1, dx1 == 0),
   And(dy1 == 0, dx1 == 1),
   And(dy1 == -1, dx1 == 0),
   And(dy1 == 0, dx1 == -1),
   And(dy1 == 0, dx1 == 0))
   1   Or(And(dy2 == -1, dx2 == 0),
   And(dy2 == 0, dx2 == -1),
   And(dy2 == 1, dx2 == 0),
   And(dy2 == 0, dx2 == 0),
   And(dy2 == 0, dx2 == 1))
   2   Or(And(y1 + dy1 == 1, x1 + dx1 == 2),
   And(y1 + dy1 == 2, x1 + dx1 == 1),
   And(y1 + dy1 == 8, x1 + dx1 == 1),
   And(y1 + dy1 == 0, x1 + dx1 == 3),
   And(y1 + dy1 == 8, x1 + dx1 == 6),
   And(y1 + dy1 == 3, x1 + dx1 == 3),
   And(y1 + dy1 == 8, x1 + dx1 == 7),
   And(y1 + dy1 == 0, x1 + dx1 == 2),
   And(y1 + dy1 == 8, x1 + dx1 == 0),
   And(y1 + dy1 == 5, x1 + dx1 == 7),
   And(y1 + dy1 == 2, x1 + dx1 == 6),
   And(y1 + dy1 == 8, x1 + dx1 == 5),
   And(y1 + dy1 == 7, x1 + dx1 == 1),
   And(y1 + dy1 == 2, x1 + dx1 == 4),
   And(y1 + dy1 == 9, x1 + dx1 == 1),
   And(y1 + dy1 == 4, x1 + dx1 == 0),
   And(y1 + dy1 == 6, x1 + dx1 == 8),
   And(y1 + dy1 == 9, x1 + dx1 == 3),
   And(y1 + dy1 == 9, x1 + dx1 == 5),
   And(y1 + dy1 == 2, x1 + dx1 == 7),
   And(y1 + dy1 == 9, x1 + dx1 == 7),
   And(y1 + dy1 == 1, x1 + dx1 == 5),
   And(y1 + dy1 == 8, x1 + dx1 == 4),
   And(y1 + dy1 == 8, x1 + dx1 == 3),
   And(y1 + dy1 == 1, x1 + dx1 == 7),
   And(y1 + dy1 == 2, x1 + dx1 == 3),
   And(y1 + dy1 == 1, x1 + dx1 == 3),
   And(y1 + dy1 == 2, x1 + dx1 == 5),
   And(y1 + dy1 == 7, x1 + dx1 == 7),
   And(y1 + dy1 == 6, x1 + dx1 == 0),
   And(y1 + dy1 == 3, x1 + dx1 == 7),
   And(y1 + dy1 == 2, x1 + dx1 == 8),
   And(y1 + dy1 == 2, x1 + dx1 == 2),
   And(y1 + dy1 == 2, x1 + dx1 == 0),
   And(y1 + dy1 == 6, x1 + dx1 == 1),
   And(y1 + dy1 == 8, x1 + dx1 == 8),
   And(y1 + dy1 == 4, x1 + dx1 == 1),
   And(y1 + dy1 == 3, x1 + dx1 == 1),
   And(y1 + dy1 == 8, x1 + dx1 == 2),
   And(y1 + dy1 == 5, x1 + dx1 == 1),
   And(y1 + dy1 == 4, x1 + dx1 == 8))
   3   Or(And(x2 + dx2 == 7, y2 + dy2 == 9),
   And(x2 + dx2 == 6, y2 + dy2 == 8),
   And(x2 + dx2 == 8, y2 + dy2 == 4),
   And(x2 + dx2 == 8, y2 + dy2 == 8),
   And(x2 + dx2 == 7, y2 + dy2 == 7),
   And(x2 + dx2 == 1, y2 + dy2 == 2),
   And(x2 + dx2 == 3, y2 + dy2 == 9),
   And(x2 + dx2 == 2, y2 + dy2 == 8),
   And(x2 + dx2 == 1, y2 + dy2 == 7),
   And(x2 + dx2 == 3, y2 + dy2 == 3),
   And(x2 + dx2 == 0, y2 + dy2 == 2),
   And(x2 + dx2 == 7, y2 + dy2 == 5),
   And(x2 + dx2 == 7, y2 + dy2 == 3),
   And(x2 + dx2 == 3, y2 + dy2 == 2),
   And(x2 + dx2 == 0, y2 + dy2 == 6),
   And(x2 + dx2 == 1, y2 + dy2 == 3),
   And(x2 + dx2 == 0, y2 + dy2 == 4),
   And(x2 + dx2 == 3, y2 + dy2 == 0),
   And(x2 + dx2 == 1, y2 + dy2 == 4),
   And(x2 + dx2 == 7, y2 + dy2 == 8),
   And(x2 + dx2 == 6, y2 + dy2 == 2),
   And(x2 + dx2 == 3, y2 + dy2 == 1),
   And(x2 + dx2 == 7, y2 + dy2 == 2),
   And(x2 + dx2 == 8, y2 + dy2 == 2),
   And(x2 + dx2 == 5, y2 + dy2 == 8),
   And(x2 + dx2 == 7, y2 + dy2 == 1),
   And(x2 + dx2 == 4, y2 + dy2 == 8),
   And(x2 + dx2 == 5, y2 + dy2 == 2),
   And(x2 + dx2 == 5, y2 + dy2 == 1),
   And(x2 + dx2 == 4, y2 + dy2 == 2),
   And(x2 + dx2 == 0, y2 + dy2 == 8),
   And(x2 + dx2 == 8, y2 + dy2 == 6),
   And(x2 + dx2 == 2, y2 + dy2 == 1),
   And(x2 + dx2 == 1, y2 + dy2 == 5),
   And(x2 + dx2 == 3, y2 + dy2 == 8),
   And(x2 + dx2 == 2, y2 + dy2 == 2),
   And(x2 + dx2 == 1, y2 + dy2 == 6),
   And(x2 + dx2 == 1, y2 + dy2 == 8),
   And(x2 + dx2 == 1, y2 + dy2 == 9),
   And(x2 + dx2 == 5, y2 + dy2 == 9),
   And(x2 + dx2 == 2, y2 + dy2 == 0))
   4   Not(And(x1 + dx1 == x2 + dx2, y1 + dy1 == y2 + dy2))
   5   Or(And(dx1 == 0, dy1 == 0),
   And(dx1 == 0, dy1 == -1),
   And(dx1 == 0, dy1 == 1),
   And(dx1 == -1, dy1 == 0),
   And(dx1 == 1, dy1 == 0))
   6   Or(And(dx2 == 0, dy2 == 0),
   And(dx2 == 0, dy2 == -1),
   And(dx2 == 0, dy2 == 1),
   And(dx2 == -1, dy2 == 0),
   And(dx2 == 1, dy2 == 0))
   7   x1X == x1 + dx1
   8   y1X == y1 + dy1
   9   x2X == x2 + dx2
   10   y2X == y2 + dy2
'''

# Python version of the synthesized invariant (same as safetyProps -- the
# fixpoint didn't need to strengthen it, since "stay" is always a safe move).
def solnExists(x1, y1, x2, y2):
    return (x1, y1) in MAP and (x2, y2) in MAP and (x1, y1) != (x2, y2)

def OK(dx1, dy1, dx2, dy2, x1, y1, x2, y2):
    return solnExists(x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2)