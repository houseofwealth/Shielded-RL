# Discretized grid-world MR model: (Fig. 7, Elsayed-Aly et al., AAMAS 2021,
# arXiv:2101.11196v2). Floor-cell layout reconstructed by eye from the published
# figure (not pixel-exact)
#
# Two agents move on the shared grid (5 actions each: stay/up/down/left/right) and
# must never occupy the same cell at the same time -- this mirrors their Section 4
# centralized-shielding example (a single shield monitors the joint action of both
# agents, so this is CTCE).
#
# NOTE: only static collision is modeled (their d=0 case).
# Collision during a mutual swap (their d=+-1 case) is not yet encoded -- add later if needed.
#
# ABSOLUTE-COORDINATE MODEL: validCell is back inside the Z3 synthesis (the
# relative-coordinate, runtime-only-wallOk version is saved for reference in
# model_ma_grid_no_collision_test_tacked_on_after_synth.py). Feeding validCell
# into Z3 requires 'stay' to remain an available move -- see that file's header
# comment for why the runtime-only shortcut doesn't generalize if it's removed.
#
# validCell uses a compact strip/teeth/stride encoding (4 solid strips + 3
# castle-top stride-2 teeth regions + 2 irregular bottom regions + 1 leftover
# point) instead of a 43-term point disjunction, to keep this tractable for Z3.
# Only ISR_MAP has this compact form currently -- PENTAGON_MAP is unsupported here.


from z3    import *
from types import SimpleNamespace

''' -------- ISR MAP FLOOR CELLS (x=col, y=row), y=0 is bottom row ---------'''
ISR_MAP = \
[
           (1,9),        (3,9),        (5,9),        (7,9),
    (0,8), (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8), (8,8),
           (1,7),                                    (7,7),
    (0,6), (1,6),                                    (7,6), (8,6), 
           (1,5),                                    (7,5),
    (0,4), (1,4),                                    (7,4), (8,4),
           (1,3),        (3,3),                      (7,3),
    (0,2), (1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2),
                  (2,1), (3,1),        (5,1),        (7,1),
                  (2,0), (3,0),
]

# Figure 7 markers:
ISR_AGENT1_START = (1,3); ISR_AGENT1_TARGET = (3,0)
ISR_AGENT2_START = (0,2); ISR_AGENT2_TARGET = (3,3)

MAP = ISR_MAP
MAP_S = set(MAP)  # O(1) lookup, kept for callers that want a membership test outside Z3
# generic aliases so callers (e.g. the env) don't need to know which map is active
AGENT1_START = ISR_AGENT1_START; AGENT1_TARGET = ISR_AGENT1_TARGET
AGENT2_START = ISR_AGENT2_START; AGENT2_TARGET = ISR_AGENT2_TARGET

MOVES = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]   # stay, up, down, left, right

'''-------------------------- Z3 VARS -------------------------------------'''
x1, y1, x1X, y1X = Ints('x1 y1 x1X y1X')
x2, y2, x2X, y2X = Ints('x2 y2 x2X y2X')
dx1, dy1, dx2, dy2 = Ints('dx1 dy1 dx2 dy2')    #the moves of an agent

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
    '''Compact ISR floor-cell encoding -- see header comment. Checked cell-by-cell
    against ISR_MAP: covers exactly its 43 cells, no extras.'''
    return Or(
        And(y == 8, 0 <= x, x <= 8),                      # top strip
        And(y == 2, 0 <= x, x <= 8),                       # bottom strip
        And(x == 1, 3 <= y, y <= 7),                       # left strip
        And(x == 7, 1 <= y, y <= 9),                       # right strip
        And(y == 9, x % 2 == 1, 1 <= x, x <= 7),           # top teeth (stride 2)
        And(x == 0, y % 2 == 0, 4 <= y, y <= 6),           # left teeth (stride 2)
        And(x == 8, y % 2 == 0, 4 <= y, y <= 6),           # right teeth (stride 2)
        And(y == 1, Or(x == 2, x == 3, x == 5, x == 7)),   # bottom row (irregular)
        And(y == 0, 2 <= x, x <= 3),                       # corridor stub
        And(x == 3, y == 3),                               # leftover point
    )

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
    name        = 'multi-agent grid (absolute coords, compact validCell)',
    initNode    = singleton,
    nodes       = [singleton],
    transitions = transitions,
    )

initProps = []

# Safety properties: both agents stay on valid floor cells, and never share a cell.
safetyProps = [validCell(x1, y1), validCell(x2, y2), Not(And(x1 == x2, y1 == y2))]

'''Final Model - fixpoint at iteration 2
Node: main
  Vars: [x1, y1, x2, y2]
  Node Invariant: And(Or(And(y1 == 8, x1 >= 0, x1 <= 8),
       And(y1 == 2, x1 >= 0, x1 <= 8),
       And(x1 == 1, y1 >= 3, y1 <= 7),
       And(x1 == 7, y1 >= 1, y1 <= 9),
       And(y1 == 9, x1%2 == 1, x1 >= 1, x1 <= 7),
       And(x1 == 0, y1%2 == 0, y1 >= 4, y1 <= 6),
       And(x1 == 8, y1%2 == 0, y1 >= 4, y1 <= 6),
       And(y1 == 0, x1 >= 2, x1 <= 3),
       And(y1 == 3, x1 == 3),
       And(y1 == 1, Or(x1 == 7, x1 == 2, x1 == 3, x1 == 5))),
    Or(And(y2 == 8, x2 >= 0, x2 <= 8),
       And(y2 == 2, x2 >= 0, x2 <= 8),
       And(x2 == 1, y2 >= 3, y2 <= 7),
       And(x2 == 7, y2 >= 1, y2 <= 9),
       And(y2 == 9, x2%2 == 1, x2 >= 1, x2 <= 7),
       And(x2 == 0, y2%2 == 0, y2 >= 4, y2 <= 6),
       And(x2 == 8, y2%2 == 0, y2 >= 4, y2 <= 6),
       And(y2 == 0, x2 >= 2, x2 <= 3),
       And(x2 == 3, y2 == 3),
       And(y2 == 1, Or(x2 == 7, x2 == 2, x2 == 3, x2 == 5))),
    Not(And(x1 == x2, y1 == y2)))

The fixpoint converges at iteration 2 to exactly safetyProps (restated in the compact
strip/teeth/stride form) -- the invariant didn't need to strengthen, since 'stay' is
always a safe move. Synthesis time: ~2m0s (vs a much larger disjunction previously).
'''

# Python version of the synthesized invariant (same as safetyProps -- the fixpoint
# didn't need to strengthen it). Uses plain set membership against MAP_S rather than
# re-deriving the compact strip/teeth/stride form -- that form only exists to help Z3
# converge quickly during synthesis; at runtime a plain lookup is simpler and faster.
def solnExists(x1, y1, x2, y2):
    return (x1, y1) in MAP_S and (x2, y2) in MAP_S and (x1, y1) != (x2, y2)

def OK(dx1, dy1, dx2, dy2, x1, y1, x2, y2):
    return solnExists(x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2)