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
# Collision during a mutual swap (their d=+-1 case) is not yet encoded -- if added, it may
# reintroduce coupling with corridor width / validCell (see caveat below) -- add later if needed.
#
# RELATIVE-COORDINATE REDUCTION: state is (diffx, diffy) = (x1-x2, y1-y2) instead of
# (x1,y1,x2,y2) -- 2-D instead of 4-D, same trick as model_gd_dist.py. validCell is left
# out of the Z3 model becaause it doesnt require lookaead past the proposed action to check so its just checked directly at runtime via wallOk. Old 4-D version took >5 min to
# synthesize; this one doesn't need validCell in the loop at all.
#


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

''' -------- PENTAGON MAP FLOOR CELLS (x=col, y=row), y=0 is bottom row ---------'''
PENTAGON_MAP = \
[   #11 x 9 (x=0..10, y=0..8)
           (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8), (8,8), (9,8),
           (1,7),                      (5,7),                      (9,7),
           (1,6),                      (5,6),                      (9,6),
           (1,5),               (4,5), (5,5), (6,5), (7,5),        (9,5),
           (1,4), (2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (9,4), (10,4),
           (1,3),               (4,3), (5,3), (6,3),               (9,3),
           (1,2),                      (5,2),                      (9,2),
           (1,1),                      (5,1), (6,1),               (9,1),
    (0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0),
]
# Figure 7 markers:
PENTAGON_AGENT1_START = (6,1); PENTAGON_AGENT1_TARGET = (5,8)  #blue
PENTAGON_AGENT2_START = (2,4); PENTAGON_AGENT2_TARGET = (10,4) #orange

'''--- other maps from their examples go here---'''
'''---'''

# MAP = ISR_MAP
MAP = PENTAGON_MAP
MAP_S = set(MAP)  # O(1) lookup for the runtime wall filter (wallOk, below)
# generic aliases so callers (e.g. the env) don't need to know which map is active
AGENT1_START = PENTAGON_AGENT1_START; AGENT1_TARGET = PENTAGON_AGENT1_TARGET
AGENT2_START = PENTAGON_AGENT2_START; AGENT2_TARGET = PENTAGON_AGENT2_TARGET

MOVES = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]   # stay, up, down, left, right

# Reachable relative moves: Minkowski difference of MOVES with itself.
MOVE_DIFFS = sorted(set((mx1 - mx2, my1 - my2) for (mx1, my1) in MOVES for (mx2, my2) in MOVES))

'''-------------------------- Z3 VARS -------------------------------------'''
# Relative state: position of agent 1 w.r.t. agent 2
diffx, diffxX, diffy, diffyX = Ints('diffx diffxX diffy diffyX')
diffdx, diffdy = Ints('diffdx diffdy')

state     = [diffx, diffy]
stateX    = [diffxX, diffyX]
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

def moveDiffOk(ddx, ddy):
    return Or([And(ddx == dx, ddy == dy) for (dx, dy) in MOVE_DIFFS])

# Single action: both agents move simultaneously, represented as the relative
# move (diffdx, diffdy) = (dx1 - dx2, dy1 - dy2) -- same reduction as relative_accelerate
# in model_gd_dist.py.
action = SimpleNamespace(
          name        = 'relative move',
          actionPred  = And(diffxX == diffx + diffdx, diffyX == diffy + diffdy),
          envVars     = [],
          envPred     = True,
          controlVars = [diffdx, diffdy],
          controlPred = moveDiffOk(diffdx, diffdy),
          precNode    = singleton,
          postNode    = singleton,
          )
transitions = [action]

model = SimpleNamespace(
    name        = 'multi-agent grid (relative coords)',
    initNode    = singleton,
    nodes       = [singleton],
    transitions = transitions,
    )

initProps = []

# Safety property: agents never share a cell. Purely relative -- validCell is enforced
# separately at runtime (see wallOk), not fed into the shield.
safetyProps = [Not(And(diffx == 0, diffy == 0))]

'''
Relative-model fixpoint (2-D state, diffx/diffy only):
  Node Invariant: Not(And(diffx == 0, diffy == 0))
  Transition 'relative move':
    moveDiffOk(diffdx, diffdy)   -- Minkowski difference of MOVES with itself
    diffxX == diffx + diffdx
    diffyX == diffy + diffdy
  Converges unchanged (same argument as the old absolute model: 'stay' is always safe).
'''

# ---------------------------------------------------------------------------
# Runtime check functions (mirrors the fnOr/fnAnd/OKDist pattern in model_gd_dist.py).
# ---------------------------------------------------------------------------
def wallOk(x, y, dx, dy):
    '''One-step wall-legality filter -- NOT part of the shield; enforced directly
    because stay always preserves validity, so no fixpoint is needed.'''
    return (x + dx, y + dy) in MAP_S

def solnExists(diffx, diffy):
    '''Relative-state invariant: the synthesised shield reduces to the raw safety spec.'''
    return (diffx, diffy) != (0, 0)

def OKStep(diffdx, diffdy, diffx, diffy):
    return solnExists(diffx + diffdx, diffy + diffdy)

def OK(dx1, dy1, dx2, dy2, x1, y1, x2, y2):
    '''Full per-step safety check: wall-legality (direct filter) + collision
    avoidance (synthesised relative shield).'''
    return (wallOk(x1, y1, dx1, dy1) and wallOk(x2, y2, dx2, dy2)
            and OKStep(dx1 - dx2, dy1 - dy2, x1 - x2, y1 - y2))

'''
Final Model - fixpoint at iteration 2
Node: main
  Vars: [diffx, diffy]
  Node Invariant: Not(And(diffx == 0, diffy == 0))
Transition: relative move
   0   Or(And(diffdy == 0, diffdx == 1),
   And(diffdy == 1, diffdx == -1),
   And(diffdy == -1, diffdx == 0),
   And(diffdy == -1, diffdx == 1),
   And(diffdy == 1, diffdx == 0),
   And(diffdy == -1, diffdx == -1),
   And(diffdy == 0, diffdx == -2),
   And(diffdy == -2, diffdx == 0),
   And(diffdy == 1, diffdx == 1),
   And(diffdy == 2, diffdx == 0),
   And(diffdy == 0, diffdx == -1),
   And(diffdy == 0, diffdx == 0),
   And(diffdy == 0, diffdx == 2))
   1   Or(Not(0 == diffy + diffdy), Not(0 == diffx + diffdx))
   2   Or(And(diffdx == -2, diffdy == 0),
   And(diffdx == -1, diffdy == -1),
   And(diffdx == -1, diffdy == 0),
   And(diffdx == -1, diffdy == 1),
   And(diffdx == 0, diffdy == -2),
   And(diffdx == 0, diffdy == -1),
   And(diffdx == 0, diffdy == 0),
   And(diffdx == 0, diffdy == 1),
   And(diffdx == 0, diffdy == 2),
   And(diffdx == 1, diffdy == -1),
   And(diffdx == 1, diffdy == 0),
   And(diffdx == 1, diffdy == 1),
   And(diffdx == 2, diffdy == 0))
   3   diffxX == diffx + diffdx
   4   diffyX == diffy + diffdy
'''