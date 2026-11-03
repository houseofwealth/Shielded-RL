# Discretized grid-world MR model: (Fig. 7, Elsayed-Aly et al., AAMAS 2021,
# arXiv:2101.11196v2). Floor-cell layout reconstructed by eye from the published
# figure (not pixel-exact)
#
# N agents (NUM_AGENTS below) move on the shared grid (5 actions each:
# stay/up/down/left/right) and must never occupy the same cell at the same time --
# generalizes their Section 4 centralized-shielding example (a single shield
# monitors the joint action of all agents, so this is CTCE) from 2 to N agents.
#
# NOTE: only static collision is modeled (their d=0 case).
# Collision during a mutual swap (their d=+-1 case) is not yet encoded -- add later if needed.
#
# ABSOLUTE-COORDINATE MODEL: validCell is fed into the Z3 synthesis (the
# relative-coordinate, runtime-only-wallOk version is saved for reference in
# model_ma_grid_no_collision_test_tacked_on_after_synth.py). Feeding validCell
# into Z3 requires 'stay' to remain an available move -- see that file's header
# comment for why the runtime-only shortcut doesn't generalize if it's removed.
#
# validCell uses a compact strip/teeth/stride encoding (4 solid strips + 3
# castle-top stride-2 teeth regions + 2 irregular bottom regions + 1 leftover
# point) instead of a 43-term point disjunction, to keep this tractable for Z3.
# Only ISR_MAP has this compact form currently -- PENTAGON_MAP is unsupported here.
#
# N-AGENT GENERALIZATION: NUM_AGENTS is fixed at import time -- change it (and add
# one more ISR_STARTS/ISR_TARGETS entry per extra agent) then re-run synthesis to
# retarget. The confirmed-working 2-agent-only version (with its fixpoint dump) is
# saved for reference in 'model_ma_grid - Copy.py'. safetyProps below is definitely
# a CORRECT joint constraint for any N, but whether synthesis is STABLE (fixpoint
# doesn't need to strengthen, as it didn't for N=2) is unverified for N>2 -- more
# agents sharing this small map means more opportunities for one agent to need a
# cell another is passing through, which previously only mattered within one pair.


from z3    import *
from types import SimpleNamespace

NUM_AGENTS = 3  # <- change this, add ISR_STARTS/ISR_TARGETS entries, then re-run synthesis

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

# Figure 7 markers, one (x,y) pair per agent, same order -- only 2 are from the
# paper; add a 3rd (and beyond) start/target before running with NUM_AGENTS>2.
ISR_STARTS  = [(1,3), (0,2)]
ISR_TARGETS = [(3,0), (3,3)]

MAP = ISR_MAP
MAP_S = set(MAP)  # O(1) lookup, kept for callers that want a membership test outside Z3
# generic aliases so callers (e.g. the env) don't need to know which map is active
STARTS  = ISR_STARTS
TARGETS = ISR_TARGETS

MOVES = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]   # stay, up, down, left, right

'''-------------------------- Z3 VARS -------------------------------------'''
xs  = [Int(f'x{i+1}')  for i in range(NUM_AGENTS)]
ys  = [Int(f'y{i+1}')  for i in range(NUM_AGENTS)]
xsX = [Int(f'x{i+1}X') for i in range(NUM_AGENTS)]
ysX = [Int(f'y{i+1}X') for i in range(NUM_AGENTS)]
dxs = [Int(f'dx{i+1}') for i in range(NUM_AGENTS)]
dys = [Int(f'dy{i+1}') for i in range(NUM_AGENTS)]  # the moves of an agent

state     = [v for i in range(NUM_AGENTS) for v in (xs[i], ys[i])]    # [x1,y1,x2,y2,...]
stateX    = [v for i in range(NUM_AGENTS) for v in (xsX[i], ysX[i])]
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

# Single joint action: all N agents move simultaneously (centralized shield picks
# the joint (move1, ..., moveN) tuple, same structure as their centralized shielding).
action = SimpleNamespace(
          name        = f'{NUM_AGENTS} agent move',
          actionPred  = And([And(xsX[i] == xs[i] + dxs[i], ysX[i] == ys[i] + dys[i])
                              for i in range(NUM_AGENTS)]),
          envVars     = [],
          envPred     = True,
          controlVars = [v for i in range(NUM_AGENTS) for v in (dxs[i], dys[i])],
          controlPred = And([moveOk(dxs[i], dys[i]) for i in range(NUM_AGENTS)]),
          precNode    = singleton,
          postNode    = singleton,
          )
transitions = [action]

model = SimpleNamespace(
    name        = f'multi-agent grid ({NUM_AGENTS} agents, absolute coords, compact validCell)',
    initNode    = singleton,
    nodes       = [singleton],
    transitions = transitions,
    )

initProps = []

# Safety properties: every agent stays on a valid floor cell, and no two agents
# ever share a cell (one Not(...) per unordered pair -- C(N,2) terms).
safetyProps = (
    [validCell(xs[i], ys[i]) for i in range(NUM_AGENTS)] +
    [Not(And(xs[i] == xs[j], ys[i] == ys[j]))
     for i in range(NUM_AGENTS) for j in range(i + 1, NUM_AGENTS)]
)

# Re-run synthesis for this NUM_AGENTS and paste the resulting fixpoint dump here
# (was confirmed at fixpoint iteration 2, ~2m0s for N=2 -- see 'model_ma_grid - Copy.py').

# Python version of the synthesized invariant -- ASSUMES the fixpoint still converges
# to exactly safetyProps (true for N=2; unverified for N>2, see header comment).
# Uses plain set membership against MAP_S rather than re-deriving the compact
# strip/teeth/stride form -- that form only exists to help Z3 converge quickly
# during synthesis; at runtime a plain lookup is simpler and faster.
def solnExists(positions):
    '''positions: list of NUM_AGENTS (x,y) tuples, one per agent.'''
    return (len(positions) == NUM_AGENTS
            and all(p in MAP_S for p in positions)
            and len(set(positions)) == NUM_AGENTS)

def OK(moves, positions):
    '''moves, positions: lists of NUM_AGENTS (dx,dy)/(x,y) tuples, one per agent
    (same order). Replaces the old flat OK(dx1,dy1,dx2,dy2,x1,y1,x2,y2) signature.'''
    new_positions = [(x + dx, y + dy) for (dx, dy), (x, y) in zip(moves, positions)]
    return solnExists(new_positions)