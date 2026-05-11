# Shield synthesis for drone separation constraint.
#
# State is the RELATIVE position/velocity of predator 1 w.r.t. predator 2:
#   (diffx, diffv_x, diffy, diffv_y)  where  diffx = x1 - x2,  diffv_x = vx1 - vx2,  etc.
#
# Control is the RELATIVE acceleration:
#   diffa_x = a_x1 - a_x2,  diffa_y = a_y1 - a_y2
#
# This reduction is valid because the double-integrator dynamics are linear:
#   diffx'   = diffx  + diffv_x + (1/2)*diffa_x   (= x1' - x2')
#   diffv_x' = diffv_x + diffa_x                   (= vx1' - vx2')
# and likewise for y.  The E ax1 E ax2 problem therefore collapses to E diffa_x E diffa_y,
# so synthesis is a standard single-agent problem in 4-D relative state.
#
# Safety properties (each independently optional, set to -1 to disable):
#   MAX_SEP: |diffx| <= MAX_SEP and |diffy| <= MAX_SEP  (predators stay close, L-inf)
#   MIN_SEP: predators stay at least MIN_SEP apart       (collision avoidance, L-inf disjunction)
#
# To run synthesis:  swap the import in buildShield.py to
#   from ..mr_models.model_dist import *
# then run buildShield.py.  The printed controlPred at fixpoint is the body of OKDist.

# To run synthesis: swap the import in buildShield.py to
#   from shields.mr_models.model_gd_dist import *
# then run buildShield.py.  The printed controlPred at fixpoint is the body of OKDist.

from z3       import *
from types    import SimpleNamespace
from config_gd import DEFAULT_CONFIG

''' ------------------ CONSTANTS ------------------------------'''
DELTA_T = DEFAULT_CONFIG['env_config']['STEP_SIZE']
A_MAX   = DEFAULT_CONFIG['env_config']['max_acceleration']
A_MIN   = -A_MAX
# Each predator's acceleration is bounded by A_MAX, so the relative
# acceleration diffa = a1 - a2 lies in [-2*A_MAX, 2*A_MAX].
DIFFA_MAX = 2 * A_MAX
DIFFA_MIN = -DIFFA_MAX

# Separation bounds.  Set to -1 to disable.
MAX_SEP = DEFAULT_CONFIG['env_config'].get('MAX_SEP', -1)   # predators must stay within this distance
MIN_SEP = DEFAULT_CONFIG['env_config'].get('MIN_SEP', -1)  # predators must stay at least this far apart
DOING_SEP = DEFAULT_CONFIG['env_config'].get('DOING_SEP', False)  # enable pred-pred separation shield
print('DOING_SEP =', DOING_SEP)
if DOING_SEP: print('MAX_SEP =', MAX_SEP, '  MIN_SEP =', MIN_SEP)
# print('relative acceleration bounds DIFFA_MAX =', DIFFA_MAX)

'''-------------------------- Z3 VARS -------------------------------------'''
# Relative state: position and velocity of pred1 relative to pred2
diffx,  diffxX,  diffv_x, diffv_xX, diffa_x = Reals('diffx diffxX diffv_x diffv_xX diffa_x')
diffy,  diffyX,  diffv_y, diffv_yX, diffa_y = Reals('diffy diffyX diffv_y diffv_yX diffa_y')

state     = [diffx,  diffy,  diffv_x,  diffv_y]
stateX    = [diffxX, diffyX, diffv_xX, diffv_yX]
subst     = list(zip(state, stateX))
inv_subst = list(zip(stateX, state))

global_props = [] # globals() is used by Python

singleton = SimpleNamespace(
             name         = 'main',
             vars         = state,
             postVars     = stateX,
             subst        = subst,
             invariant    = True,
             stateInvDelta = True,
             )

def delta_v(a):
    return a * DELTA_T

def delta_s(v, a):
    deltav = delta_v(a)
    return v * DELTA_T + (deltav / 2) * DELTA_T

# Single action: both predators accelerate (represented as relative accel diffa)
action = SimpleNamespace(
          name         = 'relative_accelerate',
          actionPred   = And(diffxX  == diffx  + delta_s(diffv_x, diffa_x),
                             diffv_xX == diffv_x + delta_v(diffa_x),
                             diffyX  == diffy  + delta_s(diffv_y, diffa_y),
                             diffv_yX == diffv_y + delta_v(diffa_y)),
          envVars      = [],
          envPred      = True,
          controlVars  = [diffa_x, diffa_y],
          controlPred  = And(DIFFA_MIN <= diffa_x, diffa_x <= DIFFA_MAX,
                             DIFFA_MIN <= diffa_y, diffa_y <= DIFFA_MAX),
          precNode     = singleton,
          postNode     = singleton
          )
transitions = [action]

model = SimpleNamespace(
         name         = 'predator-predator separation (relative coords), 2-D',
         initNode     = singleton,
         nodes        = [singleton],
         transitions  = transitions
         )

initProps = []

# Safety properties fed into buildSafetyShield().
# MAX_SEP: predators must not drift further than MAX_SEP from each other (L-inf box).
# MIN_SEP: predators must not get closer than MIN_SEP to each other (L-inf, disjunctive —
#          same structure as obstacle avoidance in model_gd.py).
max_sep_props = ([-MAX_SEP <= diffx, diffx <= MAX_SEP,
                  -MAX_SEP <= diffy, diffy <= MAX_SEP]
                 if MAX_SEP != -1 else [])

min_sep_props = ([Or(diffx >= MIN_SEP, diffx <= -MIN_SEP,
                     diffy >= MIN_SEP, diffy <= -MIN_SEP)]
                 if MIN_SEP != -1 else [])

safetyProps = max_sep_props + min_sep_props

# ---------------------------------------------------------------------------
# Runtime check functions (to be called from DronesActionSelector / OK())
# These mirror OKGeo / OKBR in model_gd.py.
# ---------------------------------------------------------------------------
def fnOr(xs):  return any(xs)
def fnAnd(xs): return all(xs)
def fnNot(x):  return not x
def implies(p, q):
    return not p or q

'''MIN_SEP = 1'''
def solnExistsDist_MIN_SEP_1(curr_st1, curr_st2):
    """State invariant for predator-predator separation.
    Returns True if the current relative state satisfies the synthesised invariant.
    Synthesised invariants (by case):
      MIN_SEP=1, MAX_SEP=-1: Or(diffx >= 1, diffx <= -1, diffy >= 1, diffy <= -1)
      MAX_SEP only:           TODO (pending synthesis run)
      both:                   TODO (pending synthesis run)
    """
    x1, y1, vx1, vy1 = curr_st1
    x2, y2, vx2, vy2 = curr_st2
    _diffx   = x1 - x2
    _diffy   = y1 - y2
    _diffv_x = vx1 - vx2
    _diffv_y = vy1 - vy2

    assert MIN_SEP == 1, f'solnExistsDistPy: invariant only synthesised for MIN_SEP=1, got {MIN_SEP}'
    ok = fnOr([
        _diffx  >= MIN_SEP,
        _diffx  <= -MIN_SEP,
        _diffy  >= MIN_SEP,
        _diffy  <= -MIN_SEP,
    ])
    assert MAX_SEP == -1, 'solnExistsDistPy: MAX_SEP invariant not yet implemented'
    return ok


def OKDist_MIN_SEP_1(agent1_action, curr_st1, agent2_action, curr_st2):
    """Runtime separation check (synthesised by buildSafetyShield).
    Handles MIN_SEP (collision avoidance) and MAX_SEP (stay-close) independently;
    each is skipped when set to -1.
    agent1_action = (ax1, ay1),  curr_st1 = (x1, y1, vx1, vy1)  (absolute coords)
    agent2_action = (ax2, ay2),  curr_st2 = (x2, y2, vx2, vy2)
    """
    if not DOING_SEP:
        return True

    ax1, ay1 = agent1_action
    x1,  y1,  vx1, vy1 = curr_st1
    ax2, ay2 = agent2_action
    x2,  y2,  vx2, vy2 = curr_st2

    # Convert to relative coordinates (matching the Z3 vars above)
    _diffx   = x1  - x2
    _diffy   = y1  - y2
    _diffv_x = vx1 - vx2
    _diffv_y = vy1 - vy2
    _diffa_x = ax1 - ax2
    _diffa_y = ay1 - ay2

    ok = fnAnd([
        _diffa_x >= -DIFFA_MAX,
        _diffa_x <= DIFFA_MAX,
        _diffa_y >= -DIFFA_MAX,
        _diffa_y <= DIFFA_MAX,
    ])

    assert MIN_SEP == 1, f'OKDist: this OKDist is only for MIN_SEP=1, got {MIN_SEP}'
    ok = ok and fnOr([
        _diffx + _diffv_x + 0.5 * _diffa_x <= -1,
        _diffx + _diffv_x + 0.5 * _diffa_x >= 1,
        _diffy + _diffv_y + 0.5 * _diffa_y >= 1,
        _diffy + _diffv_y + 0.5 * _diffa_y <= -1,
    ])
    assert MAX_SEP == -1, 'OKDist: this OKDist is only for MAX_SEP=-1, got {MAX_SEP}'

    return ok

    '''
    Synthesised for MIN_SEP=1, MAX_SEP=-1, DIFFA_MAX=20:
    0   diffa_x >= -20
    1   diffa_x <= 20
    2   diffa_y >= -20
    3   diffa_y <= 20
    4   Or(diffx + diffv_x + 1/2*diffa_x <= -1,
           diffx + diffv_x + 1/2*diffa_x >= 1,
           diffy + diffv_y + 1/2*diffa_y >= 1,
           diffy + diffv_y + 1/2*diffa_y <= -1)
    5   diffxX == diffx + diffv_x*1 + ((diffa_x*1)/2)*1
    6   diffv_xX == diffv_x + diffa_x*1
    7   diffyX == diffy + diffv_y*1 + ((diffa_y*1)/2)*1
    8   diffv_yX == diffv_y + diffa_y*1
    '''

''' ------------------  MIN_SEP = 2 ----------------------------------------'''
def solnExists(curr_st1, curr_st2):
    """State invariant for predator-predator separation (MIN_SEP=2).
    Synthesised: Or(diffx >= 2, diffx <= -2, diffy >= 2, diffy <= -2)
    """
    x1, y1, vx1, vy1 = curr_st1
    x2, y2, vx2, vy2 = curr_st2
    _diffx   = x1 - x2
    _diffy   = y1 - y2
    _diffv_x = vx1 - vx2
    _diffv_y = vy1 - vy2

    assert MIN_SEP == 2, f'solnExistsDistPy: invariant only synthesised for MIN_SEP=2, got {MIN_SEP}'
    ok = fnOr([
        _diffx  >= 2,
        _diffx  <= -2,
        _diffy  >= 2,
        _diffy  <= -2,
    ])
    assert MAX_SEP == -1, 'solnExistsDistPy: MAX_SEP invariant not yet implemented'
    return ok


def OKDist(agent1_action, curr_st1, agent2_action, curr_st2):
    """Runtime separation check (synthesised by buildSafetyShield, MIN_SEP=2).
    agent1_action = (ax1, ay1),  curr_st1 = (x1, y1, vx1, vy1)
    agent2_action = (ax2, ay2),  curr_st2 = (x2, y2, vx2, vy2)
    """
    if not DOING_SEP:
        return True

    ax1, ay1 = agent1_action
    x1,  y1,  vx1, vy1 = curr_st1
    ax2, ay2 = agent2_action
    x2,  y2,  vx2, vy2 = curr_st2

    # Convert to relative coordinates
    _diffx   = x1  - x2
    _diffy   = y1  - y2
    _diffv_x = vx1 - vx2
    _diffv_y = vy1 - vy2
    _diffa_x = ax1 - ax2
    _diffa_y = ay1 - ay2

    ok = fnAnd([
        _diffa_x >= -DIFFA_MAX,
        _diffa_x <= DIFFA_MAX,
        _diffa_y >= -DIFFA_MAX,
        _diffa_y <= DIFFA_MAX,
    ])

    assert MIN_SEP == 2, f'OKDist: this OKDist is only for MIN_SEP=2, got {MIN_SEP}'
    ok = ok and fnOr([
        _diffx + _diffv_x + 0.5 * _diffa_x <= -2,
        _diffx + _diffv_x + 0.5 * _diffa_x >= 2,
        _diffy + _diffv_y + 0.5 * _diffa_y >= 2,
        _diffy + _diffv_y + 0.5 * _diffa_y <= -2,
    ])
    assert MAX_SEP == -1, 'OKDist: this OKDist is only for MAX_SEP=-1, got {MAX_SEP}'

    return ok

    '''
    Synthesised for MIN_SEP=2, MAX_SEP=-1, DIFFA_MAX=20:
    0   diffa_x >= -20
    1   diffa_x <= 20
    2   diffa_y >= -20
    3   diffa_y <= 20
    4   Or(diffx + diffv_x + 1/2*diffa_x <= -2,
           diffx + diffv_x + 1/2*diffa_x >= 2,
           diffy + diffv_y + 1/2*diffa_y >= 2,
           diffy + diffv_y + 1/2*diffa_y <= -2)
    '''
