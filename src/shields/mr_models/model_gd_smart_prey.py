# Shield synthesis for single-predator tracking of an adversarial (smart) prey.
#
# State is the RELATIVE position/velocity of a predator w.r.t. the prey:
#   (diffx, diffy, diffv_x, diffv_y)
#   where diffx = x_pred - x_prey,  diffv_x = vx_pred - vx_prey, etc.
#
# The prey is adversarial: it chooses acceleration (prey_ax, prey_ay) from
#   [-A_PREY_MAX, A_PREY_MAX]^2 to try to escape.  (∀ quantifier)
# The predator chooses (a_pred_x, a_pred_y) from [-A_MAX, A_MAX]^2 to track. (∃ quantifier)
#
# Relative dynamics (DELTA_T = 1):
#   diffxX   = diffx  + diffv_x + (1/2)*(a_pred_x - prey_ax)
#   diffv_xX = diffv_x + (a_pred_x - prey_ax)
#   diffyX   = diffy  + diffv_y + (1/2)*(a_pred_y - prey_ay)
#   diffv_yX = diffv_y + (a_pred_y - prey_ay)
#
# This collapses to a single-agent 4-D problem with "net" relative acceleration
#   diffa_x = a_pred_x - prey_ax   (∈ [-A_MAX-A_PREY_MAX, A_MAX+A_PREY_MAX]),
# but exposing the two variables separately lets ∀∃ synthesis treat the prey
# acceleration as an adversary and the predator acceleration as the controller.
#
# Safety property: predator stays within MAX_TRACK_DIST of prey in L-inf norm:
#   -MAX_TRACK_DIST <= diffx <= MAX_TRACK_DIST
#   -MAX_TRACK_DIST <= diffy <= MAX_TRACK_DIST
# Optional min separation (e.g. physical collision avoidance, usually disabled here):
#   Or(diffx >= MIN_TRACK_DIST, diffx <= -MIN_TRACK_DIST,
#      diffy >= MIN_TRACK_DIST, diffy <= -MIN_TRACK_DIST)
#
# To run synthesis: set the import in buildShield.py to
#   from shields.mr_models.model_gd_smart_prey import *
# then run buildShield.py.  The printed controlPred at fixpoint is the body of OKTrack.

from z3    import *
from types import SimpleNamespace
from config_gd import DEFAULT_CONFIG

''' ------------------ CONSTANTS ------------------------------'''
DELTA_T      = DEFAULT_CONFIG['env_config']['STEP_SIZE']
A_MAX        = DEFAULT_CONFIG['env_config']['max_acceleration']
A_MIN        = -A_MAX
A_PREY_MAX   = DEFAULT_CONFIG['env_config'].get('A_PREY_MAX',   A_MAX // 2)
A_PREY_MIN   = -A_PREY_MAX
MAX_TRACK_DIST = DEFAULT_CONFIG['env_config'].get('MAX_TRACK_DIST', 8)
MIN_TRACK_DIST = DEFAULT_CONFIG['env_config'].get('MIN_TRACK_DIST', -1)

print(f'A_MAX={A_MAX}, A_PREY_MAX={A_PREY_MAX}, '
      f'MAX_TRACK_DIST={MAX_TRACK_DIST}, MIN_TRACK_DIST={MIN_TRACK_DIST}')

'''-------------------------- Z3 VARS -------------------------------------'''
# Relative state: position and velocity of predator relative to prey
diffx,   diffxX,   diffv_x, diffv_xX = Reals('diffx diffxX diffv_x diffv_xX')
diffy,   diffyX,   diffv_y, diffv_yX = Reals('diffy diffyX diffv_y diffv_yX')

# Control vars: predator acceleration (∃ — chosen by controller)
a_pred_x, a_pred_y = Reals('a_pred_x a_pred_y')

# Env vars: prey acceleration (∀ — adversary)
prey_ax, prey_ay = Reals('prey_ax prey_ay')

state     = [diffx,  diffy,  diffv_x,  diffv_y]
stateX    = [diffxX, diffyX, diffv_xX, diffv_yX]
subst     = list(zip(state, stateX))
inv_subst = list(zip(stateX, state))

global_props = []

singleton = SimpleNamespace(
             name          = 'main',
             vars          = state,
             postVars      = stateX,
             subst         = subst,
             invariant     = True,
             stateInvDelta = True,
             )

def delta_v(a):
    return a * DELTA_T

def delta_s(v, a):
    deltav = delta_v(a)
    return v * DELTA_T + (deltav / 2) * DELTA_T

# Net relative acceleration is (a_pred - prey_a).
# Both variables appear explicitly so ∀∃ synthesis can quantify over them separately.
action = SimpleNamespace(
          name        = 'track',
          actionPred  = And(
                            diffxX   == diffx  + delta_s(diffv_x, a_pred_x - prey_ax),
                            diffv_xX == diffv_x + delta_v(a_pred_x - prey_ax),
                            diffyX   == diffy  + delta_s(diffv_y, a_pred_y - prey_ay),
                            diffv_yX == diffv_y + delta_v(a_pred_y - prey_ay),
                        ),
          envVars     = [prey_ax, prey_ay],
          envPred     = And(A_PREY_MIN <= prey_ax, prey_ax <= A_PREY_MAX,
                            A_PREY_MIN <= prey_ay, prey_ay <= A_PREY_MAX),
          controlVars = [a_pred_x, a_pred_y],
          controlPred = And(A_MIN <= a_pred_x, a_pred_x <= A_MAX,
                            A_MIN <= a_pred_y, a_pred_y <= A_MAX),
          precNode    = singleton,
          postNode    = singleton,
          )

transitions = [action]

model = SimpleNamespace(
         name                = 'predator-prey tracking (relative coords), 2-D adversarial',
         initNode            = singleton,
         nodes               = [singleton],
         transitions         = transitions,
        #  DOING_FORALL_EXISTS = True,
         )

initProps = []

# Safety: predator must stay within MAX_TRACK_DIST of prey (L-inf box)
max_track_props = ([-MAX_TRACK_DIST <= diffx, diffx <= MAX_TRACK_DIST,
                    -MAX_TRACK_DIST <= diffy, diffy <= MAX_TRACK_DIST]
                   if MAX_TRACK_DIST != -1 else [])

# Optional: predator-prey minimum separation (usually disabled for pursuit)
min_track_props = ([Or(diffx >= MIN_TRACK_DIST, diffx <= -MIN_TRACK_DIST,
                       diffy >= MIN_TRACK_DIST, diffy <= -MIN_TRACK_DIST)]
                   if MIN_TRACK_DIST != -1 else [])

safetyProps = max_track_props + min_track_props

# ---------------------------------------------------------------------------
# Runtime check functions (synthesised by buildShield.py, ∃∀ form).
# Parameters: MAX_TRACK_DIST=10, MIN_TRACK_DIST=-1, A_PREY_MAX=5, A_MAX=10, DELTA_T=1
# Pattern mirrors OKGeo / OKDist in model_gd.py / model_gd_dist.py.
# ---------------------------------------------------------------------------
def fnOr(xs):  return any(xs)
def fnAnd(xs): return all(xs)


def solnExistsTrack(curr_st_pred, curr_st_prey):
    """State invariant for predator-prey tracking (∃∀ synthesis, MAX_TRACK_DIST=10).
    Returns True iff the current relative state lies in the synthesised invariant region.
    Synthesised node invariant at fixpoint (iteration 4).
    """
    x_pred, y_pred, vx_pred, vy_pred = curr_st_pred
    x_prey, y_prey, vx_prey, vy_prey = curr_st_prey

    _diffx   = x_pred - x_prey
    _diffy   = y_pred - y_prey
    _diffv_x = vx_pred - vx_prey
    _diffv_y = vy_pred - vy_prey

    assert MAX_TRACK_DIST == 10, f'solnExistsTrack: invariant only synthesised for MAX_TRACK_DIST=10, got {MAX_TRACK_DIST}'
    assert MIN_TRACK_DIST == -1, f'solnExistsTrack: invariant only synthesised for MIN_TRACK_DIST=-1, got {MIN_TRACK_DIST}'
    assert A_PREY_MAX == 5,      f'solnExistsTrack: invariant only synthesised for A_PREY_MAX=5, got {A_PREY_MAX}'

    return fnAnd([
        _diffx                            >= -10,      #  diffx >= -10
        _diffx                            <=  10,      #  diffx <=  10
        _diffy                            >= -10,      #  diffy >= -10
        _diffy                            <=  10,      #  diffy <=  10
        2*_diffv_y + 2*_diffy             <=  25,      #  2*diffv_y + 2*diffy <= 25
        -_diffx - _diffv_x                <=  25/2,    # -diffx - diffv_x <= 25/2
        -_diffv_y - _diffy                <=  25/2,    # -diffv_y - diffy <= 25/2
        2*_diffx + 2*_diffv_x             <=  25,      #  2*diffx + 2*diffv_x <= 25
        2/3*_diffx + 4/3*_diffv_x        <=  40/3,    #  2/3*diffx + 4/3*diffv_x <= 40/3
        4/3*_diffv_y + 2/3*_diffy         <=  40/3,   #  4/3*diffv_y + 2/3*diffy <= 40/3
        -2*_diffv_y - _diffy              <=  20,      # -2*diffv_y - diffy <= 20
        -_diffx - 2*_diffv_x              <=  20,      # -diffx - 2*diffv_x <= 20
        6/5*_diffv_y + 2/5*_diffy         <=  13,      #  6/5*diffv_y + 2/5*diffy <= 13
        2/5*_diffx + 6/5*_diffv_x        <=  13,      #  2/5*diffx + 6/5*diffv_x <= 13
        -_diffx - 3*_diffv_x              <=  65/2,    # -diffx - 3*diffv_x <= 65/2
        -3*_diffv_y - _diffy              <=  65/2,    # -3*diffv_y - diffy <= 65/2
    ])


def OKTrack(agent_action, curr_st_pred, curr_st_prey):
    """Runtime tracking check (∃∀ synthesis, MAX_TRACK_DIST=10, A_PREY_MAX=5).
    Returns True iff agent_action keeps the relative state in the tracking invariant
    regardless of worst-case prey acceleration.

    agent_action  = (a_pred_x, a_pred_y)
    curr_st_pred  = (x_pred, y_pred, vx_pred, vy_pred)
    curr_st_prey  = (x_prey, y_prey, vx_prey, vy_prey)

    Synthesised controlPred at fixpoint (∃∀, iteration 4):
      0  a_pred_x >= -10
      1  a_pred_x <= 10
      2  a_pred_y >= -10
      3  a_pred_y <= 10
      4  -15 <= -a_pred_y  - 2*diffv_y - 2*diffy
      5  -15/2 <= 1/2*a_pred_x + diffx  + diffv_x
      6  -15/2 <= 1/2*a_pred_y + diffv_y + diffy
      7  -15 <= -a_pred_x  - 2*diffx  - 2*diffv_x
      8  -5  <= diffy  + 2*diffv_y + 3/2*a_pred_y
      9  -10/3 <= -a_pred_x - 2/3*diffx  - 4/3*diffv_x
     10  -10/3 <= -2/3*diffy - 4/3*diffv_y - a_pred_y
     11  -5  <= 3/2*a_pred_x + diffx  + 2*diffv_x
     12  -3  <= -6/5*diffv_y - 2/5*diffy  - a_pred_y
     13  -15/2 <= 5/2*a_pred_x + diffx  + 3*diffv_x
     14  -15/2 <= 3*diffv_y  + diffy  + 5/2*a_pred_y
     15  -3  <= -a_pred_x  - 2/5*diffx  - 6/5*diffv_x
    """
    apx, apy = agent_action
    x_pred, y_pred, vx_pred, vy_pred = curr_st_pred
    x_prey, y_prey, vx_prey, vy_prey = curr_st_prey

    _diffx   = x_pred - x_prey
    _diffy   = y_pred - y_prey
    _diffv_x = vx_pred - vx_prey
    _diffv_y = vy_pred - vy_prey

    assert MAX_TRACK_DIST == 10, f'OKTrack: invariant only synthesised for MAX_TRACK_DIST=10, got {MAX_TRACK_DIST}'
    assert MIN_TRACK_DIST == -1, f'OKTrack: invariant only synthesised for MIN_TRACK_DIST=-1, got {MIN_TRACK_DIST}'
    assert A_PREY_MAX == 5,      f'OKTrack: invariant only synthesised for A_PREY_MAX=5, got {A_PREY_MAX}'

    return fnAnd([
        apx                                        >= -10,     # 0
        apx                                        <=  10,     # 1
        apy                                        >= -10,     # 2
        apy                                        <=  10,     # 3
        -apy  - 2*_diffv_y - 2*_diffy             >= -15,     # 4
        0.5*apx + _diffx  + _diffv_x              >= -15/2,   # 5
        0.5*apy + _diffv_y + _diffy               >= -15/2,   # 6
        -apx  - 2*_diffx  - 2*_diffv_x            >= -15,     # 7
        _diffy + 2*_diffv_y + 1.5*apy             >= -5,      # 8
        -apx  - 2/3*_diffx  - 4/3*_diffv_x        >= -10/3,   # 9
        -2/3*_diffy - 4/3*_diffv_y - apy          >= -10/3,   # 10
        1.5*apx + _diffx  + 2*_diffv_x            >= -5,      # 11
        -6/5*_diffv_y - 2/5*_diffy - apy          >= -3,      # 12
        2.5*apx + _diffx  + 3*_diffv_x            >= -15/2,   # 13
        3*_diffv_y + _diffy + 2.5*apy             >= -15/2,   # 14
        -apx  - 2/5*_diffx  - 6/5*_diffv_x        >= -3,      # 15
    ])
