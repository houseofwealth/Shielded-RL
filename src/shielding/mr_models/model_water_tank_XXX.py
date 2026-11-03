# model of water tank taken from "Safe Reinforcement Learning via Shielding"
# converted from shieldedRL/prototype format to mr.py format

from z3          import *
from types       import SimpleNamespace

Time = IntSort()
global_props = [] # globals() is used by Python

# State variables
t, tX    = Consts('t tX', Time)
s, sX    = Consts('s sX', Time)
tank, tankX = Reals('tank tankX')

state  = [t, s, tank]
stateX = [tX, sX, tankX]
subst  = list(zip(state, stateX))

# Env variables
inn = Real('inn')
out = Real('out')

# Control variable
open = Bool('open')
NUM_NEXTS     = 2

# Node
singleton = SimpleNamespace(
    name         = 'singleton',
    vars         = state,
    postVars     = stateX,
    subst        = subst,
    invariant    = And( 0 <= t, t <= NUM_NEXTS, 0 <= s, s <= NUM_NEXTS), #True
    stateInvDelta = True,   # initialised; updated by initInvsWithSafetyProps
)

# Arcs/Transitions
"""open ∧ Xclosed → XXclosed  tracked by t (t=1: just opened, t=2: [open,close] obligation)
 """
openAction = SimpleNamespace(
    name        = "openAction",
    actionPred  = 
        And(tankX == tank + inn - out,
            Implies(And(t == 0, open),      tX == 1),   # open @ t=0 → t'=1
            Implies(And(t == 1, open),      tX == 1),   # open  @ t=1 → t'=1 (stay)
            Implies(And(s == 0, open),      sX == 0),   # open  @ s=0 → s'=0
            Implies(And(s == 1, open),      sX == 2),   # open  @ s=1 → s'=2 (obligation)
            Implies(s == 2, And(open, sX == 0))),        # must open @ s=2
    envVars     = [inn, out],
    envPred     = And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
    controlVars = [open],
    controlPred = open,
    strengthening = True,   # initialised; may be updated by mkWPC
    precNode    = singleton,
    postNode    = singleton,
)

"""closed ∧ Xopen → XXopen    tracked by s (s=1: just closed, s=2: [close,open] obligation)
 """
closeAction = SimpleNamespace(
    name        = "closeAction",
    actionPred  = 
        And(tankX == tank - out,
            Implies(And(s == 0, Not(open)), sX == 1),   # close @ s=0 → s'=1
            Implies(And(t == 0, Not(open)), tX == 0),   # close @ t=0 → t'=0
            Implies(And(t == 1, Not(open)), tX == 2),   # close @ t=1 → t'=2 (obligation)
            Implies(And(s == 1, Not(open)), sX == 1),   # close @ s=1 → s'=1 (stay)
            Implies(t == 2, And(Not(open), tX == 0))),   # must close @ t=2),
    envVars     = [inn, out],
    envPred     = And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
    controlVars = [open],
    controlPred = Not(open),
    strengthening = True,   # initialised; may be updated by mkWPC
    precNode    = singleton,
    postNode    = singleton,
)

transitions = [openAction, closeAction]

model = SimpleNamespace(
    name           = 'Water Tank',
    initNode       = singleton,
    nodes          = [singleton],
    transitions    = transitions,
    step_invariant = True,   # arc safety props that are not obivously step props ie they dont involve both pre and post vars, land here (otehrwise in actionPred)
)

# Safety properties
TANK_CAPACITY = 100

initProps = [tank == 0, out == 0]
safetyProps = [0 <= tank, tank <= TANK_CAPACITY]

""" if you want to do the props for Open and Closed seperately need to remove whichever from the corresponding action def. They are not properties to be satisfied but essentially step invariants being asserted
safetyPropsOpen = [
    0 <= tank, tank <= TANK_CAPACITY,
    # 0 <= t,    t <= NUM_NEXTS,
    # Implies(And(t == 0, open),      tX == 1),   # open @ t=0 → t'=1
    # Implies(And(t == 0, Not(open)), tX == 0),   # close @ t=0 → t'=0
    # Implies(And(t == 1, Not(open)), tX == 2),   # close @ t=1 → t'=2 (obligation)
    # Implies(And(t == 1, open),      tX == 1),   # open  @ t=1 → t'=1 (stay)
    # Implies(t == 2, And(Not(open), tX == 0)),   # must close @ t=2
]

safetyPropsClsd = [
    0 <= tank, tank <= TANK_CAPACITY,
    # 0 <= s,    s <= NUM_NEXTS,
    # Implies(And(s == 0, Not(open)), sX == 1),   # close @ s=0 → s'=1
    # Implies(And(s == 0, open),      sX == 0),   # open  @ s=0 → s'=0
    # Implies(And(s == 1, open),      sX == 2),   # open  @ s=1 → s'=2 (obligation)
    # Implies(And(s == 1, Not(open)), sX == 1),   # close @ s=1 → s'=1 (stay)
    # Implies(s == 2, And(open, sX == 0)),        # must open @ s=2
]
safetyProps = safetyPropsOpen + safetyPropsClsd """

def fnOr(xs): return any(xs)
def fnAnd(xs): return all(xs)
def fnNot(x): return not x

""" And(tank >= 0,
    tank <= 100,
    Or(And(t == 0, s == 2, tank <= 98),
       And(1 == s, t == 1, tank <= 98),
       And(t == 1, s == 0, tank <= 98),
       And(s == 0, t == s, tank <= 98),
       And(t == 0, s == 1, tank <= 98),
       And(s == 1, t == 2, 1 <= tank),
       And(t == 1, s == 0, 1 <= tank),
       And(1 == s, t == 1, Not(1 <= tank)),
       And(s == 0, t == 2, 1 <= tank),
       And(t == 1, s == 2, tank <= 98),
       And(t == 0, s == 1, 1 <= tank),
       And(1 == s, t == 1, 1 <= tank),
       And(s == 0, t == s, 1 <= tank))) """

def solnExists(tank_level, t, s):
  return fnAnd([tank_level >= 0,
                tank_level <= 100,
                fnOr([fnAnd([t == 0, s == 2, tank_level <= 98]),
                      fnAnd([s == 1, t == 1, tank_level <= 98]),
                      fnAnd([t == 1, s == 0, tank_level <= 98]),
                      fnAnd([s == 0, t == s, tank_level <= 98]),
                      fnAnd([t == 0, s == 1, tank_level <= 98]),
                      fnAnd([s == 1, t == 2, 1 <= tank_level]),
                      fnAnd([t == 1, s == 0, 1 <= tank_level]),
                      fnAnd([s == 1, t == 1, fnNot(1 <= tank_level)]),
                      fnAnd([s == 0, t == 2, 1 <= tank_level]),
                      fnAnd([t == 1, s == 2, tank_level <= 98]),
                      fnAnd([t == 0, s == 1, 1 <= tank_level]),
                      fnAnd([s == 1, t == 1, 1 <= tank_level]),
                      fnAnd([s == 0, t == s, 1 <= tank_level])])])

def OK(open_valve, tank_level, t, s):
    return OKProp(open_valve, tank_level, t, s)

""" Transition: openAction
   0   open
   1   Or(And(t == 1, 1 == s, Not(1 <= tank)),
   And(t == 1, 1 == s),
   And(s == 1, t == 0),
   And(s == 0, t == 1, tank <= 98),
   And(t == s, s == 0, tank <= 98),
   And(s == 2, t == 0),
   And(s == 2, t == 1))
   2   Or(And(t == 0, s == 1, tank <= 96),
   And(t == 0, s == 2),
   And(t == 1, s == 2),
   And(t == 0, 0 == s),
   And(t == s, s == 1, tank <= 96),
   And(t == 1, s == 0))
   3   Or(And(t == 1, s == 0, tank <= 98),
   And(s == 0, t == s, tank <= 98),
   And(t == 0, s == 2),
   And(t == 1, s == 2),
   And(1 == s, t == 1, Not(1 <= tank)),
   And(1 == s, t == 1),
   And(t == 0, s == 1))
   4   Or(And(0 == s, t == 0),
   And(s == 1, t == s, tank <= 96),
   And(s == 1, t == 0, tank <= 96),
   And(s == 2, t == 0),
   And(s == 0, t == 1),
   And(s == 2, t == 1))
   5   tankX == tank + inn - out
   6   Implies(And(t == 0, open), tX == 1)
   7   Implies(And(t == 1, open), tX == 1)
   8   Implies(And(s == 0, open), sX == 0)
   9   Implies(And(s == 1, open), sX == 2)
   10   Implies(s == 2, And(open, sX == 0))
   
Transition: closeAction
   0   Not(open)
   1   Or(And(t == 1, 1 == s),
   And(t == 2, s == 1),
   And(t == s, s == 0, 1 <= tank),
   And(s == 1, t == 0, 1 <= tank),
   And(t == 2, s == 0),
   And(s == 0, t == 1))
   2   Or(And(t == s, s == 0),
   And(t == 1, 1 == s, 2 <= tank),
   And(s == 1, t == 0),
   And(t == 2, s == 1),
   And(t == 1, s == 0, 2 <= tank),
   And(t == 2, s == 0))
   3   Or(And(s == 1, t == 2),
   And(t == 0, s == 1, 1 <= tank),
   And(s == 0, t == 2),
   And(s == 0, t == s, 1 <= tank),
   And(1 == s, t == 1),
   And(t == 1, s == 0))
   4   Or(And(s == 1, t == 2),
   And(s == 0, t == 2),
   And(s == 0, t == s),
   And(t == 0, s == 1),
   And(s == 0, t == 1, 2 <= tank),
   And(1 == s, t == 1, 2 <= tank))
   5   tankX == tank - out
   6   Implies(And(s == 0, Not(open)), sX == 1)
   7   Implies(And(t == 0, Not(open)), tX == 0)
   8   Implies(And(t == 1, Not(open)), tX == 2)
   9   Implies(And(s == 1, Not(open)), sX == 1)
   10   Implies(t == 2, And(Not(open), tX == 0))
 """

'''TWO CONSTRAINTS, EACH WITH 2 NEXTS:
   open∧Xclosed→XXclosed  tracked by t (t=1: just opened, t=2: [open,close] obligation)
   closed∧Xopen→XXopen    tracked by s (s=1: just closed, s=2: [close,open] obligation)
***NOTE: if an action choice fails, agent needs to retry with the negated choice, otherwise counters wont be correctly updated'''
def OKProp(open_valve, tank_level, t, s):
   # print(f'DEBUG OKProp: open_valve={open_valve!r} ({type(open_valve).__name__}), tank={tank_level!r} ({type(tank_level).__name__}), t={t!r} ({type(t).__name__}), s={s!r} ({type(s).__name__})')
   assert t < 3, f't should never reach 3 if the shield is correct, but got t={t}'
   assert s < 3, f's should never reach 3 if the shield is correct, but got s={s}'
   if open_valve:
      # s=2 is a hard obligation to open — overrides the high-tank guard to avoid deadlock
      if not fnAnd([
            fnOr([fnAnd([t == 1, s == 1, fnNot(1 <= tank_level)]),
                  fnAnd([t == 1, s == 1]),
                  fnAnd([s == 1, t == 0]),
                  fnAnd([s == 0, t == 1, tank_level <= 98]),
                  fnAnd([t == s, s == 0, tank_level <= 98]),
                  fnAnd([s == 2, t == 0]),
                  fnAnd([s == 2, t == 1])]),
            fnOr([fnAnd([t == 0, s == 1, tank_level <= 96]),
                  fnAnd([t == 0, s == 2]),
                  fnAnd([t == 1, s == 2]),
                  fnAnd([t == 0, s == 0]),
                  fnAnd([t == s, s == 1, tank_level <= 96]),
                  fnAnd([t == 1, s == 0])]),
            fnOr([fnAnd([t == 1, s == 0, tank_level <= 98]),
                  fnAnd([s == 0, t == s, tank_level <= 98]),
                  fnAnd([t == 0, s == 2]),
                  fnAnd([t == 1, s == 2]),
                  fnAnd([s == 1, t == 1, fnNot(1 <= tank_level)]),
                  fnAnd([s == 1, t == 1]),
                  fnAnd([t == 0, s == 1])]),
            fnOr([fnAnd([s == 0, t == 0]),
                  fnAnd([s == 1, t == s, tank_level <= 96]),
                  fnAnd([s == 1, t == 0, tank_level <= 96]),
                  fnAnd([s == 2, t == 0]),
                  fnAnd([s == 0, t == 1]),
                  fnAnd([s == 2, t == 1])])]):
        return (False, t, s)
      match t:
         case 0: new_t = 1                       #open @ t=0, record first open
         case 1: new_t = 1                       #open @ t=1, consecutive opens stay
         case 2: return (False, t, s)            #open @ t=2, violation (must close)
      match s:
         case 0: new_s = 0                       #open @ s=0, no obligation
         case 1: new_s = 2                       #open after one close → obligation starts
         case 2: new_s = 0                       #open @ s=2, obligation fulfilled → reset
      return (True, new_t, new_s)
   else:
      # t=2 is a hard obligation to close — overrides the low-tank guard to avoid deadlock
      if not fnAnd([
            fnOr([fnAnd([t == 1, s == 1]),
                  fnAnd([t == 2, s == 1]),
                  fnAnd([t == s, s == 0, 1 <= tank_level]),
                  fnAnd([s == 1, t == 0, 1 <= tank_level]),
                  fnAnd([t == 2, s == 0]),
                  fnAnd([s == 0, t == 1])]),
            fnOr([fnAnd([t == s, s == 0]),
                  fnAnd([t == 1, s == 1, 2 <= tank_level]),
                  fnAnd([s == 1, t == 0]),
                  fnAnd([t == 2, s == 1]),
                  fnAnd([t == 1, s == 0, 2 <= tank_level]),
                  fnAnd([t == 2, s == 0])]),
            fnOr([fnAnd([s == 1, t == 2]),
                  fnAnd([t == 0, s == 1, 1 <= tank_level]),
                  fnAnd([s == 0, t == 2]),
                  fnAnd([s == 0, t == s, 1 <= tank_level]),
                  fnAnd([s == 1, t == 1]),
                  fnAnd([t == 1, s == 0])]),
            fnOr([fnAnd([s == 1, t == 2]),
                  fnAnd([s == 0, t == 2]),
                  fnAnd([s == 0, t == s]),
                  fnAnd([t == 0, s == 1]),
                  fnAnd([s == 0, t == 1, 2 <= tank_level]),
                  fnAnd([s == 1, t == 1, 2 <= tank_level])])]):
        return (False, t, s)
      match t:
         case 0: new_t = 0                       #close @ t=0, no obligation
         case 1: new_t = 2                       #close after one open → obligation starts
         case 2: new_t = 0                       #close @ t=2, obligation fulfilled → reset
      match s:
         case 0: new_s = 1                       #close @ s=0, record first close
         case 1: new_s = 1                       #close @ s=1, consecutive closes stay
         case 2: return (False, t, s)            #close @ s=2, violation (must open)
      return (True, new_t, new_s)
