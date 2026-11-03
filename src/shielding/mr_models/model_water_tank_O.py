# model of water tank taken from "Safe Reinforcement Learning via Shielding"
# formatted for mr.py (uses SimpleNamespace, controlVars plural, step_invariant, etc.)

from z3         import *
from types      import SimpleNamespace

Time = IntSort()

# State variables
t, tX       = Consts('t tX', Time)
s, sX       = Consts('s sX', Time)
tank, tankX = Reals('tank tankX')

state  = [t, s, tank]
stateX = [tX, sX, tankX]
subst  = list(zip(state, stateX))

# Env variables
inn = Real('inn')
out = Real('out')

# Control variable
open = Bool('open')

# Node
singleton = SimpleNamespace(
    name          = 'singleton',
    vars          = state,
    postVars      = stateX,
    subst         = subst,
    invariant     = True,
    stateInvDelta = True,   # initialised; updated by initInvsWithSafetyProps
)

# Arcs/Transitions
openAction = SimpleNamespace(
    name          = "openAction",
    actionPred    = (tankX == tank + inn - out),
    envVars       = [inn, out],
    envPred       = And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
    controlVars   = [open],
    controlPred   = open,
    strengthening = True,   # initialised; may be updated by mkWPC
    precNode      = singleton,
    postNode      = singleton,
)

closeAction = SimpleNamespace(
    name          = "closeAction",
    actionPred    = (tankX == tank - out),
    envVars       = [inn, out],
    envPred       = And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
    controlVars   = [open],
    controlPred   = Not(open),
    strengthening = True,   # initialised; may be updated by mkWPC
    precNode      = singleton,
    postNode      = singleton,
)

transitions = [openAction, closeAction]

model = SimpleNamespace(
    name           = 'Water Tank',
    initNode       = singleton,
    nodes          = [singleton],
    transitions    = transitions,
    step_invariant = True,   # arc safety props land here (not in actionPred)
)

# Safety properties
TANK_CAPACITY = 100
NUM_NEXTS     = 2

initProps = [tank == 0, out == 0]

safetyPropsOpen = [
    0 <= tank, tank <= TANK_CAPACITY,
    0 <= t,    t <= NUM_NEXTS,
    Implies(And(t == 0, open),      tX == 1),   # open  @ t=0 → t'=1
    Implies(And(t == 0, Not(open)), tX == 0),   # close @ t=0 → t'=0
    Implies(And(t == 1, Not(open)), tX == 2),   # close @ t=1 → t'=2 (obligation)
    Implies(And(t == 1, open),      tX == 1),   # open  @ t=1 → t'=1 (stay)
    Implies(t == 2, And(Not(open), tX == 0)),   # must close @ t=2
]

safetyPropsClsd = [
    0 <= tank, tank <= TANK_CAPACITY,
    0 <= s,    s <= NUM_NEXTS,
    Implies(And(s == 0, Not(open)), sX == 1),   # close @ s=0 → s'=1
    Implies(And(s == 0, open),      sX == 0),   # open  @ s=0 → s'=0
    Implies(And(s == 1, open),      sX == 2),   # open  @ s=1 → s'=2 (obligation)
    Implies(And(s == 1, Not(open)), sX == 1),   # close @ s=1 → s'=1 (stay)
    Implies(s == 2, And(open, sX == 0)),        # must open  @ s=2
]

safetyProps = safetyPropsOpen + safetyPropsClsd

def fnOr(xs): return any(xs)
def fnAnd(xs): return all(xs)
def fnNot(x): return not x

def solnExists(tank_level, t):
  return fnAnd( tank_level >= 0,
                tank_level <= 100,
                t >= 0,
                t <= NUM_NEXTS)

def OK(open_valve, tank_level, t, s):
    return OKProp(open_valve, tank_level, t, s)

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
      if tank_level > 98 and s != 2: return (False, t, s)
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
      if tank_level < 1 and t != 2: return (False, t, s)
      match t:
         case 0: new_t = 0                       #close @ t=0, no obligation
         case 1: new_t = 2                       #close after one open → obligation starts
         case 2: new_t = 0                       #close @ t=2, obligation fulfilled → reset
      match s:
         case 0: new_s = 1                       #close @ s=0, record first close
         case 1: new_s = 1                       #close @ s=1, consecutive closes stay
         case 2: return (False, t, s)            #close @ s=2, violation (must open)
      return (True, new_t, new_s)
