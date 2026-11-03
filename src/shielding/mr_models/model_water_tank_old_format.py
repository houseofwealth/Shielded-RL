# model of water tank taken from "Safe Reinforcement Learning via Shielding"

from unittest import case

from z3       import *

Time = IntSort() 

#  state  variables
t, tX       = Consts('t tX', Time)
s, sX       = Consts('s sX', Time)
tank, tankX = Reals('tank tankX')
globals     = []
# state       = [t, tank] 
# stateX      = [tX,tankX]
state       = [t, s, tank] 
stateX      = [tX,sX,tankX]
subst       = list(zip(state,stateX))
stateInv    = True 

# Env variables
inn = Real('inn')
out = Real('out')

# Control var
open = Bool('open')

# Nodes
singleton  = {'name'         : 'singleton',
              'globals'      : globals,
              'vars'         : state,
              'postVars'     : stateX,    
              'subst'        : subst,    
              'invariant'    : True}  

openAction = {  'name'       : "openAction",
                'actionPred' : And( #moved -> U: open,      
                                    tankX == tank + inn - out), 
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'invariant'  : True,
                'controlVar' : [open],
                #'controlPred': And(open,Implies(t==2, Not(open))),    
                #this is a hack b/c of entire guard not being updated. w/o it, chooses U = ~open
                #this is open /\ (t=2 -> ~open /\ t=2 -> ~open )
                #ie And(open,Implies(Or(t==2, t==3), Not(open))), which simplifies to:
                'controlPred': open, #, #And(open, t!=2, t!=3),   
                'precNode'   : singleton,
                'postNode'   : singleton
                }

closeAction = { 'name'       : "closeAction",
                'actionPred' : And( #moved -> U: Not(open), 
                                    tankX == tank - out),
                'envVar'     : [inn, out],
                'envPred'    : And(1 <= inn, inn <= 2, 0 <= out, out <= 1),
                'invariant'  : True,
                'controlVar' : [open],
                # why does this: 'controlPred': And(Not(open), s!=2, s!=3), instead of the one below lead to a *more* complex guard on the two actions? two extraneous conditions show up
                'controlPred': Not(open), #True, #And(Not(open), Implies(Or(s==2, s==3), open)),
                'precNode'   : singleton,
                'postNode'   : singleton
              }

model       = {'name'        : 'Water Tank',
               'initNode'    : singleton,
               'nodes'       : [singleton],
               'transitions' : [openAction, closeAction]
              }

# Required Properties : open & Xclosed --> XXclosed, not whats below!
""" open & Xclosed --> XXclosed & XXXclosed == 
    ~open | X~closed | (XX~open & XXX ~open) ===
    (closed | Xopen | XXclosed) & (closed | Xopen | XXXclosed)
"""
TANK_CAPACITY = 100
NUM_NEXTS = 2 #NOTE: FIX THIS FOR WHEN YOU ADD MORE TERMS!!
initProps   = [tank == 0, out == 0]
safetyPropsOpen = [ 0 <= tank, tank <= TANK_CAPACITY,
                    0 <= t, t <= NUM_NEXTS,
                    Implies(And(t==0, open), tX==1),        #at the top level, if open, inc the counter
                    Implies(And(t==0, Not(open)), tX==0),   #if its not open, reset

                    Implies(And(t==1, Not(open)), tX==2),   #if its closed at level 2, inc the counter
                    Implies(And(t==1, open), tX==1),        #if its open, keep tracking

                  #   Implies(t==2, tX==3),                   #inc the counter
                  #   Implies(t==3, tX==0),                   #end of cycle, reset
                    Implies(t==2, And(Not(open), tX==0)),
                    #for the reqts on t==2 and t==3 see openActions's control pred
                  ]
safetyPropsClsd = [ 0 <= tank, tank <= TANK_CAPACITY,
                    0 <= s, s <= NUM_NEXTS,
                    Implies(And(s==0, Not(open)), sX==1),   #at the top level, if closed, inc the counter
                    Implies(And(s==0, open), sX==0),        #if its open, reset

                    Implies(And(s==1, open), sX==2),        #if its open at level 2, inc the counter
                    Implies(And(s==1, Not(open)), sX==1),   #if its closed, reset

                  #   Implies(s==2, sX==3),                   #inc the counter
                  #   Implies(s==3, sX==0),                   #end of cycle, reset
                    Implies(s==2, And(open, sX==0)),                   #inc the counter
                    #for the reqts on t==2 and t==3 see closedActions's control pred
                  ]              #   0 <= s, s <= NUM_NEXTS,
              #   Implies(And(s==0, Not(open)), sX==1),   #at the top level, if not open, inc the counter
              #   Implies(And(s==0, open), sX==0),        #if its  open, no change
              #   Implies(And(s==1, open), sX==2),        #if its open at level 2, inc the counter
              #   Implies(And(s==1, Not(open)), sX==0),   #if its closed, reset
              #  #  Implies(s==2, sX==0)                  #end of cycle, resetj
              #   Implies(s==2, sX==3),                   #inc the counter
              #   Implies(s==3, sX==0)                    #end of cycle, reset
safetyProps = safetyPropsOpen + safetyPropsClsd

def fnOr(xs): return any(xs)
def fnAnd(xs): return all(xs)
def fnNot(x): return not x

def solnExists(tank_level, t):
  return fnAnd( tank_level >= 0,
                tank_level <= 100,
                t >= 0,
                t <= NUM_NEXTS)
               #  Or(Not(t == 3), 1 <= tank),
               #  Or(Not(t == 2), 2 <= tank))

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





"""
with just safetyPropsOpen produces:
stateInv
 And(tank >= 0, tank <= 100, t >= 0, t <= 2)
newInvDelta:
 []
no change to invariant on singleton

----------------
Final Model - fixpoint at iteration 2
Node: singleton
  Vars: [t, s, tank]
  Node Invariant: True
Transition: openAction
   0   open
   1   Or(tank <= 98, t == 2)
   2   inn >= 1
   3   inn <= 2
   4   out >= 0
   5   out <= 1
   6   tankX == tank + inn + -1*out
   7   Or(Not(open), tX == 1, Not(t == 0))
   8   Or(open, tX == 0, Not(t == 0))
   9   Or(open, tX == 2, Not(t == 1))
   10   Or(Not(open), tX == 0, Not(t == 1))
   11   Or(Not(t == 2), And(Not(open), tX == 0))
Transition: closeAction
   0   Not(open)
   1   1 <= tank
   2   inn >= 1
   3   inn <= 2
   4   out >= 0
   5   out <= 1
   6   tankX == tank + -1*out
   7   Or(Not(open), tX == 1, Not(t == 0))
   8   Or(open, tX == 0, Not(t == 0))
   9   Or(open, tX == 2, Not(t == 1))
   10   Or(Not(open), tX == 0, Not(t == 1))
   11   Or(Not(t == 2), And(Not(open), tX == 0))

..safety shield construction complete


------------------ when safetyPropsClsd is added you get: --------------------
stateInv
 And(tank >= 0, tank <= 100, t >= 0, t <= 2, s >= 0, s <= 2)
----------------
Final Model - fixpoint at iteration 2
Node: singleton
  Vars: [t, s, tank]
  Node Invariant: True
Transition: openAction
   0   open
   1   Or(t == 2, tank <= 98)
   2   inn >= 1
   3   inn <= 2
   4   out >= 0
   5   out <= 1
   6   tankX == tank + inn + -1*out

   7   Or(Not(open), tX == 1, Not(t == 0))
   8   Or(open, tX == 0, Not(t == 0))
   9   Or(open, tX == 2, Not(t == 1))
   10   Or(Not(open), tX == 1, Not(t == 1))
   11   Or(Not(t == 2), And(Not(open), tX == 0))

   12   Or(open, Not(s == 0), sX == 1)
   13   Or(Not(s == 0), Not(open), sX == 0)
   14   Or(Not(open), sX == 2, Not(s == 1))
   15   Or(open, sX == 1, Not(s == 1))
   16   Or(Not(s == 2), And(open, sX == 0))

   
Transition: closeAction
   0   Not(open)
   1   Or(1 <= tank, s == 2)
   2   inn >= 1
   3   inn <= 2
   4   out >= 0
   5   out <= 1
   6   tankX == tank + -1*out

   7   Or(Not(open), tX == 1, Not(t == 0))
   8   Or(open, tX == 0, Not(t == 0))
   9   Or(open, tX == 2, Not(t == 1))
   10   Or(Not(open), tX == 1, Not(t == 1))
   11   Or(Not(t == 2), And(Not(open), tX == 0))

   12   Or(open, Not(s == 0), sX == 1)
   13   Or(Not(s == 0), Not(open), sX == 0)
   14   Or(Not(open), sX == 2, Not(s == 1))
   15   Or(open, sX == 1, Not(s == 1))
   16   Or(Not(s == 2), And(open, sX == 0))

..safety shield construction complete


---------------------------------------------------------------------------------
but when you simplify only one of the initial control preds (see above) but not both you get a simpler solution:
----------------
Final Model - fixpoint at iteration 4
Node: singleton
  Vars: [t, s, tank]
  Node Invariant: True
Transition: openAction
   0   open
   1   Not(t == 2)
   2   Not(t == 3)
   3   Or(s == 2, s == 3, tank <= 94, And(s == 0, tank <= 98))
   4   inn >= 1
   5   inn <= 2
   6   out >= 0
   7   out <= 1
   8   tankX == tank + inn + -1*out
   9   Or(Not(open), tX == 1, Not(t == 0))
   10   Or(open, tX == 0, Not(t == 0))
   11   Or(open, tX == 2, Not(t == 1))
   12   Or(Not(open), tX == 0, Not(t == 1))
   13   Or(Not(t == 2), tX == 3)
   14   Or(Not(t == 3), tX == 0)
   15   Or(open, Not(s == 0), sX == 1)
   16   Or(Not(s == 0), Not(open), sX == 0)
   17   Or(Not(open), sX == 2, Not(s == 1))
   18   Or(open, sX == 0, Not(s == 1))
   19   Or(Not(s == 2), sX == 3)
   20   Or(Not(s == 3), sX == 0)
Transition: closeAction
   0   Not(open)
   1   Not(s == 2)
   2   Not(s == 3)
   3   Or(t == 2,
   t == 3,
   And(t == 0, 1 <= tank),
   Not(tank <= 97),
   3 <= tank)
   4   inn >= 1
   5   inn <= 2
   6   out >= 0
   7   out <= 1
   8   tankX == tank + -1*out
   9   Or(Not(open), tX == 1, Not(t == 0))
   10   Or(open, tX == 0, Not(t == 0))
   11   Or(open, tX == 2, Not(t == 1))
   12   Or(Not(open), tX == 0, Not(t == 1))
   13   Or(Not(t == 2), tX == 3)
   14   Or(Not(t == 3), tX == 0)
   15   Or(open, Not(s == 0), sX == 1)
   16   Or(Not(s == 0), Not(open), sX == 0)
   17   Or(Not(open), sX == 2, Not(s == 1))
   18   Or(open, sX == 0, Not(s == 1))
   19   Or(Not(s == 2), sX == 3)
   20   Or(Not(s == 3), sX == 0)

..safety shield construction complete

"""