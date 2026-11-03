
from z3        import *

def toZ3Type(v):                          #wraps the vals in their Z3 type so Z3 doesnt complain
  if isinstance(v,bool)   : return BoolVal(v)
  if isinstance(v,int)    : return IntVal(v)
  if isinstance(v,float)  : return RealVal(v)
  if isinstance(v,str)    : return StringVal(v)
  raise BaseException('toZ3Type: unknown type for value ', v)