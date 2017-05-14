#!/etc/bin/env python3

# Aim at Unconstrained optimization
# Using Traceback linear searching
# Default use gradient descent direction but alternative

# Waring: In this program, use numpy.array to replace all lists and tuples
#           pay attention when you transfer argus into functions

import sympy as sp
import numpy as np

# Get first-order derivation
def GetDF(f,args):return np.array([sp.diff(f,x) for x in args])
# Get substitution dict
def GetSubs(args,point):return {args[i]:point[i] for i in range(len(args))}
# Plug x into f
def GetValue(f,args,point):
    if isinstance(f,np.ndarray):
        return np.array([i.evalf(subs=GetSubs(args,point)) for i in f]).astype('float')
    else:
        return np.array([f.evalf(subs=GetSubs(args, point))]).astype('float')

def GetStep(f,
            args,a,b,
            direction,point,
            former,df):
    # Init
    t = 1
    dfValue = GetValue(df, args, point)
    # Calc x = x+td
    d = GetValue(direction,args,point)
    x = point + t*d
    # Traceback searching
    while GetValue(f,args,x) > former + t*a*np.dot(dfValue,d) :
        # That means can not find next step
        # if a and e is not reasonable, this method may can not find next step
        if t == 0:break
        t = b * t
        # Calc x = point +td
        x = point + t*d
    return t

def Search(f, args, a, b, initPoint, e):
    # Init
    fStack = []
    x = initPoint

    fValue = GetValue(f,args,x)
    fStack.append(fValue)

    df = GetDF(f,args)
    # gradient descent direction
    direction = np.array([-x for x in df])

    # Search
    while True:
        # save array as 'float' instead of 'object' to avoid error of function "np.linalg.norm"
        dfValue = GetValue(df,args,x)
        directionValue = GetValue(direction, args, x)
        # Check
        if np.linalg.norm(dfValue, 2) < e: break
        t = GetStep(f,args,a,b,direction,x,fStack[-1:][0],df)
        # That means can not find next step due to the limit of a
        if t == 0:
            print("t == 0 ")
            break
        x = x + t*directionValue
        fValue = GetValue(f,args,x)
        fStack.append(fValue)
    print("The optimize solution: "+str(x))
    print("error = ",end ="")
    print(np.linalg.norm(dfValue, 2))
    return fStack

if __name__ != "__main__":
    x1 = sp.Symbol("x1")
    x2 = sp.Symbol("x2")
    f = 2*x1*x1+2*x2
    print(GetValue(f,np.array([x1,x2]),np.array([1,2])))
    print(GetDF(f,np.array([x1,x2])))
    print(GetSubs(np.array([x1,x2]),np.array([1,2])))

if __name__ == "__main__":
    X1 = sp.Symbol("x1")
    X2 = sp.Symbol("x2")
    args = np.array((X1,X2))
    # function
    f = sp.exp(X1 + 3 * X2 - 0.1) + sp.exp(X1 - 3 * X2 - 0.1) + sp.exp(-X1 - 0.1)
    # Init point
    x0 = np.array([1, 1])
    # Error range
    e = 0.001
    # α&β for back tracking
    a = 0.01
    b = 0.1
    # Start
    fStack = Search(f,args,a,b,x0,e)
    for i in fStack:
        print(i)