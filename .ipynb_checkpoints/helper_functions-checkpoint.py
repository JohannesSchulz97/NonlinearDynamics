import numpy as np

def logistic_iterator(x, r):
    return r * x * (1-x)
    
def iterate_map(x0, r, n, iterator=logistic_iterator):
    x = x0
    xs = [x0]
    while n>0:
        x = iterator(x,r)
        xs.append(x)
        n -= 1
    return np.array(xs)

def logistic_map(x0, r, n): 
    return iterate_map(x0, r, n)
