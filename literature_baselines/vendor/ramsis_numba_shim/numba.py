"""Pure-Python stand-in for numba.jit (numba is not installed): returns the function unchanged."""
def jit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f
