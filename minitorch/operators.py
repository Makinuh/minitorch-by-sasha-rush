"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    return x * y
    ":math:`f(x, y) = x * y`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')


def id(x):
    return x
    ":math:`f(x) = x`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')


def add(x, y):
    return x + y
    ":math:`f(x, y) = x + y`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')


def neg(x):
    return -x
    ":math:`f(x) = -x`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')


def lt(x, y):
    if x < y:
        return 1.0
    else:
        return 0.0
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')


def eq(x, y):
    if x == y:
        return 1.0
    else:
        return 0.0
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")


def max(x, y):
    if x > y:
        return x
    else:
        return y
    ":math:`f(x) =` x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")


def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    return abs(x - y) <= 1e-2


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    return 1.0 / (1.0 + exp(-x)) if x >= 0 else (exp(x)) / (1.0 + exp(x))


def relu(x):
    if x > 0:
        return x
    else:
        return 0
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    if x == math.log(x + EPS):
        return d / (x + EPS)
    else:
        return 0.0
    r"If :math:`f = log` as above, compute :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def inv(x):
    return 1 / x
    ":math:`f(x) = 1/x`"
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def inv_back(x, d):
    if x == (1.0 / x):
        return d / x
    else:
        return 0.0
    r"If :math:`f(x) = 1/x` compute :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x, d):
    if x > 0.0:
        return d
    else:
        return 0.0
    r"If :math:`f = relu` compute :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """

    def mapper(ls):
        return [fn(x) for x in ls]

    return mapper


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    fn = map(neg)
    return fn(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def zippo(ls1, ls2):
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return zippo


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """

    def function(ls):
        res = start
        for x in ls:
            res = fn(res, x)
        return res

    return function


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    return reduce(add, 0)(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    return reduce(mul, 1)(ls)
