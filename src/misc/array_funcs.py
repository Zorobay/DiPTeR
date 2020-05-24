import typing


def index_of(a: typing.Iterable, fcond: typing.Callable[..., bool]) -> int:
    """Runs the conditional function 'fcond' on array 'a' and returns the first index where the function evaluates to True.

        :param a: a numpy array
        :param fcond: A boolean function to be evaluated for each element
        :return: The index where 'fcond' first evaluates to True for 'a' or -1 if it was never evaluated as True.
    """

    for i, elem in enumerate(a):
        if fcond(elem):
            return i

    return -1
