import collections
import typing


class FIFOQueue:

    def __init__(self, maxsize: int = 0):
        self._maxsize = maxsize
        self._items = collections.deque()

    def put(self, item: typing.Any):
        self._items.append(item)

    def pop(self) -> typing.Any:
        return self._items.popleft()

    def remove(self, item: typing.Any):
        self._items.remove(item)

    def is_full(self):
        return len(self) >= self._maxsize

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return self._items.__contains__(item)
