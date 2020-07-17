import typing
import uuid


class GraphElement:
    """Basic class for all elements in a node graph."""

    def __init__(self, container: typing.Any = None):
        """
        :param container: A reference to any object that contains this element.
        """""
        self._container = container
        self._id = uuid.uuid4()

    def get_container(self) -> typing.Any:
        return self._container

    def set_container(self, container: typing.Any) -> typing.Any:
        """Sets the container for this GraphElement. If a container has already been previously set, this method does nothing."""
        if self._container is None:
            self._container = container

    def id(self) -> uuid.UUID:
        return self._id

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.id() == self.id()

        return False

    def __hash__(self):
        return self._id.__hash__()

