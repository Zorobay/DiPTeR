import uuid

from node_graph.graph_element import GraphElement


class Edge(GraphElement):

    def __init__(self, source: 'Socket', destination: 'Socket', container=None):
        super().__init__(container)
        self._src_socket = source
        self._dest_socket = destination

    def get_source(self) -> 'Socket':
        """
        Returns the Socket that is the source of this Edge.
        """
        return self._src_socket

    def get_destination(self) -> 'Socket':
        """
        Returns the Socket that is the destination of this Edge.
        """
        return self._dest_socket

    def connects(self, s1, s2) -> bool:
        """
        Check if two sockets are connected with this Edge.
        :param s1: a Socket
        :param s2: another Socket
        :return: True if this Edge connects 's1' and 's2', otherwise False.
        """
        return (self._src_socket == s1 and self._dest_socket == s2) or (self._src_socket == s2 and self._dest_socket == s1)

    def __contains__(self, item):
        if item == self._src_socket:
            return True
        elif item == self._dest_socket:
            return True

        return False
