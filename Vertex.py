from enum import Enum


class Vertex:

    Infinity = float("inf")

    class EmptySortTypeStack(Exception):
        pass

    class SortType(Enum):
        DIST = 0
        DATA = 1

    sort_key = [SortType.DATA]

    def __init__(self, data=None):
        self.data = data
        self.edge_pairs = dict()
        self.dist = None
        self.prev_in_path = None


    def add_adj(self, vertex, cost=None):
        self.edge_pairs[vertex] = cost

    @classmethod
    def push_sort_type(cls, sort_type):
        cls.sort_key.append(sort_type)

    @classmethod
    def pop_sort_type(cls):
        if len(cls.sort_key) > 1:
            cls.sort_key.pop()
        else:
            raise Vertex.EmptySortTypeStack

    def __lt__(self, other):
        if self.sort_key[-1] is self.SortType.DIST:
            return self.dist < other.dist
        elif self.sort_key[-1] is self.SortType.DATA:
            return self.data < other.data

    def __eq__(self, other):
        if self.sort_key[-1] is self.SortType.DIST:
            return self.dist == other.dist
        elif self.sort_key[-1] is self.SortType.DATA:
            return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def show_adj_list(self):
        print("Adj list for ", self.data,": ", sep="", end="")
        for vertex in self.edge_pairs:
            print(vertex.data, "(", self.edge_pairs[vertex], ")",
                  sep="", end=" ")