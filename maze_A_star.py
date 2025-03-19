import time
import random
import collections
from collections import deque
from enum import Enum
from MinHeap import MinHeap
import pandas as pd
import heapq


# Note: You may need to adjust console font to get the maze to look right


class Vertex:

    Infinity = float("inf")

    class EmptySortTypeStack(Exception):
        pass

    class SortType(Enum):
        DIST = 0
        HEUR = 1

    sort_key = [SortType.HEUR]

    def __init__(self, data=None):
        self.data = data
        self.edge_pairs = dict()
        self.dist = None
        self.prev_in_path = None
        self.heuristic = None

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
        elif self.sort_key[-1] is self.SortType.HEUR:
            return self.heuristic < other.heuristic

    def __eq__(self, other):
        if self.sort_key[-1] is self.SortType.DIST:
            return self.dist == other.dist
        elif self.sort_key[-1] is self.SortType.HEUR:
            return self.heuristic == other.heuristic

    def __hash__(self):
        return hash(self.data)

    def show_adj_list(self):
        print("Adj list for ", self.data,": ", sep="", end="")
        for vertex in self.edge_pairs:
            print(vertex.data, "(", self.edge_pairs[vertex], ")",
                  sep="", end=" ")


class Graph:

    def __init__(self):
        self._vertices = {}

    def get_vertex_object(self, vertex_data):
        try:
            vertex = self._vertices[vertex_data]
            return vertex
        except KeyError:
            Vertex.push_sort_type(Vertex.SortType.HEUR)
            new_vertex = Vertex(vertex_data)
            self._vertices[vertex_data] = new_vertex
            Vertex.pop_sort_type()
            return new_vertex

    def add_edge(self, src, dest, cost=None):
        src_vertex = self.get_vertex_object(src)
        dest_vertex = self.get_vertex_object(dest)
        src_vertex.add_adj(dest_vertex, cost)

    def show_adj_table(self):
        print("------------------------ \n")
        for vertex in self._vertices:
            self._vertices[vertex].show_adj_list()

    def clear(self):
        self._vertices = {}

    def dijkstra(self, src):
        Infinity = float("inf")
        src_vertex = self._vertices[src]
        partially_processed = collections.deque()
        for vdata, vobj in self._vertices.items():
            vobj.dist = Infinity
        src_vertex.dist = 0
        partially_processed.append(src_vertex)
        while len(partially_processed) > 0:
            current_vertex = partially_processed.popleft()
            for vobj in current_vertex.edge_pairs:
                if current_vertex.dist + current_vertex.edge_pairs[vobj] < \
                        vobj.dist:
                    vobj.dist = current_vertex.dist + \
                                current_vertex.edge_pairs[vobj]
                    partially_processed.append(vobj)
                    vobj.prev_in_path = current_vertex

    def show_distance_to(self, src):
        self.dijkstra(src)
        print(f"Distance from {src} to:")
        for vdata, vobj in self._vertices.items():
            print(f"{vdata}: {vobj.dist}")

    def show_shortest_path(self, start, stop):

        start_vert = self._vertices[start]
        stop_vert = self._vertices[stop]
        self.dijkstra(start)
        print(
            f"Cost of shortest path from {start} to {stop}: {stop_vert.dist}")
        if stop_vert.dist < float("inf"):
            path_stack = collections.deque()
            current_vert = stop_vert
            while current_vert is not start_vert:
                path_stack.append(current_vert)
                current_vert = current_vert.prev_in_path

            print(start_vert.data, end="")
            while len(path_stack) > 0:
                print(f"--->{path_stack.pop().data}", end="")
        print("")

    def dijkstra_solve(self, start, stop):
        start_vert = self._vertices[start]
        stop_vert = self._vertices[stop]
        self.dijkstra(start)
        res = []
        if stop_vert.dist <= float("inf"):
            current_vert = stop_vert
            while current_vert is not start_vert:
                res.append(current_vert.data)
                current_vert = current_vert.prev_in_path
            res.append(start_vert.data)
        return res[::-1]

    def a_star_solve(self, start, stop):
        start_vert = self._vertices[start]
        stop_vert = self._vertices[stop]
        self.dijkstra(start)
        stop_vert.heuristic = 0
        # queue = MinHeap()
        # queue.insert((stop_vert.dist, stop_vert))
        queue = []
        heapq.heapify(queue)
        heapq.heappush(queue, (stop_vert.dist, stop_vert))
        res = []
        # visit = set()
        # while queue.size > 0:
        while len(queue) > 0:
            val, current_vert = heapq.heappop(queue)
            res.append(current_vert.data)
            if current_vert == start_vert:
                break
            current_vert = current_vert.prev_in_path
            current_vert.heuristic = abs(current_vert.data[0] -
                                         stop_vert.data[0]) + abs(
                current_vert.data[1] - stop_vert.data[1])
            f = current_vert.dist + current_vert.heuristic
            # queue.insert((f, current_vert))
            heapq.heappush(queue, (f, current_vert))
        return res[::-1]


class Node:
    class PathState(Enum):
        CLEAR = 0
        VISITED = 1

    def __init__(self):
        self.next_in_solution = None
        self.prev_in_path = None
        self.state = self.PathState.CLEAR


class Maze:
    class NoSolutionGenerated(Exception):
        pass

    class Method(Enum):
        STACK = 0
        RANDOM = 1
        BIAS = 2

    bias_value = .5
    open_char = " "
    h_block_char = "\u2588"
    v_block_char = "\u2588"
    sol_char = "*"

    def __init__(self, width=10, height=10):
        self._h_walls = [[1] * (height - 1) for _ in range(width)]
        self._v_walls = [[1] * height for _ in range(width - 1)]
        self._grid = [[Node() for _ in range(height)] for _ in range(width)]
        self._width = width
        self._height = height
        self._start = None
        self._end = None
        self._solution_path = None

    @property
    def start(self):
        return self._start, 0

    @property
    def end(self):
        return self._end, self._height - 1

    def _build_solution_path(self):

        self._solution_path = [(self._end, self._height - 1)]
        while self._solution_path[-1] != (self._start, 0):
            curr_x, curr_y = self._solution_path[-1]
            next_pos = self._grid[curr_x][curr_y].prev_in_path
            self._solution_path.append(next_pos)
        self._solution_path.reverse()

    @property
    def solution_path(self):
        if not self._solution_path:
            raise Maze.NoSolutionGenerated
        return self._solution_path

    def print_maze(self, with_solution=False):
        top_line = self.h_block_char
        for pos in range(self._width):
            top_line += self.open_char if pos == self._end \
                else self.h_block_char
            top_line += self.h_block_char
        print(top_line)
        for y in range(self._height - 1, -1, -1):
            # Print horizontal walls (except the first time)
            if y < self._height - 1:
                row_line = self.h_block_char
                for x in range(0, self._width):
                    row_line += self.h_block_char if self._h_walls[x][y] \
                        else self.open_char
                    row_line += self.h_block_char
                print(row_line)

            # Print vertical walls and path
            row_line = self.v_block_char
            for x in range(0, self._width):
                if with_solution and (x, y) in self._solution_path:
                    row_line += self.sol_char
                else:
                    row_line += self.open_char
                if x < self._width - 1:
                    row_line += self.v_block_char if self._v_walls[x][y] \
                        else self.open_char
            row_line += self.v_block_char
            print(row_line)
        bot_line = self.h_block_char
        for pos in range(self._width):
            bot_line += self.open_char if pos == self._start \
                else self.h_block_char
            bot_line += self.h_block_char
        print(bot_line)

    def valid_position(self, x, y):
        if x < 0 or x >= self._width or y < 0 or y >= self._height:
            return False
        return True

    def is_wall(self, curr_x, curr_y, prop_x, prop_y):
        if prop_x < curr_x:
            if self._v_walls[prop_x][prop_y]:
                return True
        if curr_x < prop_x:
            if self._v_walls[curr_x][prop_y]:
                return True
        if prop_y < curr_y:
            if self._h_walls[prop_x][prop_y]:
                return True
        if curr_y < prop_y:
            if self._h_walls[prop_x][curr_y]:
                return True
        return False

    def break_wall(self, curr_x, curr_y, prop_x, prop_y):
        if prop_x < curr_x:
            self._v_walls[prop_x][prop_y] = 0
        if curr_x < prop_x:
            self._v_walls[curr_x][prop_y] = 0
        if prop_y < curr_y:
            self._h_walls[prop_x][prop_y] = 0
        if curr_y < prop_y:
            self._h_walls[prop_x][curr_y] = 0

    def create_graph(self):
        my_graph = Graph()
        for x in range(self._width - 1, -1, -1):
            for y in range(self._height - 1, -1, -1):
                if self.valid_position(x + 1, y) and not \
                        self.is_wall(x, y, x + 1, y):
                    my_graph.add_edge((x, y), (x + 1, y), 1)
                    my_graph.add_edge((x + 1, y), (x, y), 1)

                if self.valid_position(x, y + 1) and not \
                        self.is_wall(x, y, x, y + 1):
                    my_graph.add_edge((x, y), (x, y + 1), 1)
                    my_graph.add_edge((x, y + 1), (x, y), 1)

        return my_graph

    def create_solution_path(self, method=Method.RANDOM):

        def random_move(curr_pos):
            next_pos = [*curr_pos]
            move = random.randint(0, 3)
            if move == 0:
                next_pos[0] += 1
            elif move == 1:
                next_pos[0] -= 1
            elif move == 2:
                next_pos[1] += 1
            else:
                next_pos[1] -= 1
            return tuple(next_pos)

        def paths_remain(curr_x, curr_y):
            # Top row always has a path
            if curr_y == self._height - 1:
                return True
            for choice in [(curr_x + i, curr_y + j) for i, j in
                           [[-1, 0], [0, -1], [1, 0], [0, 1]]]:
                if choice in unvisited_set:
                    return True
            return False

        self._start = self._width // 2
        if method == self.Method.STACK:
            backtrack_stack = deque()
        else:
            backtrack_stack = list()

        backtrack_stack.append((self._start, 0))
        unvisited_set = {(x, y) for x in range(self._width)
                         for y in range(self._height)}
        while True:
            if method == self.Method.RANDOM:
                current_pos = random.choice(backtrack_stack)
                backtrack_stack.remove(current_pos)
            elif method == self.Method.BIAS:
                if random.random() > self.bias_value:
                    current_pos = random.choice(backtrack_stack)
                    backtrack_stack.remove(current_pos)
                else:
                    current_pos = backtrack_stack[0]
                    backtrack_stack.remove(current_pos)
            else:
                current_pos = backtrack_stack.pop()
            # It's possible that this node has been boxed in
            if not paths_remain(*current_pos):
                continue
            unvisited_set.discard(current_pos)
            proposed_next = random_move(current_pos)

            if proposed_next[1] == self._height:
                self._end = current_pos[0]
                self._grid[current_pos[0]][current_pos[1]].next_in_solution = \
                    proposed_next
                break

            if tuple(proposed_next) not in unvisited_set:
                backtrack_stack.append(current_pos)
                continue
            else:
                # We are forging new ground, tear down the wall
                self.break_wall(*current_pos, *proposed_next)
                self._grid[current_pos[0]][current_pos[1]].next_in_solution = \
                    proposed_next
                self._grid[proposed_next[0]][proposed_next[1]].prev_in_path = \
                    current_pos
                if paths_remain(*current_pos):
                    backtrack_stack.append(current_pos)
                backtrack_stack.append(proposed_next)
                unvisited_set.discard(proposed_next)

        # Now fill in the rest of the routes
        while unvisited_set:
            iterable_version = list(unvisited_set)
            for node in iterable_version:
                proposed_next = random_move(node)
                if self.valid_position(*proposed_next) and \
                        proposed_next not in unvisited_set:
                    self.break_wall(*node, *proposed_next)
                    unvisited_set.discard(node)

        self._build_solution_path()


def main():
    my_maze = Maze(10, 10)
    my_maze.create_solution_path()
    my_maze.print_maze()


def create_and_solve():
    d_stack_times = []
    d_random_times = []
    d_bias_times = []
    a_stack_times = []
    a_random_times = []
    a_bias_times = []
    sizes = [5, 10, 20, 40, 80, 160]
    # sizes = [5]
    for size in sizes:
        for method in Maze.Method:
            print("Maze Size", size)
            d_total_time = 0
            a_total_time = 0
            trials = 20
            # print to check whether the code working properly
            # trials = 2

            for a in range(trials):
                random.seed(a)
                my_maze = Maze(size, size + 5)
                my_maze.create_solution_path(method=method)
                # Uncomment to print the maze and solution path
                # my_maze.print_maze(True)
                # print()
                d_start = time.perf_counter()
                maze_graph = my_maze.create_graph()
                d_path = maze_graph.dijkstra_solve(my_maze.start, my_maze.end)
                # Uncomment to see the actual and proposed solution paths
                # print(d_path, my_maze.solution_path)
                if d_path != my_maze.solution_path:
                    print("Error: Proposed Dijkstra solution is invalid")
                d_end = time.perf_counter()
                d_total_time += (d_end - d_start)
                # Uncomment for A* graph testing
                a_start = time.perf_counter()
                random.seed(a)
                maze_graph = my_maze.create_graph()
                a_path = maze_graph.a_star_solve(my_maze.start, my_maze.end)
                # Uncomment to see the actual and proposed solution paths
                # print("d_path", d_path)
                # print("my_maze.solution_path", my_maze.solution_path)
                # print("a_path", a_path)
                if a_path != my_maze.solution_path:
                    print("Error: Proposed A* solution is invalid")
                a_end = time.perf_counter()
                a_total_time += (a_end - a_start)
            print(f"Dijkstra took {d_total_time / trials * 1000:.3f} "
                  f"ms with {method}")
            if method is Maze.Method.STACK:
                d_stack_times.append(
                    f"{round((d_total_time / trials * 1000), 3)} ms")
            if method is Maze.Method.RANDOM:
                d_random_times.append(
                    f"{round((d_total_time / trials * 1000), 3)} ms")
            if method is Maze.Method.BIAS:
                d_bias_times.append(
                    f"{round((d_total_time / trials * 1000), 3)} ms")
            # Uncomment for A* results
            print(f"A* took {a_total_time / trials * 1000:.3f} "
                  f"ms with {method}")
            if method is Maze.Method.STACK:
                a_stack_times.append(
                    f"{round((a_total_time / trials * 1000), 3)} ms")
            if method is Maze.Method.RANDOM:
                a_random_times.append(
                    f"{round((a_total_time / trials * 1000), 3)} ms")
            if method is Maze.Method.BIAS:
                a_bias_times.append(
                    f"{round((a_total_time / trials * 1000), 3)} ms")

    dataframe = pd.DataFrame()
    dataframe["Maze Size"] = sizes
    dataframe["d_STACK"] = d_stack_times
    dataframe["a_STACK"] = a_stack_times
    dataframe["d_RANDOM"] = d_random_times
    dataframe["a_RANDOM"] = a_random_times
    dataframe["d_BIAS"] = d_bias_times
    dataframe["a_BIAS"] = a_bias_times
    print(dataframe)
    dataframe.to_csv("1modified_a_d_Maze_Times.csv", index=False)


if __name__ == "__main__":
    create_and_solve()



"""
Maze Size 5
█ █████████
█* *█* *  █
█ █ █ █ ███
█ █* *█*  █
█ █████ ███
█   █* *  █
█████ ███ █
█* * *  █ █
█ ███ ███ █
█*█     █ █
█ ███ █████
█* *█   █ █
█ █ █████ █
█ █* * *█ █
███ ███ █ █
█   █  *  █
███████ ███
█     █*  █
█████ █ ███
█    * *  █
█████ █████
d_path [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (3, 9), (2, 9), (2, 8), (1, 8), (1, 9), (0, 9)]
my_maze.solution_path [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (3, 9), (2, 9), (2, 8), (1, 8), (1, 9), (0, 9)]
a_path [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (3, 9), (2, 9), (2, 8), (1, 8), (1, 9), (0, 9)]
█████████ █
█     █  *█
███ █████ █
█    * *█*█
█ ███ █ █ █
█ █  *█*█*█
█████ █ █ █
█   █*█* *█
███ █ █████
█* *█* * *█
█ █ █████ █
█*█* *█* *█
█ ███ █ █ █
█* *█* *█ █
█ █ ███████
█ █* * * *█
█ ███████ █
█ █* * *█*█
█ █ ███ █ █
█ █* *█* *█
█████ █████
d_path [(2, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0), (4, 1), (4, 2), (3, 2), (2, 2), (1, 2), (1, 3), (0, 3), (0, 4), (0, 5), (1, 5), (1, 4), (2, 4), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (3, 5), (2, 5), (2, 6), (2, 7), (2, 8), (3, 8), (3, 7), (3, 6), (4, 6), (4, 7), (4, 8), (4, 9)]
my_maze.solution_path [(2, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0), (4, 1), (4, 2), (3, 2), (2, 2), (1, 2), (1, 3), (0, 3), (0, 4), (0, 5), (1, 5), (1, 4), (2, 4), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (3, 5), (2, 5), (2, 6), (2, 7), (2, 8), (3, 8), (3, 7), (3, 6), (4, 6), (4, 7), (4, 8), (4, 9)]
a_path [(2, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 0), (4, 0), (4, 1), (4, 2), (3, 2), (2, 2), (1, 2), (1, 3), (0, 3), (0, 4), (0, 5), (1, 5), (1, 4), (2, 4), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (3, 5), (2, 5), (2, 6), (2, 7), (2, 8), (3, 8), (3, 7), (3, 6), (4, 6), (4, 7), (4, 8), (4, 9)]
Dijkstra took 1.744 ms with Method.STACK
A* took 1.925 ms with Method.STACK
Maze Size 5
█████ █████
█    *    █
█████ █████
█    *    █
█████ █████
█ █  *█   █
█ ███ █ ███
█   █* *  █
███ ███ ███
█   █* *█ █
███ █ ███ █
█  * *█   █
███ ███ ███
█ █*█     █
█ █ ███ ███
█* *█     █
█ █████ ███
█* * *    █
███ █ ███ █
█   █*  █ █
█████ █████
d_path [(2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 5), (3, 6), (2, 6), (2, 7), (2, 8), (2, 9)]
my_maze.solution_path [(2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 5), (3, 6), (2, 6), (2, 7), (2, 8), (2, 9)]
a_path [(2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 5), (3, 6), (2, 6), (2, 7), (2, 8), (2, 9)]
███ ███████
█  *█ █   █
███ █ ███ █
█ █*█ █   █
█ █ █ █ ███
█ █*█ █   █
█ █ █ █ ███
█  * *█   █
█████ █ ███
█    *█   █
█ ███ █ ███
█ █  *    █
█████ █████
█   █*    █
███ █ █████
█    *█   █
█████ █ ███
█   █* *█ █
███ ███ █ █
█    * *  █
█████ █████
d_path [(2, 0), (3, 0), (3, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (1, 6), (1, 7), (1, 8), (1, 9)]
my_maze.solution_path [(2, 0), (3, 0), (3, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (1, 6), (1, 7), (1, 8), (1, 9)]
a_path [(2, 0), (3, 0), (3, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (1, 6), (1, 7), (1, 8), (1, 9)]
Dijkstra took 1.021 ms with Method.RANDOM
A* took 1.402 ms with Method.RANDOM
Maze Size 5
███ ███████
█* *█ █ █ █
█ ███ █ █ █
█*█       █
█ █████ ███
█* *█     █
███ █ ███ █
█ █*█ █ █ █
█ █ █ █ ███
█  * *█ █ █
█████ █ █ █
█    *█   █
█████ ███ █
█   █* *█ █
█ █████ █ █
█     █*█ █
█ █████ █ █
█      *  █
███████ █ █
█    * *█ █
█████ █████
d_path [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (1, 6), (1, 7), (0, 7), (0, 8), (0, 9), (1, 9)]
my_maze.solution_path [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (1, 6), (1, 7), (0, 7), (0, 8), (0, 9), (1, 9)]
a_path [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (1, 6), (1, 7), (0, 7), (0, 8), (0, 9), (1, 9)]
███ ███████
█  *      █
███ ███████
█ █*█ █   █
█ █ █ █ ███
█  *█   █ █
███ █ ███ █
█* *  █ █ █
█ █ ███ █ █
█*█   █ █ █
█ █████ █ █
█* *█ █ █ █
███ █ █ █ █
█  *█   █ █
███ █ ███ █
█ █* *  █ █
█ ███ ███ █
█ █  *█ █ █
█ ███ █ █ █
█    *    █
█████ █████
d_path [(2, 0), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (1, 6), (1, 7), (1, 8), (1, 9)]
my_maze.solution_path [(2, 0), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (1, 6), (1, 7), (1, 8), (1, 9)]
a_path [(2, 0), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (1, 6), (1, 7), (1, 8), (1, 9)]
Dijkstra took 0.714 ms with Method.BIAS
A* took 0.872 ms with Method.BIAS
Maze Size 10
███ █████████████████
█ █*  █ █ █ █ █ █   █
█ █ ███ █ █ █ █ █ ███
█* *█ █     █   █   █
█ ███ █ ███████ █ ███
█*█* *    █       █ █
█ █ █ █████ ███████ █
█* *█*█ █       █   █
█████ █ ███ █████ ███
█* * *█   █ █ █ █   █
█ █████ ███ █ █ █ ███
█* * * *█* *      █ █
█ █████ █ █ █ █████ █
█     █* *█*█ █ █   █
█████ █████ ███ █ ███
█ █     █* *      █ █
█ █ █████ ███████ █ █
█ █ █* * *      █   █
█ █ █ ███ ███ ███████
█   █* *█ █ █ █     █
█ █████ ███ ███ █ █ █
█     █* *    █ █ █ █
█████ ███ █████ █████
█     █ █* * *█     █
█ █████ █████ █ █████
█ █         █*  █   █
█ ███████ █ █ ███ ███
█ █   █   █ █*      █
█ █ █ █ █ ███ ███ █ █
█   █   █ █* *  █ █ █
███████████ █████████
d_path [(5, 0), (6, 0), (6, 1), (6, 2), (6, 3), (5, 3), (4, 3), (4, 4), (3, 4), (3, 5), (2, 5), (2, 6), (3, 6), (4, 6), (4, 7), (5, 7), (5, 8), (5, 9), (4, 9), (4, 8), (3, 8), (3, 9), (2, 9), (1, 9), (0, 9), (0, 10), (1, 10), (2, 10), (2, 11), (2, 12), (1, 12), (1, 11), (0, 11), (0, 12), (0, 13), (1, 13), (1, 14)]
my_maze.solution_path [(5, 0), (6, 0), (6, 1), (6, 2), (6, 3), (5, 3), (4, 3), (4, 4), (3, 4), (3, 5), (2, 5), (2, 6), (3, 6), (4, 6), (4, 7), (5, 7), (5, 8), (5, 9), (4, 9), (4, 8), (3, 8), (3, 9), (2, 9), (1, 9), (0, 9), (0, 10), (1, 10), (2, 10), (2, 11), (2, 12), (1, 12), (1, 11), (0, 11), (0, 12), (0, 13), (1, 13), (1, 14)]
a_path [(5, 0), (6, 0), (6, 1), (6, 2), (6, 3), (5, 3), (4, 3), (4, 4), (3, 4), (3, 5), (2, 5), (2, 6), (3, 6), (4, 6), (4, 7), (5, 7), (5, 8), (5, 9), (4, 9), (4, 8), (3, 8), (3, 9), (2, 9), (1, 9), (0, 9), (0, 10), (1, 10), (2, 10), (2, 11), (2, 12), (1, 12), (1, 11), (0, 11), (0, 12), (0, 13), (1, 13), (1, 14)]
█████ ███████████████
█* * *█   █   █ █ █ █
█ █████ ███ ███ █ █ █
█* * *    █   █   █ █
█ ███ █████ █████ █ █
█ █* *  █     █     █
███ █ ███ █████████ █
█  *█ █   █       █ █
███ ███ █████ █████ █
█  *  █   █   █  * *█
███ █████ █ █████ █ █
█  *█ █* * * * * *█*█
███ █ █ ███████████ █
█  *  █*█       █* *█
███ █ █ ███ ███ █ █ █
█  *█ █* *█ █   █*█ █
███ █████ █ █ █ █ ███
█* *█* * *█ █ █ █* *█
█ █ █ ███████ █ ███ █
█*█ █* *█     █   █*█
█ █ ███ █ █ ███████ █
█*█ █* *█ █   █* * *█
█ ███ ███ ███ █ █████
█* *█*█   █   █* * *█
███ █ █████ █ █████ █
█  *█*█     █ █* * *█
███ █ █ ███████ ███ █
█* *█* *█* * *█* *█ █
█ █████ █ ███ ███ █ █
█* * * *█* *█* * *█ █
███████████ █████████
d_path [(5, 0), (4, 0), (4, 1), (5, 1), (6, 1), (6, 0), (7, 0), (8, 0), (8, 1), (7, 1), (7, 2), (8, 2), (9, 2), (9, 3), (8, 3), (7, 3), (7, 4), (8, 4), (9, 4), (9, 5), (9, 6), (8, 6), (8, 7), (8, 8), (9, 8), (9, 9), (9, 10), (8, 10), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (3, 9), (3, 8), (3, 7), (4, 7), (4, 6), (3, 6), (2, 6), (2, 5), (3, 5), (3, 4), (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 12), (2, 13), (1, 13), (0, 13), (0, 14), (1, 14), (2, 14)]
my_maze.solution_path [(5, 0), (4, 0), (4, 1), (5, 1), (6, 1), (6, 0), (7, 0), (8, 0), (8, 1), (7, 1), (7, 2), (8, 2), (9, 2), (9, 3), (8, 3), (7, 3), (7, 4), (8, 4), (9, 4), (9, 5), (9, 6), (8, 6), (8, 7), (8, 8), (9, 8), (9, 9), (9, 10), (8, 10), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (3, 9), (3, 8), (3, 7), (4, 7), (4, 6), (3, 6), (2, 6), (2, 5), (3, 5), (3, 4), (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 12), (2, 13), (1, 13), (0, 13), (0, 14), (1, 14), (2, 14)]
a_path [(5, 0), (4, 0), (4, 1), (5, 1), (6, 1), (6, 0), (7, 0), (8, 0), (8, 1), (7, 1), (7, 2), (8, 2), (9, 2), (9, 3), (8, 3), (7, 3), (7, 4), (8, 4), (9, 4), (9, 5), (9, 6), (8, 6), (8, 7), (8, 8), (9, 8), (9, 9), (9, 10), (8, 10), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (3, 9), (3, 8), (3, 7), (4, 7), (4, 6), (3, 6), (2, 6), (2, 5), (3, 5), (3, 4), (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 4), (0, 5), (0, 6), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 12), (2, 13), (1, 13), (0, 13), (0, 14), (1, 14), (2, 14)]
Dijkstra took 2.034 ms with Method.STACK
A* took 2.471 ms with Method.STACK
Maze Size 10
█ ███████████████████
█*  █ █     █ █ █   █
█ ███ █ █████ █ █ ███
█*█   █ █ █   █ █ █ █
█ ███ █ █ █ ███ █ █ █
█*      █ █ █       █
█ ███████ █ █ ███ ███
█*█   █ █ █   █ █ █ █
█ ███ █ █ █ ███ ███ █
█*█   █         █   █
█ █ █ █ █████████ ███
█*█ █       █       █
█ ███ █████ ███ █████
█*      █ █ █   █ █ █
█ █ █████ █████ █ █ █
█*█ █ █ █ █ █   █ █ █
█ ███ █ █ █ ███ █ █ █
█*      █ █     █   █
█ ███████ █ █████ ███
█*█ █   █         █ █
█ █ █ ███ █████████ █
█* *  █     █   █   █
███ ███ ███████ █ ███
█  *█ █   █ █       █
███ █ █ ███ █ █████ █
█  * * *  █   █   █ █
███ █ █ █████ █ █████
█   █ █* * *        █
█ █ ███ ███ ███ █████
█ █ █     █*  █     █
███████████ █████████
d_path [(5, 0), (5, 1), (4, 1), (3, 1), (3, 2), (2, 2), (1, 2), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14)]
my_maze.solution_path [(5, 0), (5, 1), (4, 1), (3, 1), (3, 2), (2, 2), (1, 2), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14)]
a_path [(5, 0), (5, 1), (4, 1), (3, 1), (3, 2), (2, 2), (1, 2), (1, 3), (1, 4), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14)]
█ ███████████████████
█* *█ █   █   █ █   █
███ █ █ █████ █ █ ███
█ █*█ █ █ █       █ █
█ █ █ █ █ █████ ███ █
█  * * * *█ █     █ █
█████████ █ ███ ███ █
█   █   █*█     █ █ █
███ ███ █ █ █ ███ █ █
█ █   █  *  █   █   █
█ █ █████ █ ███████ █
█   █   █*█ █       █
█ █████ █ █████████ █
█ █      *█ █       █
█ ███████ █ ███ █████
█   █ █  *█ █       █
█ ███ ███ █ ███ █ █ █
█ █ █    *    █ █ █ █
█ █ █████ ███ █ █████
█   █ █  *█ █   █ █ █
███ █ ███ █ █████ █ █
█   █   █*  █ █     █
█ █ ███ █ ███ █ █████
█ █      *  █     █ █
█████████ ███ █████ █
█ █ █ █ █* *█ █   █ █
█ █ █ █ ███ █ ███ █ █
█          *  █     █
███ ███ ███ ███ █████
█   █     █*        █
███████████ █████████
d_path [(5, 0), (5, 1), (5, 2), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (3, 12), (2, 12), (1, 12), (1, 13), (1, 14), (0, 14)]
my_maze.solution_path [(5, 0), (5, 1), (5, 2), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (3, 12), (2, 12), (1, 12), (1, 13), (1, 14), (0, 14)]
a_path [(5, 0), (5, 1), (5, 2), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (3, 12), (2, 12), (1, 12), (1, 13), (1, 14), (0, 14)]
Dijkstra took 2.028 ms with Method.RANDOM
A* took 2.534 ms with Method.RANDOM
Maze Size 10
███████████ █████████
█ █   █ █  *█ █ █   █
█ █ ███ ███ █ █ █ ███
█ █ █     █*█   █   █
█ █ ███ ███ █ ███ ███
█ █   █ █* *█ █ █ █ █
█ ███ █ █ ███ █ █ █ █
█        *  █ █ █ █ █
█████████ ███ █ █ █ █
█ █ █    *    █   █ █
█ █ █████ ███████ █ █
█        * *█ █   █ █
███████████ █ █ ███ █
█   █ █ █  *      █ █
███ █ █ ███ ███████ █
█ █ █ █   █*█ █ █   █
█ █ █ ███ █ █ █ █ ███
█   █ █ █  * *█ █   █
███ █ █ █████ █ █ ███
█ █   █   █  *    █ █
█ ███ ███ ███ █████ █
█   █ █ █    *█     █
███ █ █ █████ █ █████
█     █ █ █ █*  █ █ █
█████ █ █ █ █ ███ █ █
█ █ █   █ █* *█ █   █
█ █ ███ █ █ ███ █ ███
█ █ █      * *█   █ █
█ █ █████████ █ ███ █
█          * *      █
███████████ █████████
d_path [(5, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 6), (5, 7), (5, 8), (5, 9), (4, 9), (4, 10), (4, 11), (4, 12), (5, 12), (5, 13), (5, 14)]
my_maze.solution_path [(5, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 6), (5, 7), (5, 8), (5, 9), (4, 9), (4, 10), (4, 11), (4, 12), (5, 12), (5, 13), (5, 14)]
a_path [(5, 0), (6, 0), (6, 1), (5, 1), (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 6), (5, 7), (5, 8), (5, 9), (4, 9), (4, 10), (4, 11), (4, 12), (5, 12), (5, 13), (5, 14)]
█████████████ ███████
█   █     █ █*  █   █
█ █ █████ █ █ █ █ ███
█ █       █ █*█     █
███████ █ █ █ ███ ███
█     █ █    *█ █ █ █
███ █████████ █ ███ █
█   █   █    *█   █ █
███ █ █ █████ █ ███ █
█ █   █ █ █  *      █
█ ███ ███ ███ █ █████
█   █       █*█ █   █
█ ███████ ███ ███ ███
█ █     █   █*      █
█ █████ █ ███ ███████
█ █ █   █ █* *  █ █ █
█ █ █ █ █ █ █████ █ █
█     █   █*█ █ █   █
█ ███████ █ █ █ █ ███
█   █      *█ █   █ █
███████████ █ ███ █ █
█ █ █ █ █ █*█ █ █ █ █
█ █ █ █ █ █ █ █ █ █ █
█ █ █   █ █*  █   █ █
█ █ ███ █ █ ███ ███ █
█ █       █*  █   █ █
█ ███████ █ █████ █ █
█          *█ █     █
███ █ █ █ █ █ █████ █
█   █ █ █ █*        █
███████████ █████████
d_path [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14)]
my_maze.solution_path [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14)]
a_path [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14)]
Dijkstra took 2.406 ms with Method.BIAS
A* took 2.309 ms with Method.BIAS
Maze Size 20
███ █████████████████████████████████████
█* *█ █ █   █ █ █   █   █ █ █ █         █
█ ███ █ ███ █ █ ███ █ ███ █ █ █ █████████
█*    █ █   █   █ █     █   █ █         █
█ █████ ███ █ ███ ███ ███ ███ █ ███ █████
█*  █   █     █ █   █ █ █         █   █ █
█ ███ █████ ███ ███ █ █ █████ ███ █████ █
█*  █* * *█   █   █ █ █ █       █   █ █ █
█ ███ ███ █ █████ █ █ █ ███ █ █ █████ █ █
█* * *█* *█* * * *█   █ █* *█ █ █ █ █   █
█ █ █ █ █ █ █████ █ ███ █ █ █ ███ █ █ ███
█ █ █ █*█ █*█   █* *█* * *█*█           █
█ █████ █ █ █ █ ███ █ █████ █████████████
█ █    *█ █*█ █   █* *█* *█*  █   █ █ █ █
█ █████ ███ ███ █ █████ █ █ ███ ███ █ █ █
█ █ █  * *█* *█ █     █*█* *█           █
███ █████ ███ █ █████ █ █████ ███████████
█   █ █  * *█*█   █ █ █* * * * *█ █ █   █
█ ███ ███ █ █ ███ █ █ █████████ █ █ █ ███
█   █   █ █* *    █           █*█ █     █
███ ███ ███████████████████████ █ █ █ ███
█    * * * * * * * *█* * * * *█* *  █   █
█ █ █ ███████ ███ █ █ █ █ ███ █ █ ███████
█ █ █* *█   █   █ █* *█ █ █* *█ █*█ █   █
█ █████ █ █████ █ █ ███ ███ █████ █ █ ███
█ █  * *  █   █ █ █   █   █* * * *      █
█ ███ █████ ███████████ █ ███ █ █ █ █ ███
█ █  * * *  █         █ █ █   █ █ █ █   █
███ █████ ███ █████████ ███ ███ █ █ ███ █
█   █  * *  █ █ █   █ █ █   █   █ █   █ █
███████ █████ █ ███ █ ███ █████ █ █████ █
█ █  * *█* * * * *█* *  █ █ █ █ █   █   █
█ ███ ███ ███████ █ █ █████ █ █ █ █ ███ █
█    *  █* * *█ █* *█*  █ █     █ █   █ █
█ ███ ███████ █ █████ ███ █████ █ ███ ███
█ █  * * *█* *    █* *        █ █   █   █
█ ███████ █ ███████ █████ █████ █ ███ ███
█ █  * * *█* *█* * *    █     █ █   █   █
█████ ███████ █ █████████████ █ █ █ █ █ █
█ █ █* * * * *█* *      █   █ █ █ █ █ █ █
█ █ █ █ █ █ █ ███ ███████ █████████ █████
█     █ █ █ █   █* *  █ █   █     █     █
███ █ ███ █ █ █ ███ ███ █ █████ █████████
█   █   █ █ █ █ █  * * *  █ █ █ █ █ █   █
█ █ █ ███ █ █ █ ███ ███ ███ █ █ █ █ █ ███
█ █ █   █ █ █ █ █   █  *█ █       █     █
█ ███ ███████ █ ███████ █ █ ███ ███ ███ █
█ █     █     █   █   █*      █       █ █
█ ███ ███████ █ █████ █ █ █████████ ███ █
█ █   █       █   █  * *█         █   █ █
█████████████████████ ███████████████████
d_path [(10, 0), (11, 0), (11, 1), (11, 2), (11, 3), (10, 3), (9, 3), (9, 4), (8, 4), (8, 5), (7, 5), (7, 6), (8, 6), (9, 6), (9, 7), (10, 7), (10, 8), (10, 9), (9, 9), (9, 8), (8, 8), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (4, 8), (5, 8), (6, 8), (6, 7), (5, 7), (5, 6), (6, 6), (6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (2, 6), (3, 6), (4, 6), (4, 7), (3, 7), (2, 7), (2, 8), (2, 9), (3, 9), (3, 10), (4, 10), (4, 11), (3, 11), (2, 11), (2, 12), (3, 12), (3, 13), (2, 13), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (9, 13), (10, 13), (10, 14), (11, 14), (12, 14), (13, 14), (14, 14), (14, 13), (13, 13), (13, 12), (14, 12), (15, 12), (16, 12), (16, 13), (16, 14), (15, 14), (15, 15), (15, 16), (14, 16), (13, 16), (12, 16), (11, 16), (11, 17), (11, 18), (12, 18), (12, 17), (13, 17), (13, 18), (13, 19), (13, 20), (12, 20), (12, 19), (11, 19), (10, 19), (10, 18), (9, 18), (9, 19), (8, 19), (8, 20), (7, 20), (6, 20), (5, 20), (5, 19), (5, 18), (5, 17), (6, 17), (6, 16), (6, 15), (5, 15), (5, 16), (4, 16), (4, 17), (3, 17), (3, 18), (3, 19), (3, 20), (4, 20), (4, 21), (3, 21), (2, 21), (2, 20), (1, 20), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (1, 24)]
my_maze.solution_path [(10, 0), (11, 0), (11, 1), (11, 2), (11, 3), (10, 3), (9, 3), (9, 4), (8, 4), (8, 5), (7, 5), (7, 6), (8, 6), (9, 6), (9, 7), (10, 7), (10, 8), (10, 9), (9, 9), (9, 8), (8, 8), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (4, 8), (5, 8), (6, 8), (6, 7), (5, 7), (5, 6), (6, 6), (6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (2, 6), (3, 6), (4, 6), (4, 7), (3, 7), (2, 7), (2, 8), (2, 9), (3, 9), (3, 10), (4, 10), (4, 11), (3, 11), (2, 11), (2, 12), (3, 12), (3, 13), (2, 13), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (9, 13), (10, 13), (10, 14), (11, 14), (12, 14), (13, 14), (14, 14), (14, 13), (13, 13), (13, 12), (14, 12), (15, 12), (16, 12), (16, 13), (16, 14), (15, 14), (15, 15), (15, 16), (14, 16), (13, 16), (12, 16), (11, 16), (11, 17), (11, 18), (12, 18), (12, 17), (13, 17), (13, 18), (13, 19), (13, 20), (12, 20), (12, 19), (11, 19), (10, 19), (10, 18), (9, 18), (9, 19), (8, 19), (8, 20), (7, 20), (6, 20), (5, 20), (5, 19), (5, 18), (5, 17), (6, 17), (6, 16), (6, 15), (5, 15), (5, 16), (4, 16), (4, 17), (3, 17), (3, 18), (3, 19), (3, 20), (4, 20), (4, 21), (3, 21), (2, 21), (2, 20), (1, 20), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (1, 24)]
a_path [(10, 0), (11, 0), (11, 1), (11, 2), (11, 3), (10, 3), (9, 3), (9, 4), (8, 4), (8, 5), (7, 5), (7, 6), (8, 6), (9, 6), (9, 7), (10, 7), (10, 8), (10, 9), (9, 9), (9, 8), (8, 8), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (4, 8), (5, 8), (6, 8), (6, 7), (5, 7), (5, 6), (6, 6), (6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (2, 6), (3, 6), (4, 6), (4, 7), (3, 7), (2, 7), (2, 8), (2, 9), (3, 9), (3, 10), (4, 10), (4, 11), (3, 11), (2, 11), (2, 12), (3, 12), (3, 13), (2, 13), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (9, 13), (10, 13), (10, 14), (11, 14), (12, 14), (13, 14), (14, 14), (14, 13), (13, 13), (13, 12), (14, 12), (15, 12), (16, 12), (16, 13), (16, 14), (15, 14), (15, 15), (15, 16), (14, 16), (13, 16), (12, 16), (11, 16), (11, 17), (11, 18), (12, 18), (12, 17), (13, 17), (13, 18), (13, 19), (13, 20), (12, 20), (12, 19), (11, 19), (10, 19), (10, 18), (9, 18), (9, 19), (8, 19), (8, 20), (7, 20), (6, 20), (5, 20), (5, 19), (5, 18), (5, 17), (6, 17), (6, 16), (6, 15), (5, 15), (5, 16), (4, 16), (4, 17), (3, 17), (3, 18), (3, 19), (3, 20), (4, 20), (4, 21), (3, 21), (2, 21), (2, 20), (1, 20), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (1, 24)]
█████ ███████████████████████████████████
█   █*█   █  * *  █   █ █ █ █     █ █   █
███ █ █ █████ █ ███ ███ █ █ ███ ███ █ █ █
█* * *█* *█  *█*            █     █   █ █
█ █████ █ ███ █ █ ███████████████ █ ███ █
█*█* *█*█* * *█*█   █       █   █ █   █ █
█ █ █ █ ███████ █████ █ ███████ █ █ █████
█* *█*█*█     █*█ █ █ █ █     █         █
█ ███ █ █ ███ █ █ █ █ ███ ███████ █ █ ███
█ █ █* *█ █ █ █* * *█* *█   █ █ █ █ █   █
█ █ █████ █ █ █████ █ █ █ ███ █ █ █████ █
█ █     █   █     █* *█* * *  █       █ █
█ █ █ █ ███████ █ █████████ ███ █████ ███
█ █ █ █         █   █     █* * * *█ █ █ █
█ █ █ █████████████ █ ███ █████ █ █ ███ █
█ █ █     █     █   █ █   █* *█ █*█ █ █ █
█ ███████ ███ █ █ ███ █ ███ █ ███ █ █ █ █
█   █   █   █ █   █   █ █* *█* *█* *█ █ █
███ █ █ ███ █ █████████ █ █████ ███ █ █ █
█ █ █ █     █ █   █   █ █* * *█* * *█   █
█ █ █ ███████ █ █ █ █ █ █████ ███████ ███
█ █   █     █   █   █     █ █* * *█     █
█ █████████ █████████████ █ █████ ███ ███
█ █             █ █       █     █* * * *█
█ █ █ █████████ █ █ ███████ ███ ███████ █
█   █   █     █   █   █   █   █ █     █*█
███████ █ ███ ███ ███ █ █████ █ █ █ █ █ █
█       █   █       █ █       █   █ █ █*█
█ ███████████████████ █████ █████ █ ███ █
█       █     █   █   █   █ █   █ █ █* *█
█ █████ █ ███ █ █ █ ███ █ █ █ █ ███ █ ███
█ █   █     █   █       █ █ █ █     █* *█
█ █ █ ███████ ███████████ ███ █████ ███ █
█   █ █ █* *█ █* * * *█     █ █       █*█
█████ █ █ █ ███ █████ █████ █ █████████ █
█   █   █*█* * *█   █* * *█   █* * * * *█
█ █████ █ ███████ ███████ █████ ███ █████
█ █     █*█     █        * * * *█ █ █* *█
█ █ █████ █████ █ ███████████████ ███ █ █
█   █* * *█* *█ █ █   █* *█* * * * * *█*█
█████ █████ █ █ █ █ ███ █ █ ███████████ █
█* * *█* * *█*█     █* *█*█*█     █ █* *█
█ █████ █████ █ █████ █ █ █ ███ █ █ █ █ █
█* *█* *█* * *█ █* * *█ █* *    █   █*█ █
█ █ █ ███ █████ █ █████████████████ █ ███
█ █* *█ █* * *█ █*  █ █ █* * * * *█ █* *█
█ █████ █████ ███ ███ █ █ ███ ███ █ ███ █
█   █   █   █*█* *█* * *█* *█ █* *█ █* *█
███ █ █ █ █ █ █ ███ ███ ███ ███ █████ ███
█     █   █ █* *█  * *█* * *█  * * * *  █
█████████████████████ ███████████████████
d_path [(10, 0), (9, 0), (9, 1), (10, 1), (11, 1), (11, 0), (12, 0), (13, 0), (13, 1), (12, 1), (12, 2), (13, 2), (14, 2), (15, 2), (16, 2), (16, 1), (15, 1), (15, 0), (16, 0), (17, 0), (18, 0), (18, 1), (19, 1), (19, 2), (18, 2), (18, 3), (18, 4), (19, 4), (19, 5), (19, 6), (18, 6), (18, 5), (17, 5), (16, 5), (15, 5), (14, 5), (13, 5), (13, 4), (13, 3), (12, 3), (12, 4), (12, 5), (11, 5), (11, 4), (10, 4), (10, 3), (9, 3), (8, 3), (8, 2), (8, 1), (7, 1), (7, 0), (6, 0), (6, 1), (6, 2), (5, 2), (4, 2), (4, 3), (5, 3), (6, 3), (6, 4), (6, 5), (5, 5), (5, 4), (4, 4), (3, 4), (3, 3), (2, 3), (2, 2), (1, 2), (1, 3), (0, 3), (0, 4), (1, 4), (2, 4), (2, 5), (3, 5), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7), (6, 7), (7, 7), (7, 8), (8, 8), (9, 8), (10, 8), (10, 7), (11, 7), (12, 7), (12, 6), (13, 6), (14, 6), (15, 6), (15, 7), (16, 7), (17, 7), (18, 7), (19, 7), (19, 8), (19, 9), (18, 9), (18, 10), (19, 10), (19, 11), (19, 12), (19, 13), (18, 13), (17, 13), (16, 13), (16, 14), (15, 14), (14, 14), (14, 15), (13, 15), (12, 15), (12, 16), (13, 16), (13, 17), (14, 17), (14, 16), (15, 16), (15, 15), (16, 15), (17, 15), (17, 16), (16, 16), (16, 17), (16, 18), (15, 18), (14, 18), (13, 18), (13, 19), (12, 19), (11, 19), (11, 20), (10, 20), (10, 19), (9, 19), (9, 20), (8, 20), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (6, 24), (6, 23), (6, 22), (5, 22), (4, 22), (4, 23), (3, 23), (3, 22), (3, 21), (3, 20), (2, 20), (2, 21), (2, 22), (1, 22), (1, 21), (0, 21), (0, 22), (0, 23), (1, 23), (2, 23), (2, 24)]
my_maze.solution_path [(10, 0), (9, 0), (9, 1), (10, 1), (11, 1), (11, 0), (12, 0), (13, 0), (13, 1), (12, 1), (12, 2), (13, 2), (14, 2), (15, 2), (16, 2), (16, 1), (15, 1), (15, 0), (16, 0), (17, 0), (18, 0), (18, 1), (19, 1), (19, 2), (18, 2), (18, 3), (18, 4), (19, 4), (19, 5), (19, 6), (18, 6), (18, 5), (17, 5), (16, 5), (15, 5), (14, 5), (13, 5), (13, 4), (13, 3), (12, 3), (12, 4), (12, 5), (11, 5), (11, 4), (10, 4), (10, 3), (9, 3), (8, 3), (8, 2), (8, 1), (7, 1), (7, 0), (6, 0), (6, 1), (6, 2), (5, 2), (4, 2), (4, 3), (5, 3), (6, 3), (6, 4), (6, 5), (5, 5), (5, 4), (4, 4), (3, 4), (3, 3), (2, 3), (2, 2), (1, 2), (1, 3), (0, 3), (0, 4), (1, 4), (2, 4), (2, 5), (3, 5), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7), (6, 7), (7, 7), (7, 8), (8, 8), (9, 8), (10, 8), (10, 7), (11, 7), (12, 7), (12, 6), (13, 6), (14, 6), (15, 6), (15, 7), (16, 7), (17, 7), (18, 7), (19, 7), (19, 8), (19, 9), (18, 9), (18, 10), (19, 10), (19, 11), (19, 12), (19, 13), (18, 13), (17, 13), (16, 13), (16, 14), (15, 14), (14, 14), (14, 15), (13, 15), (12, 15), (12, 16), (13, 16), (13, 17), (14, 17), (14, 16), (15, 16), (15, 15), (16, 15), (17, 15), (17, 16), (16, 16), (16, 17), (16, 18), (15, 18), (14, 18), (13, 18), (13, 19), (12, 19), (11, 19), (11, 20), (10, 20), (10, 19), (9, 19), (9, 20), (8, 20), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (6, 24), (6, 23), (6, 22), (5, 22), (4, 22), (4, 23), (3, 23), (3, 22), (3, 21), (3, 20), (2, 20), (2, 21), (2, 22), (1, 22), (1, 21), (0, 21), (0, 22), (0, 23), (1, 23), (2, 23), (2, 24)]
a_path [(10, 0), (9, 0), (9, 1), (10, 1), (11, 1), (11, 0), (12, 0), (13, 0), (13, 1), (12, 1), (12, 2), (13, 2), (14, 2), (15, 2), (16, 2), (16, 1), (15, 1), (15, 0), (16, 0), (17, 0), (18, 0), (18, 1), (19, 1), (19, 2), (18, 2), (18, 3), (18, 4), (19, 4), (19, 5), (19, 6), (18, 6), (18, 5), (17, 5), (16, 5), (15, 5), (14, 5), (13, 5), (13, 4), (13, 3), (12, 3), (12, 4), (12, 5), (11, 5), (11, 4), (10, 4), (10, 3), (9, 3), (8, 3), (8, 2), (8, 1), (7, 1), (7, 0), (6, 0), (6, 1), (6, 2), (5, 2), (4, 2), (4, 3), (5, 3), (6, 3), (6, 4), (6, 5), (5, 5), (5, 4), (4, 4), (3, 4), (3, 3), (2, 3), (2, 2), (1, 2), (1, 3), (0, 3), (0, 4), (1, 4), (2, 4), (2, 5), (3, 5), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7), (6, 7), (7, 7), (7, 8), (8, 8), (9, 8), (10, 8), (10, 7), (11, 7), (12, 7), (12, 6), (13, 6), (14, 6), (15, 6), (15, 7), (16, 7), (17, 7), (18, 7), (19, 7), (19, 8), (19, 9), (18, 9), (18, 10), (19, 10), (19, 11), (19, 12), (19, 13), (18, 13), (17, 13), (16, 13), (16, 14), (15, 14), (14, 14), (14, 15), (13, 15), (12, 15), (12, 16), (13, 16), (13, 17), (14, 17), (14, 16), (15, 16), (15, 15), (16, 15), (17, 15), (17, 16), (16, 16), (16, 17), (16, 18), (15, 18), (14, 18), (13, 18), (13, 19), (12, 19), (11, 19), (11, 20), (10, 20), (10, 19), (9, 19), (9, 20), (8, 20), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (6, 24), (6, 23), (6, 22), (5, 22), (4, 22), (4, 23), (3, 23), (3, 22), (3, 21), (3, 20), (2, 20), (2, 21), (2, 22), (1, 22), (1, 21), (0, 21), (0, 22), (0, 23), (1, 23), (2, 23), (2, 24)]
Dijkstra took 8.671 ms with Method.STACK
A* took 9.594 ms with Method.STACK
Maze Size 20
███████████ █████████████████████████████
█ █   █ █ █*█ █ █ █   █   █     █ █   █ █
█ ███ █ █ █ █ █ █ ███ █ ███ █████ █ ███ █
█     █ █  * *█ █   █       █   █ █     █
█ █ ███ █████ █ █ █████ ███████ █ █ █████
█ █     █ █  *  █ █     █   █     █     █
█ █████ █ ███ ███ █ █████ ███ █████ █ ███
█ █          *█ █ █   █ █ █         █ █ █
█ █████ █████ █ █ █ ███ █ █ █ █████████ █
█ █   █ █ █  *  █   █     █ █ █ █   █ █ █
█████ ███ ███ ███ █████ █ ███ █ █ ███ █ █
█       █    * *  █   █ █ █ █     █   █ █
███████ █ █ █ █ ███ ███ ███ ███ ███ ███ █
█ █ █   █ █ █ █*  █   █     █ █       █ █
█ █ ███ ███ █ █ ███ ███ █ ███ █ ███████ █
█     █ █   █ █*█   █   █ █   █ █   █   █
███ ███ ███████ ███ █ ███ █ ███ █ █████ █
█           █  * * *█   █ █ █ █ █   █   █
███████████ █ █ ███ █ █████ █ █ ███ █ ███
█ █ █ █ █   █ █ █  *█   █ █     █       █
█ █ █ █ █ ███████ █ █ ███ █ █████████ ███
█     █   █   █ █ █*          █ █     █ █
███ █ ███ ███ █ ███ ███ ███████ █████ █ █
█ █ █     █   █    *  █   █ █ █     █ █ █
█ ███ ███████ █████ ███████ █ █████ █ █ █
█     █     █   █* *█ █ █ █ █   █ █ █ █ █
█████ █ ███ ███ █ ███ █ █ █ █ ███ █ █ █ █
█     █ █    * * *  █   █ █   █   █ █ █ █
███ █ ███████ █████████ █ █ █████ █ █ █ █
█   █ █ █ █  *█   █ █   █   █ █   █     █
███ ███ █ ███ █ ███ █ ███ ███ ███ █ ███ █
█   █     █  *  █   █ █   █ █ █   █   █ █
███ █████ ███ ███ ███ █ █ █ █ ███ █ █████
█   █   █ █* *█   █ █   █ █   █       █ █
███ █ ███ █ ███ █ █ █ █████ █████ ███ █ █
█     █ █  *█ █ █     █ █   █ █     █   █
█████ █ ███ █ █ █ █████ ███ █ ███ ███ ███
█     █    *█ █ █ █ █ █ █ █ █ █     █   █
█████ █████ █ ███ █ █ █ █ █ █ █ █████████
█     █    *█ █   █ █     █ █           █
█████ █████ █ █ ███ ███ ███ █ ███████████
█          * *  █     █ █ █ █ █         █
█████ ███████ ███ █████ █ █ █ █ ███████ █
█ █     █    *█ █   █ █               █ █
█ █ █ ███ ███ █ █ ███ █ ███ █ █████ █████
█   █ █   █  * * *  █   █ █ █     █     █
█ ███ ███████ █ █ █████ █ █████ ███ █████
█ █   █       █ █* * *        █ █       █
█ █ █ █ █ █ █ ███ ███ ███ ███ █████ █ ███
█ █ █ █ █ █ █ █     █*  █   █   █   █   █
█████████████████████ ███████████████████
d_path [(10, 0), (10, 1), (9, 1), (8, 1), (8, 2), (7, 2), (6, 2), (6, 3), (6, 4), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (6, 9), (6, 10), (6, 11), (7, 11), (8, 11), (8, 12), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (8, 16), (7, 16), (7, 17), (7, 18), (7, 19), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (5, 23), (5, 24)]
my_maze.solution_path [(10, 0), (10, 1), (9, 1), (8, 1), (8, 2), (7, 2), (6, 2), (6, 3), (6, 4), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (6, 9), (6, 10), (6, 11), (7, 11), (8, 11), (8, 12), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (8, 16), (7, 16), (7, 17), (7, 18), (7, 19), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (5, 23), (5, 24)]
a_path [(10, 0), (10, 1), (9, 1), (8, 1), (8, 2), (7, 2), (6, 2), (6, 3), (6, 4), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (6, 9), (6, 10), (6, 11), (7, 11), (8, 11), (8, 12), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (8, 16), (7, 16), (7, 17), (7, 18), (7, 19), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (5, 23), (5, 24)]
███████████████████████████████████ █████
█ █     █     █ █ █         █     █*█ █ █
█ █ █ █ █████ █ █ ███ █████████ ███ █ █ █
█ █ █ █ █   █   █         █ █   █  * *  █
█ █ ███████ █ █ █████ █████ █ ███████ █ █
█       █   █ █ █     █   █ █ █ █ █  *█ █
█ █████████ █ ███ ███ █ █ █ █ █ █ ███ ███
█     █   █   █ █ █     █     █ █ █  *  █
█ ███ █ █████ █ █████ █ ███████ █ ███ ███
█   █ █ █ █ █ █ █     █   █   █     █*█ █
█ █████ █ █ █ █ █████████ ███ █ █ ███ █ █
█   █   █       █     █   █     █ █  *  █
█ █████ █████ ███████ ███ █ █████████ ███
█ █ █ █   █         █     █ █     █* *  █
█ █ █ █ █████████ █████ ███ █ █ ███ █ ███
█ █   █   █     █ █   █   █   █ █ █*█   █
█ ███ █ █████ ███ ███ ███ ███ ███ █ █████
█   █   █ █ █ █   █   █ █   █* * * *  █ █
███ █ ███ █ █ ███ ███ █ ███ █ █████ ███ █
█       █ █ █     █   █ █* * *  █ █     █
█ ███████ █ ███ █████ █ █ █████ █ ███████
█     █ █   █ █ █       █*  █ █   █   █ █
█ █████ ███ █ █ ███ ███ █ ███ █████ ███ █
█     █ █ █ █       █   █*█   █ █ █ █ █ █
███ ███ █ █ ███████████ █ ███ █ █ █ █ █ █
█   █     █ █ █     █ █  *        █     █
███ ███ ███ █ █████ █ ███ █ █████ █ ███ █
█ █       █   █   █   █ █*█ █   █   █ █ █
█ ███████ ███ ███ █ ███ █ ███ █ █████ ███
█   █ █ █     █ █       █*█ █ █       █ █
███ █ █ █ █████ █ █ █████ █ █ █████████ █
█ █ █ █ █ █ █ █   █ █   █*█ █ █     █   █
█ █ █ █ █ █ █ █ ███ █ ███ █ █ ███ ███ ███
█             █ █ █   █ █*█ █       █   █
█████ █ █ █ █████ █ ███ █ █ █ █ █████ ███
█ █ █ █ █ █   █ █ █* * * *    █ █   █   █
█ █ █████ █████ █ █ █████ █████████ █ ███
█           █   █  *  █ █ █ █     █   █ █
█ ███ █████ ███ ███ ███ ███ █ █████ ███ █
█ █ █ █ █ █   █    *      █ █   █     █ █
███ █ █ █ ███ █████ █ █████ █ ███ █████ █
█     █       █   █*█   █       █   █   █
███ █ █████ █ ███ █ ███████ ███████ ███ █
█ █ █ █     █      *  █   █ █ █ █       █
█ █ █ ███ █████████ ███ ███ █ █ █ █ █ █ █
█   █   █ █ █   █ █* *█ █   █     █ █ █ █
███ █ █████ █ █ █ ███ █ ███ █ ███ █ ███ █
█   █ █ █     █      *  █       █ █ █ █ █
█ █ █ █ ███ █████ ███ ███ ███████ █ █ █ █
█ █ █ █     █       █*          █ █   █ █
█████████████████████ ███████████████████
d_path [(10, 0), (10, 1), (10, 2), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (10, 7), (11, 7), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (13, 15), (14, 15), (14, 16), (15, 16), (16, 16), (17, 16), (17, 17), (17, 18), (18, 18), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (17, 23), (17, 24)]
my_maze.solution_path [(10, 0), (10, 1), (10, 2), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (10, 7), (11, 7), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (13, 15), (14, 15), (14, 16), (15, 16), (16, 16), (17, 16), (17, 17), (17, 18), (18, 18), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (17, 23), (17, 24)]
a_path [(10, 0), (10, 1), (10, 2), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (10, 7), (11, 7), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (13, 15), (14, 15), (14, 16), (15, 16), (16, 16), (17, 16), (17, 17), (17, 18), (18, 18), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (17, 23), (17, 24)]
Dijkstra took 4.104 ms with Method.RANDOM
A* took 4.085 ms with Method.RANDOM
Maze Size 20
█████████████████████████████████ ███████
█   █             █ █   █ █ █    *  █   █
███ █████████ █████ █ ███ █ █████ ███ ███
█ █ █   █   █   █   █   █ █ █    *      █
█ █ █ █████ ███ █ ███ ███ █ █████ ███████
█     █   █ █         █     █ █* *      █
█████ ███ █ █████ █████ █████ █ █ ███████
█       █   █ █ █ █ █ █   █ █  *█ █   █ █
█ █████ █ ███ █ █ █ █ █ ███ ███ ███ ███ █
█ █ █   █     █     █     █ █ █*█   █   █
███ ███ █ █████████ █ █████ █ █ █ ███ █ █
█     █   █ █ █ █ █ █ █ █ █ █  *  █   █ █
█████ █ ███ █ █ █ █ █ █ █ █ ███ ███ █ ███
█ █   █     █   █         █   █*█   █ █ █
█ ███ █ █████ ███████ █ ███ █ █ ███ ███ █
█ █   █ █   █ █ █ █   █   █ █ █*  █ █ █ █
█ ███ █ ███ █ █ █ ███ █ ███ ███ ███ █ █ █
█       █ █ █ █     █ █ █* * * *        █
███████ █ █ █ █ █████ ███ █ ███ █ █ █████
█     █     █   █ █   █ █*█ █ █ █ █ █ █ █
█████ ███ █████ █ █ ███ █ ███ ███████ █ █
█ █           █     █   █*    █     █ █ █
█ ███ █████ █████ █████ █ ███████ ███ █ █
█ █   █             █ █ █*  █ █ █       █
█ ███████████ █ █ █ █ █ █ ███ █ █ ███████
█ █       █   █ █ █ █   █*█ █   █ █ █ █ █
█ █████ █ ███ ███ █████ █ █ █ █ █ █ █ █ █
█ █   █ █ █   █ █ █ █ █* *█ █ █         █
█ ███ ███ █████ █ █ █ █ ███ ███ ███████ █
█   █ █ █   █ █ █   █  *  █       █   █ █
███ █ █ █ ███ █ █ █████ ███ █████ ███ ███
█ █ █ █     █ █ █   █* *  █   █ █   █ █ █
█ █ █ █ █████ █ █ ███ ███████ █ █████ █ █
█ █       █ █ █   █ █* *█ █ █   █ █ █   █
█ █████ ███ █ █ █ █ ███ █ █ █ ███ █ ███ █
█   █ █   █     █ █   █*    █       █ █ █
███ █ █ █████████ ███ █ █████ ███████ █ █
█     █ █ █       █ █  *  █       █   █ █
█ ███ █ █ █████ ███ ███ █████ █████ ███ █
█ █     █ █           █*█               █
███████ █ ███████████ █ █ █████ ███ ███ █
█     █ █ █   █   █    *█ █ █ █ █ █ █ █ █
█████ █ █ ███ ███ █████ █ █ █ ███ ███ ███
█     █   █ █ █ █   █  *  █       █   █ █
█████ ███ █ █ █ ███ ███ ███ █████ █ █ █ █
█ █ █ █         █ █ █* *█ █   █ █ █ █ █ █
█ █ █ ███████ ███ █ █ ███ █ ███ ███ ███ █
█   █ █ █       █    * *█     █   █ █   █
███ █ █ ███████ ███████ █ █████ ███ █ ███
█                    * *                █
█████████████████████ ███████████████████
d_path [(10, 0), (11, 0), (11, 1), (10, 1), (10, 2), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (10, 8), (10, 9), (11, 9), (11, 10), (11, 11), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (13, 16), (14, 16), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (16, 22), (16, 23), (16, 24)]
my_maze.solution_path [(10, 0), (11, 0), (11, 1), (10, 1), (10, 2), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (10, 8), (10, 9), (11, 9), (11, 10), (11, 11), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (13, 16), (14, 16), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (16, 22), (16, 23), (16, 24)]
a_path [(10, 0), (11, 0), (11, 1), (10, 1), (10, 2), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (10, 8), (10, 9), (11, 9), (11, 10), (11, 11), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (13, 16), (14, 16), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (16, 22), (16, 23), (16, 24)]
███████████ █████████████████████████████
█   █ █   █*  █   █ █ █       █ █       █
███ █ ███ █ ███ ███ █ █ ███████ █ ███████
█       █ █*█ █     █     █ █ █         █
█████ █ █ █ █ █ █████ █ █ █ █ █ █████████
█ █ █ █   █*█       █ █ █ █     █ █ █   █
█ █ ███ █ █ ███ █████ █████ █ ███ █ █ ███
█ █   █ █ █*█   █   █ █   █ █ █ █   █   █
█ ███ ███ █ █ ███ ███ █ ███ ███ █ █████ █
█   █ █   █*█           █ █   █ █     █ █
███ █ ███ █ ███ █████████ █ ███ █████ █ █
█         █*█ █ █     █ █ █ █ █ █ █     █
███ █████ █ █ █ █ █████ █ █ █ █ █ █ █████
█   █   █ █*█ █   █ █     █ █     █     █
███ █ █ █ █ █ █ █ █ █ █████ ███ █████ █ █
█   █ █    * * *█ █ █ █   █   █     █ █ █
███████ ███ █ █ ███ █ █ ███ █████ ███ ███
█ █     █   █ █* *  █ █ █ █   █         █
█ █████████ █ █ █ ███ █ █ █ █████████ ███
█   █     █ █ █ █*█ █         █ █ █ █   █
███ █████ ███████ █ █ █████████ █ █ ███ █
█ █   █     █    *█   █   █ █ █     █   █
█ █ ███████ █████ █ ███ ███ █ ███ █████ █
█ █     █   █   █*  █   █     █ █ █ █   █
█ ███ █████ ███ █ ███ ███ █████ █ █ █ █ █
█   █   █   █   █*█         █   █   █ █ █
███ █ █ ███ █ █ █ ███ █████████ █ ███ ███
█ █   █ █ █   █  *  █     █     █       █
█ ███ ███ ███████ ███ ███ ███ ███ ███████
█   █   █       █* *  █ █ █ █ █ █ █ █   █
███ ███ █████ █ ███ ███ ███ █ █ █ █ █ ███
█ █   █     █ █ █ █*█ █ █ █   █ █     █ █
█ ███ ███ █████ █ █ █ █ █ █ ███ █ █████ █
█               █ █* *█     █ █ █     █ █
███████████████ █ ███ █ █████ █ ███ ███ █
█ █ █ █       █ █   █*    █ █ █       █ █
█ █ █ ███ █ ███ █ ███ ███ █ █ ███ █████ █
█ █ █   █ █ █   █ █  *█ █ █   █ █ █   █ █
█ █ ███ ███ ███ █ ███ █ █████ █ █ █ ███ █
█     █   █     █ █* *█ █ █   █   █   █ █
█████ ███ █████ █ █ ███ █ ███ █ ███ ███ █
█     █     █ █  * *█ █   █ █ █ █ █   █ █
█ ███ █████ █ ███ ███ █ ███ █ █ █ █ ███ █
█ █ █ █         █*█ █   █ █ █   █       █
███ █ █████████ █ █ ███ █ █ ███ █ ███████
█                *█   █ █ █ █ █         █
█ ███ █ █████████ ███ █ █ █ █ █ ███ █████
█ █   █ █        * * *█   █     █ █ █   █
███ ███████ █████ ███ █ █████ ███ ███ ███
█   █       █     █  *                  █
█████████████████████ ███████████████████
d_path [(10, 0), (10, 1), (9, 1), (8, 1), (8, 2), (8, 3), (8, 4), (9, 4), (9, 5), (10, 5), (10, 6), (10, 7), (10, 8), (9, 8), (9, 9), (9, 10), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (7, 16), (7, 17), (6, 17), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24)]
my_maze.solution_path [(10, 0), (10, 1), (9, 1), (8, 1), (8, 2), (8, 3), (8, 4), (9, 4), (9, 5), (10, 5), (10, 6), (10, 7), (10, 8), (9, 8), (9, 9), (9, 10), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (7, 16), (7, 17), (6, 17), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24)]
a_path [(10, 0), (10, 1), (9, 1), (8, 1), (8, 2), (8, 3), (8, 4), (9, 4), (9, 5), (10, 5), (10, 6), (10, 7), (10, 8), (9, 8), (9, 9), (9, 10), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (7, 16), (7, 17), (6, 17), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24)]
Dijkstra took 4.493 ms with Method.BIAS
A* took 4.376 ms with Method.BIAS
Maze Size 40
█████████████████████████████████████████████████████████████████████████ ███████
█ █ █       █ █ █ █ █ █             █   █ █     █   █ █   █     █     █ █* *█   █
█ █ █████ ███ █ █ █ █ ███ █ █████ █████ █ ███ ███ ███ ███ █████ █████ █ ███ █ ███
█ █ █     █       █     █ █     █   █ █ █ █   █     █           █   █   █* *  █ █
█ █ █ ███████ ███████ █ ███ █████████ █ █ █ █████ █████████████ ███ ███ █ █████ █
█ █       █ █   █     █     █   █ █   █   █ █     █   █     █   █ █ █  * *█* * *█
█ ███████ █ █ ███████ █████ █ ███ █ █████ █ █████ ███ █████ ███ █ █ ███ ███ ███ █
█       █   █   █ █ █     █ █       █   █ █ █       █ █     █ █     █  * * *  █*█
█████ ███ █████ █ █ █ ███████ ███████ ███ █ ███████ █ ███ █ █ █████ ███████ ███ █
█ █ █ █ █ █   █ █ █ █ █ █   █ █   █ █ █ █   █   █ █ █   █ █   █           █   █*█
█ █ █ █ █ ███ █ █ █ █ █ ███ █ █ ███ █ █ ███ ███ █ █ █ █████ █ █ ███ █████ █████ █
█           █       █ █ █   █   █   █ █ █ █ █     █ █ █ █   █ █ █ █ █    * * *█*█
█ █ ███████ █████ ███ █ ███ ███ █ ███ █ █ █ ███ ███ █ █ ███ █ ███ ███████ ███ █ █
█ █ █ █         █ █     █   █ █ █   █   █ █         █       █            * *█*█*█
█████ ███ █████ █ ███ ███ ███ █ █ █████ █ ███████ █ ███████ ███ █ ███████ █ █ █ █
█ █ █ █   █   █   █     █   █       █     █   █   █     █ █ █   █ █       █*█*█*█
█ █ █ ███████ ███ ███ ███ █████ ███████ ███ ███████████ █ █████████ ███ █ █ █ █ █
█             █   █   █   █ █                 █   █   █   █     █   █   █ █*█* *█
█████████████ ███ █ ███ ███ █ ███████████████████ ███ █ █ █ █████████ █ ███ ███ █
█ █ █   █         █ █   █             █       █ █ █ █   █   █       █ █ █ █*█ █ █
█ █ ███ ███████ █ █ ███ █ █████████████ ███████ █ █ ███████ █████ ███████ █ █ █ █
█   █ █ █   █   █         █ █   █   █ █   █     █     █   █   █ █   █ █    *█   █
█ █ █ █ ███ █ ███ ███████ █ █ ███ ███ █ █████ ███████ ███ █ ███ █ ███ █████ █████
█ █           █ █ █    * * *█ █ █   █       █       █ █ █ █ █   █   █      * * *█
███████████████ ███ ███ ███ █ █ █ ███ ███████ ███████ █ █ █ ███ █ █ ███ ███ ███ █
█         █       █ █ █*█ █*█       █ █ █   █   █   █     █ █ █ █ █ █ █ █   █* *█
█ █████ █ ███████ ███ █ █ █ █ ███████ █ ███ █ ███ ███████ █ █ █ █ ███ ███ █ █ ███
█ █   █ █       █      *█  * *  █ █   █ █   █ █ █ █        * * *  █ █   █ █ █* *█
█████ █ ███ █ ███ █████ █████ ███ █ ███ █ ███ █ █ ███████ █ ███ ███ ███ █ █████ █
█     █ █   █ █   █    * *█ █* *█* * * * *  █ █     █ █   █*█* *        █   █  *█
█████ ███████ ███ ███████ █ ███ █ ███████ ███ █ █████ █████ █ █████ ███████ ███ █
█   █ █ █       █ █  * * *█   █*█*█     █* * *      █      *█* * *█   █ █ █ █  *█
███ █ █ ███████ █████ █████ █ █ █ █ ███ █████ █████ ███████ █████ █████ █ █████ █
█                 █  * *█ █ █ █* *█ █       █*█   █ █ █    * *█ █*  █       █ █*█
███████ ███████ █ █████ █ █ █ █████ ███████ █ █ █ ███ ███ █ █ █ █ █████████ █ █ █
█         █     █     █*█   █     █   █   █ █*█ █ █ █ █   █ █*█ █*    █  * *█* *█
█████ █ ███ █████████ █ █████ ███ █ █ █ █ ███ █ ███ █ █ █ ███ █ █ ███████ █ █ ███
█     █ █ █ █         █* * *█   █ █ █   █ █* *  █ █   █ █ █ █*█* *█   █  *█*█* *█
███ █ ███ █████ ███ █ █████ ███ █ █ █████ █ █████ █ ███████ █ █ ███ █ ███ █ █ █ █
█   █ █   █     █ █ █  * * *█   █   █   █ █* *█   █ █   █  * *█*  █ █  * *█*█ █*█
███ █████ ███████ █████ █████████████ █ █ ███ █ ███ ███ ███ ███ ███████ ███ ███ █
█   █   █     █     █* *█             █   █* *█   █    * * *█* *█   █* *█  * * *█
█ █████ ███ ███████ █ ███ █████████████ ███ ███ ███████ █ █ █ ███ ███ █ █████ █ █
█ █ █ █ █   █   █ █ █*█       █       █ █* *█* *█  * * *█ █ █*█* * * *█ █* *█ █ █
███ █ █ ███ ███ █ █ █ █ █████ █████ █ █ █ ███ █ ███ █████████ █ █████████ █ █████
█ █                  *█     █     █ █   █*█  *█* *█* *█     █*█* * *█* *█*█* * *█
█ ███████ █ ███ ███ █ █ ███████ █ ███████ ███ ███ ███ █ █ ███ █████ █ █ █ █████ █
█     █   █ █ █ █   █*█ █* * *█ █      * *█* *█ █*█* *  █   █* *█ █* *█* *  █ █*█
█████ █ █ █ █ ███ █ █ ███ ███ █████████ ███ ███ █ █ ███ ███ ███ █ ███████████ █ █
█   █ █ █ █ █     █ █* * *█* *█* * * *█* * *█* *█* *█ █ █ █ █* *█* * * *█* * * *█
███ █ █ █████ █████ ███ ███ █ █ █████ ███████ █ █ █ █ █ █ ███ ███ █ █ █ █ █ █ █ █
█       █     █       █ █  *█ █*█   █* *█* * *█*█ █ █ █ █ █ █* * *█ █ █* *█ █ █ █
███████ █ █ █████ █████ █ █ █ █ █ █ ███ █ █████ ███ █ ███ █ █ █ ███ ███ █████ ███
█ █     █ █ █     █     █ █*█ █*█ █   █* *█* *█*  █     █     █   █ █       █   █
█ █ ███ █████ █████████████ ███ ███ █ █████ █ █ ███████ ███████ █ █ ███ ███ ███ █
█ █ █ █ █ █     █   █ █    * *█* *█ █     █*█* *  █   █     █   █ █ █     █ █   █
█ ███ █ █ ███ █████ █ ███ ███ ███ █ █████ █ ███████ █████████ █ ███ █ █ █ █ █ █ █
█ █     █     █ █       █ █  * *█*█   █ █ █* * * * *█       █ █   █ █ █ █ █ █ █ █
█ ███ ███ ███ █ █ ███ █ ███████ █ ███ █ █ █████████ █ ███████ ███ █████████ ███ █
█     █   █   █   █   █   █   █* *    █           █*█       █   █     █     █   █
███ █ ███████ █ █████████ ███ █████████████████████ █ ███ ███ ███ █ █ █ █ █ ███ █
█   █   █     █   █     █* * * * * * * *█* * * * *█* *█ █   █ █   █ █ █ █ █   █ █
███████████ █ ███ █████ █ █████ █ █████ █ ███ ███ █ █ █ █████████ ███ █ ███ █ ███
█           █ █   █      * *  █ █ █    * *  █ █* *█ █*          █ █   █ █   █   █
█████████ █ ███████ ███████ █████████ █ █ ███ █ █████ █ █████ ███████████ █ ███ █
█         █ █       █    * *  █ █   █ █ █ █   █* * * *█     █         █   █ █   █
█ █ █████ ███ ███████████ █████ █ ███ █████ ███ ███████████ █████ █████ █ █ ███ █
█ █   █     █ █       █ █* * *  █   █   █     █           █     █   █   █ █ █   █
█ ███ █ █ █ █████████ █ █████ ███ ███████ █████ █ ███████████ █████ █ ███ ███ █ █
█ █   █ █ █ █ █            * *█ █ █     █     █ █       █   █     █ █   █   █ █ █
█████ █ █████ █ ███████ ███ ███ █ █ ███████████ ███ █ █ █ █ █ █ █ █████ █ █████ █
█     █ █   █ █ █       █* *█* * * * *█* *    █   █ █ █   █ █ █ █     █ █ █     █
█ █ █████ ███ █ █ ███████ ███ ███████ █ █ █████████ █ ███ ███ ███ █ ███████ █ ███
█ █ █ █         █ █      *█ █* * *█ █* *█*      █ █ █ █     █ █ █ █ █     █ █   █
███ █ █████ █████ ███ ███ █ █████ █ █████ ███████ █████ █ █ ███ █ █ █ ███ █ ███ █
█   █       █   █ █   █  * * *█* *    █* *  █ █     █   █ █     █ █     █ █ █   █
█████ █ ███ █ █ ███ ███ █████ █ ███████ █ ███ █ ███████ ███ █ █ █████ █ █ ███ ███
█     █ █ █ █ █     █   █* * *█* *█* * *█ █       █   █   █ █ █     █ █ █ █     █
█ ███ ███ █████ █ ███████ ███████ █ ███ ███ ███████ █ █████ ███ ███ ███ ███████ █
█ █ █ █     █   █ █      * * * * *█* *█ █ █ █     █ █     █   █   █   █   █     █
███ ███████ ███ █████ ███ █ █ █ █████ ███ █ █ █████ █ ███ ███ █████████ █ ███ ███
█     █         █ █   █   █ █ █   █  * *█       █ █ █   █   █ █       █ █ █     █
███ █ ███████████ █ █ █ █ █ █ ███████ █ ███ █████ █ ███ ███████ ███ ███ █████ █ █
█   █     █       █ █ █ █ █ █ █       █* * *█   █     █ █   █   █ █   █     █ █ █
█ ███ █ █ ███████ ███ █████ █ █████ ███ █ █ █ █████ █████ ███ ███ █████████ ███ █
█ █   █ █   █         █     █ █   █ █   █ █*█       █   █                 █   █ █
█████ ███ ███ █ ███ ███████ ███ █ ███████ █ █ ███████ ███ ███████ █ ███████████ █
█     █ █     █ █     █       █ █     █ █ █*      █   █       █   █         █   █
███████ ███ █ █ █ █ █ █ █ █ █████ █ █ █ ███ █ █████ █ ███ █████ █████ █████████ █
█           █ █ █ █ █ █ █ █ █     █ █    * *█       █         █   █           █ █
█████████████████████████████████████████ ███████████████████████████████████████
d_path [(20, 0), (21, 0), (21, 1), (21, 2), (21, 3), (20, 3), (19, 3), (19, 4), (18, 4), (18, 5), (17, 5), (17, 6), (18, 6), (19, 6), (19, 7), (20, 7), (20, 8), (20, 9), (19, 9), (19, 8), (18, 8), (18, 9), (17, 9), (16, 9), (15, 9), (14, 9), (14, 8), (15, 8), (16, 8), (16, 7), (15, 7), (15, 6), (16, 6), (16, 5), (15, 5), (14, 5), (13, 5), (12, 5), (12, 6), (13, 6), (14, 6), (14, 7), (13, 7), (12, 7), (12, 8), (12, 9), (13, 9), (13, 10), (14, 10), (14, 11), (13, 11), (12, 11), (12, 12), (13, 12), (13, 13), (12, 13), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (19, 14), (19, 13), (20, 13), (20, 14), (21, 14), (22, 14), (23, 14), (24, 14), (24, 13), (23, 13), (23, 12), (24, 12), (25, 12), (26, 12), (26, 13), (26, 14), (25, 14), (25, 15), (25, 16), (24, 16), (23, 16), (22, 16), (21, 16), (21, 17), (21, 18), (22, 18), (22, 17), (23, 17), (23, 18), (23, 19), (23, 20), (22, 20), (22, 19), (21, 19), (20, 19), (20, 18), (19, 18), (19, 19), (18, 19), (18, 20), (17, 20), (16, 20), (15, 20), (15, 19), (15, 18), (15, 17), (16, 17), (16, 16), (16, 15), (15, 15), (15, 16), (14, 16), (14, 17), (13, 17), (13, 18), (13, 19), (13, 20), (14, 20), (14, 21), (13, 21), (12, 21), (12, 20), (11, 20), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (11, 24), (11, 25), (12, 25), (13, 25), (13, 26), (12, 26), (11, 26), (11, 27), (11, 28), (10, 28), (10, 29), (11, 29), (12, 29), (12, 30), (11, 30), (11, 31), (11, 32), (11, 33), (12, 33), (13, 33), (13, 32), (13, 31), (14, 31), (14, 30), (15, 30), (15, 29), (15, 28), (16, 28), (16, 29), (16, 30), (17, 30), (18, 30), (19, 30), (20, 30), (20, 29), (21, 29), (22, 29), (22, 28), (22, 27), (22, 26), (21, 26), (21, 25), (22, 25), (22, 24), (21, 24), (21, 23), (20, 23), (20, 22), (20, 21), (19, 21), (19, 20), (20, 20), (21, 20), (21, 21), (22, 21), (22, 22), (22, 23), (23, 23), (23, 22), (24, 22), (24, 21), (24, 20), (25, 20), (25, 21), (26, 21), (26, 22), (25, 22), (25, 23), (26, 23), (27, 23), (27, 24), (28, 24), (29, 24), (29, 25), (30, 25), (30, 26), (30, 27), (30, 28), (29, 28), (29, 29), (29, 30), (29, 31), (30, 31), (31, 31), (31, 30), (30, 30), (30, 29), (31, 29), (32, 29), (32, 28), (32, 27), (32, 26), (31, 26), (31, 25), (31, 24), (30, 24), (30, 23), (30, 22), (30, 21), (31, 21), (31, 20), (30, 20), (30, 19), (31, 19), (32, 19), (32, 20), (33, 20), (34, 20), (35, 20), (35, 19), (36, 19), (36, 20), (37, 20), (38, 20), (39, 20), (39, 21), (39, 22), (38, 22), (37, 22), (37, 23), (36, 23), (36, 22), (36, 21), (35, 21), (35, 22), (34, 22), (34, 21), (33, 21), (33, 22), (32, 22), (31, 22), (31, 23), (32, 23), (33, 23), (34, 23), (34, 24), (35, 24), (35, 25), (36, 25), (36, 26), (36, 27), (37, 27), (37, 26), (37, 25), (37, 24), (38, 24), (39, 24), (39, 25), (39, 26), (38, 26), (38, 27), (39, 27), (39, 28), (39, 29), (39, 30), (39, 31), (38, 31), (38, 32), (39, 32), (39, 33), (38, 33), (37, 33), (37, 34), (37, 35), (37, 36), (37, 37), (37, 38), (36, 38), (36, 39), (37, 39), (38, 39), (38, 38), (38, 37), (38, 36), (39, 36), (39, 37), (39, 38), (39, 39), (39, 40), (39, 41), (39, 42), (38, 42), (37, 42), (37, 41), (36, 41), (35, 41), (35, 42), (36, 42), (36, 43), (37, 43), (37, 44), (36, 44)]
my_maze.solution_path [(20, 0), (21, 0), (21, 1), (21, 2), (21, 3), (20, 3), (19, 3), (19, 4), (18, 4), (18, 5), (17, 5), (17, 6), (18, 6), (19, 6), (19, 7), (20, 7), (20, 8), (20, 9), (19, 9), (19, 8), (18, 8), (18, 9), (17, 9), (16, 9), (15, 9), (14, 9), (14, 8), (15, 8), (16, 8), (16, 7), (15, 7), (15, 6), (16, 6), (16, 5), (15, 5), (14, 5), (13, 5), (12, 5), (12, 6), (13, 6), (14, 6), (14, 7), (13, 7), (12, 7), (12, 8), (12, 9), (13, 9), (13, 10), (14, 10), (14, 11), (13, 11), (12, 11), (12, 12), (13, 12), (13, 13), (12, 13), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (19, 14), (19, 13), (20, 13), (20, 14), (21, 14), (22, 14), (23, 14), (24, 14), (24, 13), (23, 13), (23, 12), (24, 12), (25, 12), (26, 12), (26, 13), (26, 14), (25, 14), (25, 15), (25, 16), (24, 16), (23, 16), (22, 16), (21, 16), (21, 17), (21, 18), (22, 18), (22, 17), (23, 17), (23, 18), (23, 19), (23, 20), (22, 20), (22, 19), (21, 19), (20, 19), (20, 18), (19, 18), (19, 19), (18, 19), (18, 20), (17, 20), (16, 20), (15, 20), (15, 19), (15, 18), (15, 17), (16, 17), (16, 16), (16, 15), (15, 15), (15, 16), (14, 16), (14, 17), (13, 17), (13, 18), (13, 19), (13, 20), (14, 20), (14, 21), (13, 21), (12, 21), (12, 20), (11, 20), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (11, 24), (11, 25), (12, 25), (13, 25), (13, 26), (12, 26), (11, 26), (11, 27), (11, 28), (10, 28), (10, 29), (11, 29), (12, 29), (12, 30), (11, 30), (11, 31), (11, 32), (11, 33), (12, 33), (13, 33), (13, 32), (13, 31), (14, 31), (14, 30), (15, 30), (15, 29), (15, 28), (16, 28), (16, 29), (16, 30), (17, 30), (18, 30), (19, 30), (20, 30), (20, 29), (21, 29), (22, 29), (22, 28), (22, 27), (22, 26), (21, 26), (21, 25), (22, 25), (22, 24), (21, 24), (21, 23), (20, 23), (20, 22), (20, 21), (19, 21), (19, 20), (20, 20), (21, 20), (21, 21), (22, 21), (22, 22), (22, 23), (23, 23), (23, 22), (24, 22), (24, 21), (24, 20), (25, 20), (25, 21), (26, 21), (26, 22), (25, 22), (25, 23), (26, 23), (27, 23), (27, 24), (28, 24), (29, 24), (29, 25), (30, 25), (30, 26), (30, 27), (30, 28), (29, 28), (29, 29), (29, 30), (29, 31), (30, 31), (31, 31), (31, 30), (30, 30), (30, 29), (31, 29), (32, 29), (32, 28), (32, 27), (32, 26), (31, 26), (31, 25), (31, 24), (30, 24), (30, 23), (30, 22), (30, 21), (31, 21), (31, 20), (30, 20), (30, 19), (31, 19), (32, 19), (32, 20), (33, 20), (34, 20), (35, 20), (35, 19), (36, 19), (36, 20), (37, 20), (38, 20), (39, 20), (39, 21), (39, 22), (38, 22), (37, 22), (37, 23), (36, 23), (36, 22), (36, 21), (35, 21), (35, 22), (34, 22), (34, 21), (33, 21), (33, 22), (32, 22), (31, 22), (31, 23), (32, 23), (33, 23), (34, 23), (34, 24), (35, 24), (35, 25), (36, 25), (36, 26), (36, 27), (37, 27), (37, 26), (37, 25), (37, 24), (38, 24), (39, 24), (39, 25), (39, 26), (38, 26), (38, 27), (39, 27), (39, 28), (39, 29), (39, 30), (39, 31), (38, 31), (38, 32), (39, 32), (39, 33), (38, 33), (37, 33), (37, 34), (37, 35), (37, 36), (37, 37), (37, 38), (36, 38), (36, 39), (37, 39), (38, 39), (38, 38), (38, 37), (38, 36), (39, 36), (39, 37), (39, 38), (39, 39), (39, 40), (39, 41), (39, 42), (38, 42), (37, 42), (37, 41), (36, 41), (35, 41), (35, 42), (36, 42), (36, 43), (37, 43), (37, 44), (36, 44)]
a_path [(20, 0), (21, 0), (21, 1), (21, 2), (21, 3), (20, 3), (19, 3), (19, 4), (18, 4), (18, 5), (17, 5), (17, 6), (18, 6), (19, 6), (19, 7), (20, 7), (20, 8), (20, 9), (19, 9), (19, 8), (18, 8), (18, 9), (17, 9), (16, 9), (15, 9), (14, 9), (14, 8), (15, 8), (16, 8), (16, 7), (15, 7), (15, 6), (16, 6), (16, 5), (15, 5), (14, 5), (13, 5), (12, 5), (12, 6), (13, 6), (14, 6), (14, 7), (13, 7), (12, 7), (12, 8), (12, 9), (13, 9), (13, 10), (14, 10), (14, 11), (13, 11), (12, 11), (12, 12), (13, 12), (13, 13), (12, 13), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (19, 14), (19, 13), (20, 13), (20, 14), (21, 14), (22, 14), (23, 14), (24, 14), (24, 13), (23, 13), (23, 12), (24, 12), (25, 12), (26, 12), (26, 13), (26, 14), (25, 14), (25, 15), (25, 16), (24, 16), (23, 16), (22, 16), (21, 16), (21, 17), (21, 18), (22, 18), (22, 17), (23, 17), (23, 18), (23, 19), (23, 20), (22, 20), (22, 19), (21, 19), (20, 19), (20, 18), (19, 18), (19, 19), (18, 19), (18, 20), (17, 20), (16, 20), (15, 20), (15, 19), (15, 18), (15, 17), (16, 17), (16, 16), (16, 15), (15, 15), (15, 16), (14, 16), (14, 17), (13, 17), (13, 18), (13, 19), (13, 20), (14, 20), (14, 21), (13, 21), (12, 21), (12, 20), (11, 20), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (11, 24), (11, 25), (12, 25), (13, 25), (13, 26), (12, 26), (11, 26), (11, 27), (11, 28), (10, 28), (10, 29), (11, 29), (12, 29), (12, 30), (11, 30), (11, 31), (11, 32), (11, 33), (12, 33), (13, 33), (13, 32), (13, 31), (14, 31), (14, 30), (15, 30), (15, 29), (15, 28), (16, 28), (16, 29), (16, 30), (17, 30), (18, 30), (19, 30), (20, 30), (20, 29), (21, 29), (22, 29), (22, 28), (22, 27), (22, 26), (21, 26), (21, 25), (22, 25), (22, 24), (21, 24), (21, 23), (20, 23), (20, 22), (20, 21), (19, 21), (19, 20), (20, 20), (21, 20), (21, 21), (22, 21), (22, 22), (22, 23), (23, 23), (23, 22), (24, 22), (24, 21), (24, 20), (25, 20), (25, 21), (26, 21), (26, 22), (25, 22), (25, 23), (26, 23), (27, 23), (27, 24), (28, 24), (29, 24), (29, 25), (30, 25), (30, 26), (30, 27), (30, 28), (29, 28), (29, 29), (29, 30), (29, 31), (30, 31), (31, 31), (31, 30), (30, 30), (30, 29), (31, 29), (32, 29), (32, 28), (32, 27), (32, 26), (31, 26), (31, 25), (31, 24), (30, 24), (30, 23), (30, 22), (30, 21), (31, 21), (31, 20), (30, 20), (30, 19), (31, 19), (32, 19), (32, 20), (33, 20), (34, 20), (35, 20), (35, 19), (36, 19), (36, 20), (37, 20), (38, 20), (39, 20), (39, 21), (39, 22), (38, 22), (37, 22), (37, 23), (36, 23), (36, 22), (36, 21), (35, 21), (35, 22), (34, 22), (34, 21), (33, 21), (33, 22), (32, 22), (31, 22), (31, 23), (32, 23), (33, 23), (34, 23), (34, 24), (35, 24), (35, 25), (36, 25), (36, 26), (36, 27), (37, 27), (37, 26), (37, 25), (37, 24), (38, 24), (39, 24), (39, 25), (39, 26), (38, 26), (38, 27), (39, 27), (39, 28), (39, 29), (39, 30), (39, 31), (38, 31), (38, 32), (39, 32), (39, 33), (38, 33), (37, 33), (37, 34), (37, 35), (37, 36), (37, 37), (37, 38), (36, 38), (36, 39), (37, 39), (38, 39), (38, 38), (38, 37), (38, 36), (39, 36), (39, 37), (39, 38), (39, 39), (39, 40), (39, 41), (39, 42), (38, 42), (37, 42), (37, 41), (36, 41), (35, 41), (35, 42), (36, 42), (36, 43), (37, 43), (37, 44), (36, 44)]
█████████████████████████████████████ ███████████████████████████████████████████
█ █ █ █ █         █                  *    █     █   █               █           █
█ █ █ █ ███ █████ ███████████ ███████ █████ ███ ███ █ ███████ ███████ ███████████
█     █     █   █ █ █         █      *█     █ █         █ █ █       █ █     █   █
█████ █████ █ █ █ █ ███████████████ █ █ █████ ███████ ███ █ █ ███ ███ █ ███ █ ███
█   █ █ █   █ █       █         █ █ █* *  █     █   █       █   █ █     █ █ █   █
█ ███ █ ███████████ ███████ ███ █ █████ ███ ███ █ █ ███ █████ ███████ ███ █████ █
█       █   █     █   █     █        * *      █   █   █   █ █       █ █   █ █ █ █
█████ █████ ███ █████ ███████ █████ █ ███████ █████████████ ███ █████ █ ███ █ █ █
█ █ █     █   █ █ █     █   █ █   █ █* * *  █         █ █ █   █ █   █     █ █   █
█ █ ███ ███ ███ █ █████ █ ███████ ███ ███ ███ █████████ █ ███ ███ ███ █ ███ ███ █
█   █     █     █ █ █     █     █     █ █*  █ █   █       █   █   █   █ █       █
███ ███ █ ███ █ █ █ █████ █████ ███ ███ █ █ █ █ ███ ███████ ███ ███ █ █████ █████
█   █ █ █   █ █ █ █   █       █ █ █ █ █* *█ █         █ █         █ █   █ █ █ █ █
███ █ ███ █████ █ ███ ███████ █ █ ███ █ ███ ███████ ███ ███ ███ ███ █████ █ █ █ █
█     █   █ █           █ █       █* * *  █     █       █ █   █ █   █ █ █   █ █ █
█ █ █████ █ █████████ ███ ███████ █ █ █████████████ █████ █ █ █████ █ █ ███ █ █ █
█ █ █     █   █ █   █ █     █   █ █*█ █           █     █ █ █ █     █       █ █ █
███ █████ █ ███ ███ █ █████ ███ █ █ ███ █████████████████ █ ███ ███████ █████ █ █
█ █ █ █     █     █     █      * * *              █   █ █         █ █   █ █     █
█ █ █ █████ █ █ █ ███ ███████ █ █ ███ ███████ █ ███ ███ █████ █████ █ ███ █████ █
█   █   █   █ █ █ █           █*█   █ █     █ █ █ █   █       █     █ █   █   █ █
███ ███ █ ███████ █████ ███ ███ █████████ ███ ███ ███ ███████ █ █████ █ █████ █ █
█     █ █   █ █ █   █     █ █ █*    █ █ █ █ █ █ █   █   █ █       █     █ █ █ █ █
█████ █ ███ █ █ █ █████ █████ █ ███ █ █ █ █ ███ ███ █ ███ █████ █████ ███ █ █ █ █
█ █   █ █     █   █     █ █    *  █         █     █     █ █   █ █   █ █ █ █ █ █ █
█ █ █ █ █████ █ █████████ █████ ███████ █████████ █ █████ ███ █ ███ █ █ █ █ █ █ █
█   █   █ █       █   █   █   █*█ █   █     █ █     █ █ █ █         █   █ █   █ █
█████ ███ ███ ███████ ███ ███ █ █ █ █████████ █████ █ █ █ ███████ ███ ███ █ █ █ █
█     █   █   █ █     █    * *█* *  █     █ █   █   █   █ █       █ █   █ █ █* *█
█████ ███ █ █ █ █ █████████ █ ███ █ █ █████ █ █████ █ ███ ███████ █ ███ █ ███ █ █
█ █ █ █ █   █* *    █ █  * *█* *█*█   █ █             █     █       █ █ █ █ █*█*█
█ █ █ █ █████ █ █████ ███ ███ █ █ █████ ███████████ ███ ███ █████ ███ █ █ █ █ █ █
█         █  *█* *  █ █  *  █ █* *      █ █ █   █     █   █ █ █   █ █ █* *█* *█*█
█████ ███████ ███ █ █ ███ █ █ ███████ ███ █ █ ███ █ ███ █████ ███ █ █ █ █ █ ███ █
█ █ █ █ █   █*█* *█   █ █*█ █       █     █   █   █* * *█* * * *█* * *█*█* *█* *█
█ █ █ █ ███ █ █ ███████ █ ███ ███ █████████ ███████ ███ █ █ █ █ █ █ █ █ █████ ███
█     █* * * *█* * * * * *  █   █ █ █ █ █ █ █ █   █*█* *█*█ █ █* *█ █* *█* * *  █
█████ █ ███████████████ ███ █ █████ █ █ █ █ █ ███ █ █ ███ ███████ ███████ █ █████
█ █* * *█* * * *      █   █ █   █ █ █ █       █* * *█* * *      █ █   █  *█ █   █
█ █ █████ █████ █████ ███ █ ███ █ █ █ ███████ █ █ █ █ ███ █████ █████ ███ █████ █
█* *█* * *█ █* *    █ █   █ █     █     █ █ █* *█ █ █ █     █ █ █ █ █ █  * * * *█
█ ███ █████ █ █ █████████████████████ ███ █ █ █ ███ ███ █ █ █ ███ █ █ █████████ █
█*  █*█     █*█ █ █       █ █       █ █   █ █*█   █   █ █ █ █ █  * *█  * * * *█*█
█ ███ █ ███ █ █ █ █ ███ ███ ███ ███ █ █ ███ █ █████ █ ███████ ███ █ ███ █████ █ █
█* * *█   █ █*█ █   █ █   █     █ █  * *    █*  █ █ █ █ █   █    *█* * *█   █* *█
█ ███████ █ █ ███ ███ █████ █ ███ █ █ █ █████ ███ █████ ███ █████ █████████ ███ █
█ █       █ █*          █ █ █ █     █*█* * * *        █ █  * *  █*█         █   █
█ ███ █████ █ ███ ███████ ███████████ █████████████████ ███ █ ███ ███ ███████ ███
█ █   █     █*█ █       █ █       █  * * * * * *█ █ █  * * *█* * *█   █       █ █
█ █ █ ███████ █ █████████ █ █ █ █ ███ █ ███ █ █ █ █ █ █ ███████████ ███ ███████ █
█   █ █* * *█* *  █   █ █ █ █ █ █     █ █ █ █ █*█ █   █* * *█ █       █ █       █
███ ███ ███ ███ ███ ███ █ ███████ ███████ █████ █ █ █ █ ███ █ █ █████ █ ███████ █
█   █* *█* *  █*█ █         █ █   █       █  * *  █ █ █ █* *█ █ █ █   █ █       █
█ ███ ███ █████ █ █ █ █ ███ █ ███ █████ █ ███ █ █████████ ███ █ █ █ ███ █ █████ █
█ █ █* *█* *█  * *  █ █   █ █ █   █ █ █ █* * *█       █  *█     █ █   █   █ █   █
█ █ ███ ███ █████ ███████ ███ █████ █ ███ █ █████████ ███ █ █████ ███ █████ █ ███
█     █*█ █*  █  *    █ █ █ █ █   █   █* *█ █       █ █  *█   █     █     █ █   █
███████ █ █ █████ █████ ███ █ ███ █ █ █ ███████ █████████ ███ █ ███ █ ███ █ ███ █
█* * * *█ █*  █  * *  █ █ █     █ █ █* *    █* *  █ █ █  *█   █ █   █ █       █ █
█ █████ █ █ ███████ ███ █ █████ █ ███ ███████ █ ███ █ ███ █ ███ █ ███ █████████ █
█* *  █ █* *█* *  █*  █     █ █   █  * * * * *█* * *█    *█ █   █   █         █ █
███ █████ ███ █ █ █ █████ ███ ███ █████████████████ ███ █ █ ███ ███ █████████ █ █
█ █* *█* *█* *█*█ █* * *█* * *█* *█* * * * *█ █* * *  █ █*█     █           █   █
█ ███ █ ███ ███ ███████ █ ███ █ █ █ ███████ █ █ █████████ █████████████████ █████
█ █  *█* * *█ █*█* * *█* *█ █* *█*█* * *█* *  █*        █* * * * * *█   █   █   █
█ ███ █████ █ █ █ ███ ███ █ █████ █████ █ █████ ███████████████████ █ █ █ ███ █ █
█* * *  █     █*█*█  * *█   █* *█*  █* *█* * * *█     █  * * * *█* *█ █ █ █   █ █
█ ███ █ █ ███ █ █ ███ █ ███ █ █ █ ███ █████████████ █████ █ █ █ █ ███ ███ █ █ █ █
█*█ █ █ █ █   █* *█ █ █* *█ █*█*█*█* *█* * *█* * *    █  *█ █ █*█* *█   █ █ █ █ █
█ █ █████ █ █ █ █ █ █████ █ █ █ █ █ ███ ███ █ ███ ███████ ███ █ ███ ███ █ █ █ █ █
█*  █   █ █ █ █ █   █    *█ █*█* *█*█* *█  * *█* *█* *  █*  █ █*█ █* *█   █ █ █ █
█ ███ █ █████████ █ █████ █ █ █████ █ █████████ █ █ █ ███ █████ █ ███ █ █████ █ █
█* * *█ █ █ █ █ █ █ █    *█ █* *█  * *█* * * *█*█ █*█* * *█    *█ █* *█ █     █ █
█████ ███ █ █ █ ███ █████ █ ███ █████ █ █████ █ ███ ███ █ ███ █ █ █ ███ ███ █████
█* * *  █    * *  █ █ █ █*█ █ █* * *█ █*█* * *█* *█*█ █ █ █   █*  █*█ █     █* *█
█ █ █████████ █ █████ █ █ █ █ █████ ███ █ ███████ █ █ █████████ ███ █ █ █████ █ █
█*█ █ █    * *█* * * *  █*█     █ █*█  *█* * * *█* *█* * * *█  *█* *█   █* * *█*█
█ ███ ███ █ █████████ ███ █████ █ █ ███ █ █████ ███ █ █████ ███ █ ███████ █████ █
█* *    █ █*  █  * *█*█  * * *█   █*█  *█ █* * *█   █* *  █* * *█* * *█* *█* * *█
███ ███████ █████ █ █ █████ █ █████ ███ █ █ ███████████ ███████ █████ █ █ █ █████
█* *█ █    *  █  *█* *█   █ █* * * *█* *█ █* *█* *  █ █*█* *█ █ █* *█* *█ █* * *█
█ ███ █████ █████ █████ ███████ █████ ███ ███ █ █ ███ █ █ █ █ ███ █ ███████ █ █ █
█*    █ █* *█* * *█    * *█ █       █*█ █ █  * *█* * * *█*█* *  █*█*█ █ █ █ █ █*█
█ █████ █ ███ ███ █████ █ █ ███ █ ███ █ █████████████████ ███ ███ █ █ █ █ █████ █
█*█* * *█*  █* *█ █ █* *█*█* *█ █   █*█   █ █* * * * *█  * *█* *█*█* *    █* * *█
█ █ ███ █ █████ █ █ █ ███ █ █ ███████ ███ █ █ ███ ███ █████ ███ █ ███ █████ ███ █
█* *█  * *  █  *█ █  *█* *█*█* * * * *█* * *█* *█ █* *█ █* *  █*█* *█*█* * *█   █
█ █ ███ █████ █ █████ █ ███ ███████ ███ ███ ███ ███ ███ █ █████ ███ █ █ █████ ███
█ █ █   █     █* * * *█* * *    █   █  * *█* * *  █* * * *  █  * * *█* *█       █
█████████████████████████████████████████ ███████████████████████████████████████
d_path [(20, 0), (19, 0), (19, 1), (20, 1), (21, 1), (21, 0), (22, 0), (23, 0), (23, 1), (22, 1), (22, 2), (23, 2), (24, 2), (25, 2), (26, 2), (26, 1), (25, 1), (25, 0), (26, 0), (27, 0), (28, 0), (28, 1), (29, 1), (29, 2), (28, 2), (28, 3), (28, 4), (29, 4), (29, 3), (30, 3), (30, 2), (31, 2), (31, 1), (31, 0), (32, 0), (33, 0), (33, 1), (32, 1), (32, 2), (32, 3), (32, 4), (33, 4), (33, 3), (33, 2), (34, 2), (34, 1), (34, 0), (35, 0), (35, 1), (36, 1), (37, 1), (37, 2), (38, 2), (39, 2), (39, 3), (39, 4), (38, 4), (37, 4), (37, 5), (38, 5), (39, 5), (39, 6), (39, 7), (38, 7), (38, 6), (37, 6), (36, 6), (36, 5), (35, 5), (35, 4), (34, 4), (34, 5), (33, 5), (32, 5), (32, 6), (33, 6), (33, 7), (33, 8), (34, 8), (34, 9), (33, 9), (33, 10), (32, 10), (32, 11), (33, 11), (33, 12), (32, 12), (31, 12), (30, 12), (29, 12), (28, 12), (28, 13), (28, 14), (28, 15), (28, 16), (28, 17), (28, 18), (29, 18), (29, 19), (28, 19), (27, 19), (27, 20), (28, 20), (29, 20), (29, 21), (30, 21), (30, 20), (31, 20), (32, 20), (32, 21), (32, 22), (32, 23), (33, 23), (33, 22), (34, 22), (35, 22), (35, 23), (36, 23), (37, 23), (38, 23), (38, 22), (39, 22), (39, 23), (39, 24), (38, 24), (37, 24), (36, 24), (36, 25), (36, 26), (37, 26), (38, 26), (38, 27), (39, 27), (39, 28), (39, 29), (39, 30), (38, 30), (38, 29), (38, 28), (37, 28), (37, 27), (36, 27), (36, 28), (35, 28), (35, 27), (35, 26), (34, 26), (34, 27), (33, 27), (32, 27), (32, 26), (31, 26), (31, 27), (30, 27), (29, 27), (28, 27), (28, 26), (28, 25), (27, 25), (26, 25), (26, 26), (27, 26), (27, 27), (26, 27), (25, 27), (25, 26), (25, 25), (24, 25), (23, 25), (23, 24), (22, 24), (22, 23), (22, 22), (22, 21), (21, 21), (20, 21), (19, 21), (19, 22), (18, 22), (18, 21), (18, 20), (19, 20), (20, 20), (21, 20), (22, 20), (23, 20), (23, 19), (23, 18), (22, 18), (22, 17), (21, 17), (20, 17), (20, 16), (19, 16), (19, 15), (18, 15), (18, 14), (19, 14), (20, 14), (21, 14), (22, 14), (22, 15), (23, 15), (23, 14), (24, 14), (25, 14), (25, 13), (24, 13), (23, 13), (23, 12), (23, 11), (22, 11), (21, 11), (20, 11), (20, 12), (21, 12), (21, 13), (20, 13), (19, 13), (18, 13), (17, 13), (17, 12), (18, 12), (19, 12), (19, 11), (18, 11), (18, 10), (17, 10), (17, 9), (17, 8), (18, 8), (18, 9), (19, 9), (19, 10), (20, 10), (21, 10), (21, 9), (22, 9), (22, 10), (23, 10), (24, 10), (24, 9), (23, 9), (23, 8), (23, 7), (24, 7), (24, 6), (25, 6), (25, 7), (25, 8), (25, 9), (26, 9), (26, 8), (27, 8), (28, 8), (28, 9), (28, 10), (28, 11), (29, 11), (30, 11), (31, 11), (31, 10), (31, 9), (31, 8), (31, 7), (31, 6), (31, 5), (30, 5), (29, 5), (29, 6), (28, 6), (27, 6), (26, 6), (26, 5), (27, 5), (27, 4), (27, 3), (26, 3), (25, 3), (24, 3), (24, 4), (23, 4), (23, 3), (22, 3), (22, 4), (21, 4), (21, 5), (22, 5), (23, 5), (23, 6), (22, 6), (21, 6), (20, 6), (20, 7), (21, 7), (22, 7), (22, 8), (21, 8), (20, 8), (19, 8), (19, 7), (19, 6), (19, 5), (19, 4), (18, 4), (18, 3), (18, 2), (18, 1), (17, 1), (16, 1), (15, 1), (14, 1), (14, 2), (13, 2), (13, 1), (13, 0), (12, 0), (11, 0), (11, 1), (12, 1), (12, 2), (12, 3), (11, 3), (11, 2), (10, 2), (10, 1), (10, 0), (9, 0), (8, 0), (7, 0), (7, 1), (7, 2), (6, 2), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5), (9, 5), (9, 4), (10, 4), (10, 5), (10, 6), (9, 6), (8, 6), (7, 6), (7, 7), (6, 7), (6, 6), (5, 6), (5, 5), (5, 4), (5, 3), (4, 3), (4, 2), (4, 1), (3, 1), (3, 2), (2, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (0, 5), (0, 6), (0, 7), (1, 7), (2, 7), (2, 8), (1, 8), (0, 8), (0, 9), (0, 10), (0, 11), (1, 11), (2, 11), (2, 12), (2, 13), (1, 13), (1, 14), (0, 14), (0, 15), (1, 15), (2, 15), (3, 15), (3, 16), (3, 17), (2, 17), (2, 18), (3, 18), (3, 19), (4, 19), (5, 19), (5, 18), (4, 18), (4, 17), (5, 17), (5, 16), (5, 15), (5, 14), (4, 14), (4, 13), (3, 13), (3, 12), (4, 12), (5, 12), (5, 13), (6, 13), (6, 14), (7, 14), (7, 13), (7, 12), (7, 11), (7, 10), (8, 10), (8, 11), (8, 12), (9, 12), (10, 12), (10, 11), (11, 11), (11, 10), (12, 10), (12, 9), (12, 8), (12, 7), (12, 6), (12, 5), (13, 5), (14, 5), (14, 4), (15, 4), (16, 4), (17, 4), (17, 5), (17, 6), (17, 7), (16, 7), (15, 7), (15, 8), (14, 8), (14, 9), (14, 10), (14, 11), (15, 11), (15, 10), (15, 9), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (15, 13), (15, 12), (14, 12), (14, 13), (13, 13), (12, 13), (12, 12), (11, 12), (11, 13), (10, 13), (9, 13), (9, 14), (9, 15), (8, 15), (8, 16), (8, 17), (7, 17), (7, 18), (7, 19), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (7, 24), (7, 25), (6, 25), (5, 25), (4, 25), (4, 24), (3, 24), (2, 24), (2, 23), (2, 22), (1, 22), (0, 22), (0, 23), (0, 24), (1, 24), (1, 25), (2, 25), (3, 25), (3, 26), (4, 26), (5, 26), (6, 26), (6, 27), (6, 28), (6, 29), (7, 29), (7, 28), (8, 28), (8, 27), (7, 27), (7, 26), (8, 26), (9, 26), (10, 26), (11, 26), (12, 26), (12, 27), (12, 28), (12, 29), (13, 29), (13, 30), (14, 30), (14, 29), (15, 29), (15, 28), (16, 28), (16, 29), (16, 30), (15, 30), (15, 31), (15, 32), (15, 33), (15, 34), (15, 35), (16, 35), (17, 35), (17, 36), (17, 37), (18, 37), (19, 37), (19, 38), (20, 38), (20, 39), (20, 40), (19, 40), (18, 40), (18, 41), (19, 41), (19, 42), (18, 42), (18, 43), (18, 44)]
my_maze.solution_path [(20, 0), (19, 0), (19, 1), (20, 1), (21, 1), (21, 0), (22, 0), (23, 0), (23, 1), (22, 1), (22, 2), (23, 2), (24, 2), (25, 2), (26, 2), (26, 1), (25, 1), (25, 0), (26, 0), (27, 0), (28, 0), (28, 1), (29, 1), (29, 2), (28, 2), (28, 3), (28, 4), (29, 4), (29, 3), (30, 3), (30, 2), (31, 2), (31, 1), (31, 0), (32, 0), (33, 0), (33, 1), (32, 1), (32, 2), (32, 3), (32, 4), (33, 4), (33, 3), (33, 2), (34, 2), (34, 1), (34, 0), (35, 0), (35, 1), (36, 1), (37, 1), (37, 2), (38, 2), (39, 2), (39, 3), (39, 4), (38, 4), (37, 4), (37, 5), (38, 5), (39, 5), (39, 6), (39, 7), (38, 7), (38, 6), (37, 6), (36, 6), (36, 5), (35, 5), (35, 4), (34, 4), (34, 5), (33, 5), (32, 5), (32, 6), (33, 6), (33, 7), (33, 8), (34, 8), (34, 9), (33, 9), (33, 10), (32, 10), (32, 11), (33, 11), (33, 12), (32, 12), (31, 12), (30, 12), (29, 12), (28, 12), (28, 13), (28, 14), (28, 15), (28, 16), (28, 17), (28, 18), (29, 18), (29, 19), (28, 19), (27, 19), (27, 20), (28, 20), (29, 20), (29, 21), (30, 21), (30, 20), (31, 20), (32, 20), (32, 21), (32, 22), (32, 23), (33, 23), (33, 22), (34, 22), (35, 22), (35, 23), (36, 23), (37, 23), (38, 23), (38, 22), (39, 22), (39, 23), (39, 24), (38, 24), (37, 24), (36, 24), (36, 25), (36, 26), (37, 26), (38, 26), (38, 27), (39, 27), (39, 28), (39, 29), (39, 30), (38, 30), (38, 29), (38, 28), (37, 28), (37, 27), (36, 27), (36, 28), (35, 28), (35, 27), (35, 26), (34, 26), (34, 27), (33, 27), (32, 27), (32, 26), (31, 26), (31, 27), (30, 27), (29, 27), (28, 27), (28, 26), (28, 25), (27, 25), (26, 25), (26, 26), (27, 26), (27, 27), (26, 27), (25, 27), (25, 26), (25, 25), (24, 25), (23, 25), (23, 24), (22, 24), (22, 23), (22, 22), (22, 21), (21, 21), (20, 21), (19, 21), (19, 22), (18, 22), (18, 21), (18, 20), (19, 20), (20, 20), (21, 20), (22, 20), (23, 20), (23, 19), (23, 18), (22, 18), (22, 17), (21, 17), (20, 17), (20, 16), (19, 16), (19, 15), (18, 15), (18, 14), (19, 14), (20, 14), (21, 14), (22, 14), (22, 15), (23, 15), (23, 14), (24, 14), (25, 14), (25, 13), (24, 13), (23, 13), (23, 12), (23, 11), (22, 11), (21, 11), (20, 11), (20, 12), (21, 12), (21, 13), (20, 13), (19, 13), (18, 13), (17, 13), (17, 12), (18, 12), (19, 12), (19, 11), (18, 11), (18, 10), (17, 10), (17, 9), (17, 8), (18, 8), (18, 9), (19, 9), (19, 10), (20, 10), (21, 10), (21, 9), (22, 9), (22, 10), (23, 10), (24, 10), (24, 9), (23, 9), (23, 8), (23, 7), (24, 7), (24, 6), (25, 6), (25, 7), (25, 8), (25, 9), (26, 9), (26, 8), (27, 8), (28, 8), (28, 9), (28, 10), (28, 11), (29, 11), (30, 11), (31, 11), (31, 10), (31, 9), (31, 8), (31, 7), (31, 6), (31, 5), (30, 5), (29, 5), (29, 6), (28, 6), (27, 6), (26, 6), (26, 5), (27, 5), (27, 4), (27, 3), (26, 3), (25, 3), (24, 3), (24, 4), (23, 4), (23, 3), (22, 3), (22, 4), (21, 4), (21, 5), (22, 5), (23, 5), (23, 6), (22, 6), (21, 6), (20, 6), (20, 7), (21, 7), (22, 7), (22, 8), (21, 8), (20, 8), (19, 8), (19, 7), (19, 6), (19, 5), (19, 4), (18, 4), (18, 3), (18, 2), (18, 1), (17, 1), (16, 1), (15, 1), (14, 1), (14, 2), (13, 2), (13, 1), (13, 0), (12, 0), (11, 0), (11, 1), (12, 1), (12, 2), (12, 3), (11, 3), (11, 2), (10, 2), (10, 1), (10, 0), (9, 0), (8, 0), (7, 0), (7, 1), (7, 2), (6, 2), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5), (9, 5), (9, 4), (10, 4), (10, 5), (10, 6), (9, 6), (8, 6), (7, 6), (7, 7), (6, 7), (6, 6), (5, 6), (5, 5), (5, 4), (5, 3), (4, 3), (4, 2), (4, 1), (3, 1), (3, 2), (2, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (0, 5), (0, 6), (0, 7), (1, 7), (2, 7), (2, 8), (1, 8), (0, 8), (0, 9), (0, 10), (0, 11), (1, 11), (2, 11), (2, 12), (2, 13), (1, 13), (1, 14), (0, 14), (0, 15), (1, 15), (2, 15), (3, 15), (3, 16), (3, 17), (2, 17), (2, 18), (3, 18), (3, 19), (4, 19), (5, 19), (5, 18), (4, 18), (4, 17), (5, 17), (5, 16), (5, 15), (5, 14), (4, 14), (4, 13), (3, 13), (3, 12), (4, 12), (5, 12), (5, 13), (6, 13), (6, 14), (7, 14), (7, 13), (7, 12), (7, 11), (7, 10), (8, 10), (8, 11), (8, 12), (9, 12), (10, 12), (10, 11), (11, 11), (11, 10), (12, 10), (12, 9), (12, 8), (12, 7), (12, 6), (12, 5), (13, 5), (14, 5), (14, 4), (15, 4), (16, 4), (17, 4), (17, 5), (17, 6), (17, 7), (16, 7), (15, 7), (15, 8), (14, 8), (14, 9), (14, 10), (14, 11), (15, 11), (15, 10), (15, 9), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (15, 13), (15, 12), (14, 12), (14, 13), (13, 13), (12, 13), (12, 12), (11, 12), (11, 13), (10, 13), (9, 13), (9, 14), (9, 15), (8, 15), (8, 16), (8, 17), (7, 17), (7, 18), (7, 19), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (7, 24), (7, 25), (6, 25), (5, 25), (4, 25), (4, 24), (3, 24), (2, 24), (2, 23), (2, 22), (1, 22), (0, 22), (0, 23), (0, 24), (1, 24), (1, 25), (2, 25), (3, 25), (3, 26), (4, 26), (5, 26), (6, 26), (6, 27), (6, 28), (6, 29), (7, 29), (7, 28), (8, 28), (8, 27), (7, 27), (7, 26), (8, 26), (9, 26), (10, 26), (11, 26), (12, 26), (12, 27), (12, 28), (12, 29), (13, 29), (13, 30), (14, 30), (14, 29), (15, 29), (15, 28), (16, 28), (16, 29), (16, 30), (15, 30), (15, 31), (15, 32), (15, 33), (15, 34), (15, 35), (16, 35), (17, 35), (17, 36), (17, 37), (18, 37), (19, 37), (19, 38), (20, 38), (20, 39), (20, 40), (19, 40), (18, 40), (18, 41), (19, 41), (19, 42), (18, 42), (18, 43), (18, 44)]
a_path [(20, 0), (19, 0), (19, 1), (20, 1), (21, 1), (21, 0), (22, 0), (23, 0), (23, 1), (22, 1), (22, 2), (23, 2), (24, 2), (25, 2), (26, 2), (26, 1), (25, 1), (25, 0), (26, 0), (27, 0), (28, 0), (28, 1), (29, 1), (29, 2), (28, 2), (28, 3), (28, 4), (29, 4), (29, 3), (30, 3), (30, 2), (31, 2), (31, 1), (31, 0), (32, 0), (33, 0), (33, 1), (32, 1), (32, 2), (32, 3), (32, 4), (33, 4), (33, 3), (33, 2), (34, 2), (34, 1), (34, 0), (35, 0), (35, 1), (36, 1), (37, 1), (37, 2), (38, 2), (39, 2), (39, 3), (39, 4), (38, 4), (37, 4), (37, 5), (38, 5), (39, 5), (39, 6), (39, 7), (38, 7), (38, 6), (37, 6), (36, 6), (36, 5), (35, 5), (35, 4), (34, 4), (34, 5), (33, 5), (32, 5), (32, 6), (33, 6), (33, 7), (33, 8), (34, 8), (34, 9), (33, 9), (33, 10), (32, 10), (32, 11), (33, 11), (33, 12), (32, 12), (31, 12), (30, 12), (29, 12), (28, 12), (28, 13), (28, 14), (28, 15), (28, 16), (28, 17), (28, 18), (29, 18), (29, 19), (28, 19), (27, 19), (27, 20), (28, 20), (29, 20), (29, 21), (30, 21), (30, 20), (31, 20), (32, 20), (32, 21), (32, 22), (32, 23), (33, 23), (33, 22), (34, 22), (35, 22), (35, 23), (36, 23), (37, 23), (38, 23), (38, 22), (39, 22), (39, 23), (39, 24), (38, 24), (37, 24), (36, 24), (36, 25), (36, 26), (37, 26), (38, 26), (38, 27), (39, 27), (39, 28), (39, 29), (39, 30), (38, 30), (38, 29), (38, 28), (37, 28), (37, 27), (36, 27), (36, 28), (35, 28), (35, 27), (35, 26), (34, 26), (34, 27), (33, 27), (32, 27), (32, 26), (31, 26), (31, 27), (30, 27), (29, 27), (28, 27), (28, 26), (28, 25), (27, 25), (26, 25), (26, 26), (27, 26), (27, 27), (26, 27), (25, 27), (25, 26), (25, 25), (24, 25), (23, 25), (23, 24), (22, 24), (22, 23), (22, 22), (22, 21), (21, 21), (20, 21), (19, 21), (19, 22), (18, 22), (18, 21), (18, 20), (19, 20), (20, 20), (21, 20), (22, 20), (23, 20), (23, 19), (23, 18), (22, 18), (22, 17), (21, 17), (20, 17), (20, 16), (19, 16), (19, 15), (18, 15), (18, 14), (19, 14), (20, 14), (21, 14), (22, 14), (22, 15), (23, 15), (23, 14), (24, 14), (25, 14), (25, 13), (24, 13), (23, 13), (23, 12), (23, 11), (22, 11), (21, 11), (20, 11), (20, 12), (21, 12), (21, 13), (20, 13), (19, 13), (18, 13), (17, 13), (17, 12), (18, 12), (19, 12), (19, 11), (18, 11), (18, 10), (17, 10), (17, 9), (17, 8), (18, 8), (18, 9), (19, 9), (19, 10), (20, 10), (21, 10), (21, 9), (22, 9), (22, 10), (23, 10), (24, 10), (24, 9), (23, 9), (23, 8), (23, 7), (24, 7), (24, 6), (25, 6), (25, 7), (25, 8), (25, 9), (26, 9), (26, 8), (27, 8), (28, 8), (28, 9), (28, 10), (28, 11), (29, 11), (30, 11), (31, 11), (31, 10), (31, 9), (31, 8), (31, 7), (31, 6), (31, 5), (30, 5), (29, 5), (29, 6), (28, 6), (27, 6), (26, 6), (26, 5), (27, 5), (27, 4), (27, 3), (26, 3), (25, 3), (24, 3), (24, 4), (23, 4), (23, 3), (22, 3), (22, 4), (21, 4), (21, 5), (22, 5), (23, 5), (23, 6), (22, 6), (21, 6), (20, 6), (20, 7), (21, 7), (22, 7), (22, 8), (21, 8), (20, 8), (19, 8), (19, 7), (19, 6), (19, 5), (19, 4), (18, 4), (18, 3), (18, 2), (18, 1), (17, 1), (16, 1), (15, 1), (14, 1), (14, 2), (13, 2), (13, 1), (13, 0), (12, 0), (11, 0), (11, 1), (12, 1), (12, 2), (12, 3), (11, 3), (11, 2), (10, 2), (10, 1), (10, 0), (9, 0), (8, 0), (7, 0), (7, 1), (7, 2), (6, 2), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5), (9, 5), (9, 4), (10, 4), (10, 5), (10, 6), (9, 6), (8, 6), (7, 6), (7, 7), (6, 7), (6, 6), (5, 6), (5, 5), (5, 4), (5, 3), (4, 3), (4, 2), (4, 1), (3, 1), (3, 2), (2, 2), (1, 2), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (0, 5), (0, 6), (0, 7), (1, 7), (2, 7), (2, 8), (1, 8), (0, 8), (0, 9), (0, 10), (0, 11), (1, 11), (2, 11), (2, 12), (2, 13), (1, 13), (1, 14), (0, 14), (0, 15), (1, 15), (2, 15), (3, 15), (3, 16), (3, 17), (2, 17), (2, 18), (3, 18), (3, 19), (4, 19), (5, 19), (5, 18), (4, 18), (4, 17), (5, 17), (5, 16), (5, 15), (5, 14), (4, 14), (4, 13), (3, 13), (3, 12), (4, 12), (5, 12), (5, 13), (6, 13), (6, 14), (7, 14), (7, 13), (7, 12), (7, 11), (7, 10), (8, 10), (8, 11), (8, 12), (9, 12), (10, 12), (10, 11), (11, 11), (11, 10), (12, 10), (12, 9), (12, 8), (12, 7), (12, 6), (12, 5), (13, 5), (14, 5), (14, 4), (15, 4), (16, 4), (17, 4), (17, 5), (17, 6), (17, 7), (16, 7), (15, 7), (15, 8), (14, 8), (14, 9), (14, 10), (14, 11), (15, 11), (15, 10), (15, 9), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (15, 13), (15, 12), (14, 12), (14, 13), (13, 13), (12, 13), (12, 12), (11, 12), (11, 13), (10, 13), (9, 13), (9, 14), (9, 15), (8, 15), (8, 16), (8, 17), (7, 17), (7, 18), (7, 19), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (7, 24), (7, 25), (6, 25), (5, 25), (4, 25), (4, 24), (3, 24), (2, 24), (2, 23), (2, 22), (1, 22), (0, 22), (0, 23), (0, 24), (1, 24), (1, 25), (2, 25), (3, 25), (3, 26), (4, 26), (5, 26), (6, 26), (6, 27), (6, 28), (6, 29), (7, 29), (7, 28), (8, 28), (8, 27), (7, 27), (7, 26), (8, 26), (9, 26), (10, 26), (11, 26), (12, 26), (12, 27), (12, 28), (12, 29), (13, 29), (13, 30), (14, 30), (14, 29), (15, 29), (15, 28), (16, 28), (16, 29), (16, 30), (15, 30), (15, 31), (15, 32), (15, 33), (15, 34), (15, 35), (16, 35), (17, 35), (17, 36), (17, 37), (18, 37), (19, 37), (19, 38), (20, 38), (20, 39), (20, 40), (19, 40), (18, 40), (18, 41), (19, 41), (19, 42), (18, 42), (18, 43), (18, 44)]
Dijkstra took 15.803 ms with Method.STACK
A* took 19.219 ms with Method.STACK
Maze Size 40
█████████████████████████████████████████████████████ ███████████████████████████
█ █       █ █   █ █   █ █ █       █ █     █   █   █  *█   █   █       █       █ █
█ ███████ █ █ ███ ███ █ █ ███ ███ █ █ █ ███ █████ ███ ███ ███ █ █ █████ █████ █ █
█ █ █   █   █ █   █     █   █ █   █ █ █ █   █       █*█ █ █ █   █           █   █
█ █ ███ ███ █ ███ █████ ███ █████ █ ███ ███ █████ █ █ █ █ █ ███ ███████████████ █
█   █   █     █ █   █ █ █ █ █ █ █ █     █ █   █ █ █  * *█           █       █ █ █
█ █████ ███ ███ ███ █ █ █ █ █ █ █ █ █████ █ ███ █████ █ █ ███████ █ █ █ █████ ███
█ █   █ █   █ █ █     █       █ █ █   █       █     █ █*█ █     █ █ █ █ █   █   █
█ █ ███ ███ █ █ █ ███████ █████ █ █ ███ ███ █████ █████ █ █ █████████ ███ █ █ ███
█   █ █ █         █   █         █   █ █ █   █ █ █   █  *  █     █   █   █ █ █ █ █
███ █ █ █████████ █ █████████ ███ ███ ███ ███ █ █ █████ █ █ █████ ███ ███ ███ █ █
█     █   █ █   █   █ █   █ █ █   █   █ █     █ █   █* *█   █   █       █ █ █ █ █
███ █ █ ███ █ █ ███ █ █ ███ █ ███ █ ███ █████ █ ███ █ █████████ ███ █████ █ █ █ █
█ █ █ █ █   █ █ █ █ █ █   █       █     █   █     █  *        █       █ █   █ █ █
█ ███ █ ███ ███ █ █ █ ███ ███ █████ ███████ █████ ███ █████████ █ █████ ███ █ █ █
█     █ █   █   █   █ █ █ █ █ █ █     █   █   █   █  *      █   █ █   █     █   █
███ ███ ███ ███ █ ███ █ █ █ █ █ ███ █████ ███ ███ ███ █ █ █████ █████ █ ███ ███ █
█     █ █ █   █   █ █ █       █   █ █       █     █* *█ █ █               █ █ █ █
███ ███ █ ███ ███ █ █ ███████ ███ █ ███████ █ █████ █████████ █████████████ █ █ █
█ █ █ █       █   █ █       █ █         █       █  *    █   █ █       █   █ █ █ █
█ █ █ ███████ ███ █ ███████ █ ███ █████████ ███████ █████ █ █ █ █ █████ █████ █ █
█     █ █     █ █   █   █ █ █ █       █ █   █   █ █*█ █ █ █ █   █     █ █       █
███ ███ █████ █ ███ ███ █ █ █ ███ █████ █ ███ ███ █ █ █ ███ █ ███ █████ ███████ █
█   █   █   █ █       █   █     █   █   █ █ █   █ █*█     █ █   █ █       █     █
█ █ ███ █ █ █ █ █████████ ███ █ █ █████ █ █ █ ███ █ █ █████ █ █████ ███████████ █
█ █ █ █ █ █     █ █       █ █ █ █ █   █ █     █ █ █*    █     █ █         █   █ █
███ █ █ ███████ █ █████ ███ █ ███ ███ █ ███ ███ █ █ ███████ ███ █ ███████████ █ █
█                 █ █ █   █ █ █ █   █     █   █ █  *  █ █ █ █ █   █     █   █   █
███████ ███ █ █████ █ █ ███ █ █ █ ███ ███ █ ███ ███ █ █ █ █ █ █ ███ ███████ █ █ █
█     █ █   █ █ █ █ █   █     █   █     █ █   █ █  *█ █ █     █       █   █ █ █ █
█████ ███████ █ █ █ █ ███ ███ ███ ███ ███████ █ ███ ███ ███ ███ █ █████ ███ █ ███
█ █ █   █             █ █ █       █   █     █     █*    █   █ █ █ █             █
█ █ ███ █████████ █████ ███████ ███ ███ █████ ███ █ ███████ █ █ █████████ ███████
█     █ █ █ █ █     █   █   █     █ █ █ █   █ █  * *  █   █         █ █ █   █   █
█ █ ███ █ █ █ █ █ █ █ █ █ █████ ███ █ █ █ ███ ███ ███████ █ █████████ █ █ █ ███ █
█ █     █   █ █ █ █ █ █   █   █     █ █     █ █ █*█     █     █   █   █ █ █ █ █ █
█████ █████ █ ███ ███████ █ █████ ███ ███ █████ █ █████ █ ███ █ ███ ███ ███ █ █ █
█ █       █ █ █   █   █     █   █   █ █     █ █ █* * *█     █ █ █   █ █     █   █
█ ███ █████ █ ███ ███ █ █████ ███ ███ █ █████ █ █ █ █ █ ███████ █ ███ █████ █ ███
█ █   █ █ █ █     █ █   █ █ █         █     █   █ █ █* *      █     █       █ █ █
█ ███ █ █ █ █████ █ ███ █ █ █████ █████ ███████ ███████ ███████ ███████████ █ █ █
█ █ █   █       █ █   █ █ █   █ █   █ █ █ █ █     █ █* *      █     █ █     █   █
█ █ █ █ █████ ███ ███ █ █ ███ █ █ ███ █ █ █ █ ███ █ █ ███ █ ███████ █ █████ █ ███
█   █ █ █ █       █   █     █ █   █   █   █ █   █    *  █ █   █   █ █ █   █ █   █
███ ███ █ ███████ ███ █ █████ █ █████ █ ███ ███████ █ █████ █ █ █ █ █ █ ███ ███ █
█   █ █   █ █ █             █     █   █   █     █ █ █*█ █ █ █ █ █     █     █   █
███ █ ███ █ █ █████ ███████████ ███ ███ ███ █████ ███ █ █ █████ █████ █ ███████ █
█ █       █ █     █   █   █ █   █   █     █   █   █ █*    █ █ █ █     █ █ █   █ █
█ ███████ █ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ █ █ █ █ █ ███████ █ █ ███ █ █
█ █     █ █ █ █   █ █ █   █ █     █ █ █ █       █    *█ █ █ █   █ █ █   █       █
█ █████ █ █ █ ███ █ █ █ ███ ███ █ █ █ █ █ ███████████ █████ ███ █ █ ███ █████ ███
█     █     █ █ █       █   █   █ █ █     █   █ █ █* *█ █   █   █ █ █ █   █ █   █
█████ █ █ ███ █ ███ ███████ ███ ███ ███ ███ ███ █ █ ███ ███ █ ███ █ █ █ ███ █ ███
█       █     █   █   █     █ █   █ █   █     █   █*█     █ █   █           █   █
█ ███████ ███ ███ █ ███████ █ █ ███ ███ █ ███ ███ █ █ █████ █ ███████ ███████ ███
█   █     █       █     █   █     █     █   █ █ █  *█ █   █   █   █     █ █     █
███████ █ █████ █ █ ███████ █████ █ █ ███ █████ ███ █ █ ███ ███ █████ ███ █ ███ █
█   █   █ █   █ █     █ █ █ █ █   █ █ █ █         █* * *█ █ █ █ █     █ █ █ █   █
███ ███ █████ █████ ███ █ █ █ █ ███ ███ ███ ███████████ █ █ █ █ █████ █ █ █ ███ █
█   █   █     █     █ █   █     █ █ █ █   █     █ █  * *█ █ █   █   █         █ █
███ █████████ █ ███ █ █ ███████ █ █ █ ███ ███ ███ ███ ███ █ █ █████ ███ ███ █████
█   █   █     █ █ █         █ █ █ █   █ █ █ █       █*    █ █   █ █ █   █ █ █   █
███ ███ █████ ███ █ ███ █████ █ █ ███ █ █ █ █ ███████ █████ █ ███ █ █ ███ ███ █ █
█     █     █       █ █     █ █ █ █ █ █ █ █ █       █* *  █   █   █   █       █ █
███ █████ █ █████████ █ █████ █ █ █ █ █ █ █ █ █████████ ███ ███ ███ █████ ███████
█     █   █ █ █   █ █   █ █ █   █     █ █   █ █       █* *█   █ █   █   █       █
███ █ █████ █ ███ █ ███ █ █ ███ ███ ███ █ ███ █ █████████ █ ███ ███ █ ███ ███████
█ █ █   █ █ █ █     █   █     █   █   █         █   █ █* *    █ █     █     █   █
█ ███ █ █ █ █ ███ █████ █████ ███ ███ ███ ███████ ███ █ ███████ █ █████ █████ ███
█ █ █ █   █   █ █ █   █ █     █ █ █   █ █ █ █ █ █* * * *█ █   █ █           █ █ █
█ █ █████ █ ███ █ ███ █ █████ █ █ █ ███ █ █ █ █ █ ███████ █ ███ ███ █ ███████ █ █
█ █ █   █     █ █ █     █   █   █   █   █   █ █* *  █ █       █   █ █ █       █ █
█ █ ███ ███ ███ █ ███ ███ ███ ███ ███ ███ ███ █ █████ █ ███████ ███ ███ ███████ █
█ █         █ █ █       █ █   █   █   █ █     █*  █ █ █           █ █ █ █ █     █
█ ███████ █ █ █ ███████ █ ███ █ ███ █ █ █ ███ █ ███ █ █ █ █████████ █ █ █ █ █████
█   █ █   █ █       █     █ █   █ █ █     █ █ █*█   █   █ █ █ █ █ █     █   █   █
███ █ █████ █████ ███████ █ ███ █ █ █ █████ ███ █ ███ █████ █ █ █ █ █████ ███ ███
█                         █     █ █ █ █ █ █ █ █*█ █     █     █ █ █ █           █
███████████ █████████ ███ █████ █ ███ █ █ █ █ █ █ █ █████ █████ █ █ █ █ ███████ █
█               █   █ █   █     █ █   █ █     █*█     █ █ █ █         █ █     █ █
█████ ███ ███ █████ █████ █████ █ █ ███ ███ ███ █ █████ █ █ ███ █ █ █ █████ █████
█       █ █   █   █     █           █     █ █ █*█ █             █ █ █   █ █     █
███ ███ ███ █████ ███ ███ ███████ ███ █████ █ █ █ █ ███████ █ █████ ███ █ █ █████
█   █   █ █ █               █     █ █   █ █* * *        █ █ █   █     █         █
███ ███ █ ███ █ █ ███ ███ ███ ███ █ █ ███ █ ███ █ █████ █ █ ███████ █ ███ ███ █ █
█     █ █     █ █ █   █   █   █         █  *█ █ █     █   █ █   █   █   █ █   █ █
███ █ ███ █ ███ ███ ███ █ ███████ █ █ █████ █ █████ ███ █████ █ ███ █ ███ ███████
█   █   █ █ █   █     █ █ █ █     █ █    * *      █ █         █   █ █   █   █   █
█ █████ █ █ █ █ ███████████ █ █ █ ███ ███ ███ ███ █████ █ ███ █ █ █ ███ █████ █ █
█     █ █ █ █ █ █             █ █ █     █*  █   █   █   █   █ █ █ █   █       █ █
█████████████████████████████████████████ ███████████████████████████████████████
d_path [(20, 0), (20, 1), (21, 1), (21, 2), (21, 3), (22, 3), (23, 3), (23, 4), (23, 5), (23, 6), (23, 7), (23, 8), (23, 9), (24, 9), (24, 10), (25, 10), (26, 10), (27, 10), (27, 11), (28, 11), (28, 12), (27, 12), (27, 13), (26, 13), (26, 14), (26, 15), (27, 15), (27, 16), (26, 16), (25, 16), (25, 17), (25, 18), (25, 19), (26, 19), (26, 20), (26, 21), (26, 22), (26, 23), (26, 24), (27, 24), (27, 25), (26, 25), (26, 26), (25, 26), (24, 26), (24, 27), (24, 28), (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (26, 36), (26, 37), (26, 38), (26, 39), (27, 39), (27, 40), (27, 41), (27, 42), (26, 42), (26, 43), (26, 44)]
my_maze.solution_path [(20, 0), (20, 1), (21, 1), (21, 2), (21, 3), (22, 3), (23, 3), (23, 4), (23, 5), (23, 6), (23, 7), (23, 8), (23, 9), (24, 9), (24, 10), (25, 10), (26, 10), (27, 10), (27, 11), (28, 11), (28, 12), (27, 12), (27, 13), (26, 13), (26, 14), (26, 15), (27, 15), (27, 16), (26, 16), (25, 16), (25, 17), (25, 18), (25, 19), (26, 19), (26, 20), (26, 21), (26, 22), (26, 23), (26, 24), (27, 24), (27, 25), (26, 25), (26, 26), (25, 26), (24, 26), (24, 27), (24, 28), (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (26, 36), (26, 37), (26, 38), (26, 39), (27, 39), (27, 40), (27, 41), (27, 42), (26, 42), (26, 43), (26, 44)]
a_path [(20, 0), (20, 1), (21, 1), (21, 2), (21, 3), (22, 3), (23, 3), (23, 4), (23, 5), (23, 6), (23, 7), (23, 8), (23, 9), (24, 9), (24, 10), (25, 10), (26, 10), (27, 10), (27, 11), (28, 11), (28, 12), (27, 12), (27, 13), (26, 13), (26, 14), (26, 15), (27, 15), (27, 16), (26, 16), (25, 16), (25, 17), (25, 18), (25, 19), (26, 19), (26, 20), (26, 21), (26, 22), (26, 23), (26, 24), (27, 24), (27, 25), (26, 25), (26, 26), (25, 26), (24, 26), (24, 27), (24, 28), (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (25, 35), (25, 36), (26, 36), (26, 37), (26, 38), (26, 39), (27, 39), (27, 40), (27, 41), (27, 42), (26, 42), (26, 43), (26, 44)]
███████████ █████████████████████████████████████████████████████████████████████
█ █ █     █*█ █         █   █ █   █     █   █ █ █   █ █     █ █ █ █   █   █   █ █
█ █ █████ █ █ █████ ███████ █ █ ███ ███████ █ █ █ ███ █ █████ █ █ ███ █ █ ███ █ █
█ █ █    * *█   █     █ █   █   █   █ █ █           █     █     █       █ █ █   █
█ █ █████ █████ ███ ███ ███ █ █ █ ███ █ █████ ███████ █████ ███████████ ███ █ ███
█ █   █  *  █         █   █   █ █   █ █   █   █ █       █ █ █ █   █   █   █   █ █
█ ███ ███ █████ █ ███████ █ ███ ███ █ ███ ███ █ █ █ ███ █ █ █ ███ ███ █ ███ ███ █
█        *█ █ █ █ █   █ █ █ █ █ █ █ █ █ █ █     █ █   █ █   █ █ █   █ █     █   █
███ ███ █ █ █ ███ ███ █ █ █ █ ███ █ █ █ █ █████ █ ███████ █ █ █ █ ███ █ ███ █ ███
█   █   █*█   █         █   █ █ █         █ █ █   █     █ █   █   █   █ █ █   █ █
█████████ ███ ███████ ███ ███ █ █ ███ █████ █ █ ███ █████ █████ █████ █ █ █████ █
█   █    *█ █ █ █ █     █   █ █ █ █ █ █ █             █   █ █ █   █         █ █ █
███ █████ █ █ █ █ █████ █ ███ █ ███ █ █ █ █████████████ ███ █ ███ ███ ███████ █ █
█ █ █   █* * * * *█ █     █ █ █ █   █       █   █   █     █       █ █   █     █ █
█ █ ███ █████ ███ █ ███ ███ █ █ █ █████ █ ███ ███ ███ █████████ ███ █ ███ █████ █
█       █ █ █ █  * *█ █   █ █       █   █ █ █ █   █         █   █         █ █   █
█████ ███ █ ███████ █ █ ███ ███ ███████ ███ █ ███ ███ █████████ ███ █ █████ ███ █
█   █   █   █     █*█         █     █ █ █ █ █ █   █ █   █   █   █   █         █ █
███ █ █████ █████ █ ███ █████ ███ ███ █ █ █ █ ███ █ █ ███ █████ ███████ ███████ █
█     █ █     █ █  *█   █   █ █       █ █   █   █ █   █           █ █       █ █ █
█ █ █ █ █ █████ ███ █ █████ █████ ███ █ ███ █ ███ ███ █ ███ ███████ ███ █████ █ █
█ █ █ █ █   █   █ █*█     █ █   █   █ █   █ █ █ █   █   █ █ █ █ █ █   █       █ █
█████ █ ███ ███ █ █ █ █████ ███ █ █ ███ █ █ █ █ █ █ █ ███ ███ █ █ █ ███ ███████ █
█         █ █     █* *    █     █ █ █ █ █ █       █ █     █ █     █   █ █   █   █
█████ █ ███ █ █ █████ █ █ ███ ███ ███ ███ █ █ ███████ █████ █ ███████ █ ███ ███ █
█     █   █ █ █ █ █  *█ █   █ █ █ █     █ █ █     █ █             █   █ █   █ █ █
█ █████ ███ ███ █ ███ ███████ █ █ █████ █ ███ █████ █ █████████ ███ ███ ███ █ █ █
█ █   █ █   █   █   █*█ █ █ █ █   █ █ █   █   █         █     █ █           █   █
█████ █ ███ ███ █ ███ █ █ █ █ █ ███ █ █ █████ ███ ███████ █████████ █████████ █ █
█ █       █ █      * *█ █ █ █     █ █   █ █   █ █   █ █     █ █ █             █ █
█ ███ █████ ███████ ███ █ █ ███ ███ ███ █ █ ███ █ ███ █ █████ █ █ ███████████████
█   █         █ █  * * * *█       █ █       █ █     █     █   █ █       █       █
███ █████████ █ █████████ ███ █████ ███ ███ █ ███ █████ ███ ███ █ ███████ ███████
█ █     █     █   █   █ █* * *  █ █   █ █           █ █ █     █     █           █
█ █████ █████ █ ███ █ █ ███ █ ███ █ █ █████ █████████ █ █ █ █████ ███ ███████████
█             █   █ █     █ █* *    █ █   █     █         █ █   █ █       █ █   █
█████ ███ ███ █ ███████ █ ███ █ █████████ ███ ███ ███████ ███ ███ ███ █████ █ █ █
█     █   █       █ █   █   █ █*    █ █   █   █     █ █ █ █ █   █ █   █   █   █ █
███ ███ █████ ███ █ ███ █████ █ ███ █ ███ ███ ███ █ █ █ ███ ███ █ ███ █ ███ █████
█ █ █   █     █             █ █*█ █ █ █ █ █   █ █ █     █ █ █                   █
█ ███████████████████ █████████ █ ███ █ █ ███ █ █ ███████ █ ███ ███ █ ███ █ █ ███
█             █       █ █ █   █*  █ █   █     █     █   █ █ █ █   █ █   █ █ █ █ █
███ █████████ █ █████ █ █ ███ █ ███ █ █████ █ █ ███ █ ███ █ █ █ ███████████████ █
█   █             █ █       █ █*█   █ █   █ █   █ █ █   █   █   █ █ █ █   █ █   █
███████ ███████ █ █ ███████ █ █ ███ █ ███ ███ █ █ ███ ███ █████ █ █ █ █ ███ █ ███
█ █ █   █       █ █ █   █   █ █*█ █     █     █ █ █ █   █ █ █ █       █ █       █
█ █ █ █████████████ ███ ███ █ █ █ █ █████ █ █ ███ █ █ ███ █ █ █ ███████ █ ███████
█   █ █ █   █ █         █ █* * *█     █ █ █ █ █ █ █   █     █       █         █ █
█ █████ █ ███ ███████ █ █ █ ███████ ███ ███ ███ █ █ ███ ███████ ███████ █ █████ █
█ █     █ █         █ █  * *█ █         █     █     █     █     █ █ █   █   █   █
█ █████ █ █ ███ ███ █████ ███ █ █ █ ███████ ███ ███ █ █████ █████ █ ███ ███████ █
█       █ █ █ █ █ █   █  *█ █   █ █ █ █ █ █ █   █ █ █   █       █       █   █   █
███████ █ ███ ███ █ █████ █ ███████ █ █ █ █ █ ███ ███ ███████ ███ ███████ ███ ███
█ █ █   █   █ █ █   █    *  █   █   █ █ █ █ █     █   █   █   █             █   █
█ █ ███ ███ █ █ █ █ ███ █ ███ █████ █ █ █ █ █ ███ █ █████ █ █ █ █████████████ ███
█   █ █ █   █ █ █ █ █ █ █* * *█   █ █   █ █ █   █   █ █     █   █ █ █ █ █ █ █   █
███ █ █ ███ █ █ ███ █ ███████ █ ███ ███ █ █ █ ███████ ███ ███████ █ █ █ █ █ ███ █
█ █ █   █       █ █     █   █* *  █ █     █ █   █ █ █ █ █           █ █ █     █ █
█ █ █ █████ ███ █ ███ ███ █████ █ █ █ ███ █ █ ███ █ █ █ █ ███ ███████ █ ███ ███ █
█   █   █   █ █ █   █       █* *█ █ █ █         █     █ █   █   █ █             █
█ █ █ █████ █ █ ███ ███████ █ █████ █████ ███████ █████ █ ███████ ███ ███████████
█ █ █     █ █ █ █ █ █ █ █   █*    █ █ █ █ █   █ █       █     █ █   █     █ █   █
███ █████ ███ █ █ █ █ █ ███ █ ███ █ █ █ █ █ ███ ███████ █ █████ █ █████ ███ █ ███
█       █ █ █   █       █    *  █ █   █   █ █   █ █ █ █   █   █   █ █   █ █     █
█████ ███ █ ███ █ █ █ █████ █ ███████ ███ █ █ ███ █ █ █ █ ███ █ ███ █ ███ █ ███ █
█ █       █ █   █ █ █   █   █* * * *  █ █         █   █ █       █   █ █       █ █
█ █████ ███ ███ █████ █████████████ ███ █ █ ███████ ███ █████████ ███ █ █████████
█         █ █ █     █   █ █ █   █  *█ █   █ █ █ █           █     █ █   █ █ █ █ █
███████ █ █ █ ███ ███ ███ █ ███ ███ █ █ █████ █ █████ █ █████ █████ █ ███ █ █ █ █
█ █     █ █ █ █ █ █ █ █ █ █     █ █*█* *    █   █     █ █ █       █ █ █   █   █ █
█ █ █████ █ █ █ █ █ █ █ █ ███ ███ █ █ █ █████ ███ █████ █ █ ███████ █ ███ █ ███ █
█ █ █   █   █         █   █   █   █* *█*█   █ █     █ █ █ █     █               █
█ █████ ███ ███ ███ ███ █████ █ ███ ███ █ ███ █ █████ ███ █ ███████ █████████████
█ █         █ █ █ █             █   █ █*  █ █ █ █       █     █ █ █       █     █
█ █████ ███ █ █ █ █ ███ ███ █ █ █████ █ ███ █ █ █ ███████ █████ █ █ ███████ █████
█     █ █ █     █ █ █     █ █ █   █ █ █*              █ █ █ █ █ █     █ █       █
█████ ███ ███████ ███████████ █████ █ █ █████ █████████ █ █ █ █ ███ ███ █ ███ █ █
█     █ █   █     █             █   █  *  █ █ █ █     █       █ █         █ █ █ █
█████ █ ███ █████ ███ ███ █████ ███ ███ ███ ███ █ █████ ███████ █ █████████ █ ███
█           █ █       █ █ █ █ █   █    *      █ █   █     █ █ █ █       █ █ █   █
███ █ ███ █ █ ███ █ █ █ █ █ █ ███ █████ █ █████ █ ███ █████ █ █ █ █████ █ █ █ ███
█   █ █   █       █ █ █   █       █   █*█   █       █   █ █       █ █ █     █ █ █
█████ █████ ███ █████ █ ███ ███ █ ███ █ ███████ ███████ █ █ ███ ███ █ █ ███ ███ █
█     █ █   █     █   █   █ █   █      *  █   █ █ █ █         █       █   █     █
█ █████ █ █ █ █████████ █ ███ █████████ ███ ███ █ █ █ █ █ ███ █ ███ ███ ███████ █
█ █   █   █ █   █       █   █ █ █   █ █* *█ █   █     █ █   █ █ █     █   █     █
█████ █ █████ █████████ ███████ █ █ █ ███ █ ███ █ ███ █ █████ █ █ █ ███ █████ █ █
█       █       █ █     █         █      *  █       █ █   █ █ █ █ █   █   █   █ █
█████ █████ ███ █ ███ █ █ █████ █████ ███ ███ ███████ █ ███ █████ ███ █ █ █ █ █ █
█     █       █ █     █ █ █     █       █*          █ █         █   █ █ █ █ █ █ █
█████████████████████████████████████████ ███████████████████████████████████████
d_path [(20, 0), (20, 1), (20, 2), (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (18, 10), (18, 9), (17, 9), (17, 10), (17, 11), (17, 12), (16, 12), (15, 12), (14, 12), (14, 13), (14, 14), (14, 15), (15, 15), (15, 16), (14, 16), (14, 17), (13, 17), (12, 17), (12, 18), (12, 19), (12, 20), (13, 20), (13, 21), (14, 21), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25), (15, 26), (15, 27), (14, 27), (14, 28), (13, 28), (12, 28), (12, 29), (11, 29), (10, 29), (9, 29), (9, 30), (10, 30), (10, 31), (10, 32), (10, 33), (9, 33), (9, 34), (9, 35), (9, 36), (9, 37), (8, 37), (8, 38), (7, 38), (6, 38), (5, 38), (4, 38), (4, 39), (4, 40), (4, 41), (4, 42), (4, 43), (5, 43), (5, 44)]
my_maze.solution_path [(20, 0), (20, 1), (20, 2), (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (18, 10), (18, 9), (17, 9), (17, 10), (17, 11), (17, 12), (16, 12), (15, 12), (14, 12), (14, 13), (14, 14), (14, 15), (15, 15), (15, 16), (14, 16), (14, 17), (13, 17), (12, 17), (12, 18), (12, 19), (12, 20), (13, 20), (13, 21), (14, 21), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25), (15, 26), (15, 27), (14, 27), (14, 28), (13, 28), (12, 28), (12, 29), (11, 29), (10, 29), (9, 29), (9, 30), (10, 30), (10, 31), (10, 32), (10, 33), (9, 33), (9, 34), (9, 35), (9, 36), (9, 37), (8, 37), (8, 38), (7, 38), (6, 38), (5, 38), (4, 38), (4, 39), (4, 40), (4, 41), (4, 42), (4, 43), (5, 43), (5, 44)]
a_path [(20, 0), (20, 1), (20, 2), (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (18, 10), (18, 9), (17, 9), (17, 10), (17, 11), (17, 12), (16, 12), (15, 12), (14, 12), (14, 13), (14, 14), (14, 15), (15, 15), (15, 16), (14, 16), (14, 17), (13, 17), (12, 17), (12, 18), (12, 19), (12, 20), (13, 20), (13, 21), (14, 21), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25), (15, 26), (15, 27), (14, 27), (14, 28), (13, 28), (12, 28), (12, 29), (11, 29), (10, 29), (9, 29), (9, 30), (10, 30), (10, 31), (10, 32), (10, 33), (9, 33), (9, 34), (9, 35), (9, 36), (9, 37), (8, 37), (8, 38), (7, 38), (6, 38), (5, 38), (4, 38), (4, 39), (4, 40), (4, 41), (4, 42), (4, 43), (5, 43), (5, 44)]
Dijkstra took 16.387 ms with Method.RANDOM
A* took 14.540 ms with Method.RANDOM
Maze Size 40
███████████████████████████████████████████████████████ █████████████████████████
█           █       █   █           █   █ █ █     █   █*  █   █   █ █     █ █   █
█████ ███ █ █████ █████ █████ █ █ ███ ███ █ █████ █ ███ █ █ ███ ███ █ █████ █ ███
█ █ █ █   █       █ █   █ █   █ █     █     █     █    *█     █   █   █   █ █   █
█ █ █████████████ █ █ ███ ███████ █████ █████████ █████ ███████ ███ ███ ███ █ ███
█       █   █ █   █ █   █ █       █   █   █   █   █ █  *    █   █       █ █     █
███ ███████ █ ███ █ █ ███ ███████ ███ ███ █ █████ █ ███ ███████ █ ███ ███ █ █████
█ █   █ █       █   █   █ █ █ █   █ █ █       █   █ █  *    █   █   █ █     █   █
█ █ ███ ███████ ███ █ ███ █ █ ███ █ █ ███ ███████ █ ███ ███ ███ █ █████ █████ ███
█         █ █ █         █   █ █     █ █ █   █ █     █  *█ █   █         █   █ █ █
███████ ███ █ █ █████ █████ █ █████ █ █ █ ███ ███ █████ █ ███████ ███████ ███ █ █
█       █     █ █ █ █ █ █     █   █   █   █ █ █    * * *    █ █ █   █ █   █     █
███████ █ ███████ █ █ █ █ ███████ ███ █ ███ █ ███ █ █████ ███ █ ███ █ ███ ███ █ █
█ █ █ █     █   █         █   █     █ █ █ █ █ █   █*█ █ █ █       █ █ █       █ █
█ █ █ █ █████ █████████ ███ ███ ███ █ █ █ █ █ █████ █ █ █████ █████ █ █ █████ █ █
█           █   █ █     █   █   █     █ █     █  * *  █     █           █ █ █ █ █
███████ ███████ █ █ ███ █ ███████████ █ ███ █████ ███████ █ █ ███████████ █ █████
█   █   █ █ █ █   █ █ █   █ █   █   █ █ █   █    *█ █ █   █ █   █     █         █
███ ███ █ █ █ █ █████ █ █ █ ███ ███ █ █ ███ █████ █ █ █ ███ █ ███ █████ █████████
█       █   █       █ █ █ █   █   █ █ █ █   █   █*█ █ █ █ █ █ █   █       █     █
███████ █ ███████ ███ ███ █ ███ ███ █ █ █ ███ ███ █ █ █ █ ███ █ ███ ███████ █████
█ █ █     █   █   █   █ █ █   █ █     █ █ █   █ █* *█       █   █ █   █   █   █ █
█ █ ███ █ ███ █ █████ █ █ ███ █ ███ ███ █ █ ███ ███ █ █████ ███ █ ███ █ ███ ███ █
█   █ █ █ █ █ █ █     █           █       █     █ █* *    █ █ █           █   █ █
███ █ ███ █ █ █ █████ ███████ █████████ ███████ █ ███ ███████ █ ███████████ ███ █
█ █   █ █ █     █     █   █ █ █ █   █ █     █   █   █*  █ █ █   █ █   █   █ █ █ █
█ ███ █ █ █ █████████ ███ █ █ █ █ ███ █ █████ ███ ███ ███ █ █ ███ ███ ███ █ █ █ █
█   █ █   █     █         █ █   █ █ █ █ █ █   █ █   █*█         █       █ █     █
███ █ ███ ███ █████ █ ███ █ ███ █ █ █ █ █ ███ █ █ ███ ███ █ █ █████ █████ █████ █
█           █ █ █ █ █ █ █   █ █   █       █   █ █   █*█   █ █   █     █ █   █ █ █
███████████ █ █ █ █████ ███ █ █ ███ █████████ █ █ ███ ███ █████████ ███ █ ███ █ █
█   █ █ █ █       █ █     █ █ █   █ █   █           █*█ █ █   █ █ █     █     █ █
███ █ █ █ █████ ███ █████ █ █ █ ███ ███ ███ █████████ █ █ ███ █ █ █ █████████ █ █
█                 █ █       █ █ █ █ █         █ █   █*█ █ █   █     █     █   █ █
███████ ███ █████ █ ███████ █ █ █ █ ███████ ███ █ ███ █ █ █ █████ █ █ █████ ███ █
█ █     █   █     █   █     █     █ █   █     █   █* *█           █ █   █ █   █ █
█ █████████ █████ █ ███████ █ █ ███ ███ ███ ███ ███ █████ █ █████ █████ █ █ ███ █
█ █ █     █ █         █   █ █ █   █     █       █  *  █ █ █ █ █     █ █   █ █   █
█ █ ███ █████████ ███████ █ █ █ ███ ███ ███ ███████ ███ █ ███ ███████ █ ███ █ ███
█   █     █   █   █ █       █ █   █ █ █ █   █ █ █* *      █ █ █   █       █   █ █
█ ███████ ███ ███ █ ███████ ███ ███ █ ███ ███ █ █ █████████ █ █ █ ███████ █ ███ █
█ █ █ █ █ █ █     █     █         █   █ █     █ █*█       █     █ █ █     █     █
█ █ █ █ █ █ █████ ███ ███ ███ █ █████ █ █ █████ █ █ █████████ █████ █████ █ █ █ █
█ █               █     █ █   █   █   █     █ █ █*  █           █     █     █ █ █
█ ███████████████ ███ █████ ███ █████ █ █ ███ █ █ █████ ███████ █████ ███ ███████
█ █   █     █ █   █ █     █ █ █ █       █ █ █ █* *  █   █ █   █ █ █     █ █ █ █ █
█ ███ █████ █ ███ █ █ ███████ █ ███████ ███ █ █ █ █████ █ █ █████ █████ █ █ █ █ █
█ █       █ █ █     █   █ █ █ █     █     █    *█ █     █ █ █       █ █ █     █ █
█ ███ █████ █ █████ ███ █ █ █ █████ ███ ███████ ███████ █ █ █ ███████ █ ███ ███ █
█   █   █ █   █ █ █     █   █   █       █ █    *█       █ █ █ █ █ █ █ █ █ █ █   █
███ █ ███ ███ █ █ █ ███ ███ ███ █████ ███ █████ █ █ █████ █ █ █ █ █ █ █ █ █ ███ █
█ █ █   █ █   █ █ █ █           █   █   █* * * *█ █ █         █ █   █ █   █     █
█ █ █ ███ ███ █ █ ███████████ █████ █ ███ ███████ ███████ █████ ███ █ █ █████ ███
█ █   █     █ █   █ █       █     █    * *  █       █ █   █   █       █ █ █ █   █
█ █ ███████ █ ███ █ ███████ █ █████████ ███ ███ █████ █ ███ ███ ███████ █ █ █ ███
█ █   █   █   █ █ █ █ █     █   █ █ █ █*  █ █   █   █ █ █   █ █ █     █   █ █   █
█ ███ █ █████ █ █ █ █ █████ █ ███ █ █ █ █ █████ █ ███ █ █ ███ █ ███ ███ ███ █ ███
█       █   █     █ █ █       █   █   █*█ █     █   █   █ █   █     █ █   █ █ █ █
█ ███ █ ███ █████ █ █ ███ ███ █ ███ ███ ███████ █ ███ ███ █ ███ █████ █ ███ █ █ █
█ █ █ █       █   █   █ █ █     █   █* *    █   █ █ █ █   █   █ █   █ █ █     █ █
███ █ ███████ ███ █ ███ █████ ███ ███ █ ███████ █ █ █ ███ █ █ █ █ ███ █ █ █████ █
█     █             █ █   █   █ █ █* *█   █ █ █     █ █     █ █ █ █ █   █ █     █
█████████████████ █ █ █ █████ █ █ █ ███████ █ █ █████ ███ █████ █ █ █ ███ █ ███ █
█       █ █       █ █   █ █       █*        █   █   █ █ █   █ █               █ █
█████ ███ ███████ █████ █ █ █████ █ ███████████ ███ █ █ █ ███ ███ █████ █ █████ █
█         █   █   █   █ █ █ █   █ █*  █ █   █     █   █   █ █       █ █ █   █ █ █
███ █████ ███ ███ █ ███ █ █████ █ █ ███ ███ █████ ███ ███ █ █ ███████ ███████ ███
█ █ █         █ █     █ █   █   █  *      █     █       █   █       █ █ █ █ █   █
█ █ ███ █████ █ █████ █ ███ ███ ███ ███████ ███████ ███████ █ █ █████ █ █ █ █ ███
█ █ █   █               █ █ █ █   █*█   █   █ █   █ █ █       █ █     █       █ █
█ █████████████ ███ ███ █ █ █ ███ █ ███ ███ █ █ ███ █ ███ ███████████ █ ███████ █
█   █   █   █ █ █     █   █     █ █*█     █       █ █       █   █       █   █ █ █
█ █████ ███ █ ███ █████ █████ ███ █ ███ █████ █████ █ █ █████ █████ █ ███ ███ █ █
█ █ █         █   █       █   █ █ █*█     █ █ █ █     █     █     █ █ █ █ █   █ █
█ █ █████████ █████ ███ █ ███ █ █ █ █████ █ █ █ ███ █████████ ███████ █ █ █ ███ █
█ █ █   █ █ █ █ █   █   █     █ █  * *  █ █     █   █   █ █ █     █ █ █ █ █   █ █
█ █ ███ █ █ █ █ █████████████ █ █████ ███ █ ███████ █ ███ █ █ █████ █ █ █ █ ███ █
█ █ █     █ █       █             █* *█ █     █       █ █   █   █       █   █   █
█ █ █████ █ ███████ █ █ █████ █ █ █ ███ ███ █████ █████ ███ █ ███ ███ █ █ ███ ███
█       █ █ █       █ █ █   █ █ █  * * * *█ █           █   █ █     █ █ █   █ █ █
███ ███ █ █ ███████ ███████ █████████████ █ █ █████ █████ ███ █ ███ █████ ███ █ █
█ █ █                 █ █   █   █ █   █  * *█ █ █ █ █     █ █   █ █           █ █
█ ███ █ ███████ █ ███ █ ███ ███ █ ███ █████ █ █ █ ███ █████ █ ███ ███████ █████ █
█     █ █       █ █ █       █ █ █ █ █   █  *  █       █ █ █         █ █ █       █
███████ █ █████████ ███████ █ █ █ █ ███ ███ ███ █████ █ █ █ █████████ █ █████████
█       █ █ █           █ █         █ █ █* *█ █   █ █ █ █ █   █ █           █   █
█████ █████ ███████████ █ ███ ███ ███ █ █ ███ █ ███ ███ █ █ ███ ███ █████████ ███
█   █ █ █ █   █ █   █ █     █ █     █    * *█     █   █ █   █   █ █ █   █   █ █ █
███ ███ █ ███ █ ███ █ ███ █████████ ███████ █ █████ ███ █ ███ ███ █ █ █████ █ █ █
█                                        * *                                    █
█████████████████████████████████████████ ███████████████████████████████████████
d_path [(20, 0), (21, 0), (21, 1), (20, 1), (20, 2), (21, 2), (21, 3), (21, 4), (20, 4), (20, 5), (19, 5), (18, 5), (17, 5), (17, 6), (18, 6), (18, 7), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (18, 14), (18, 15), (19, 15), (19, 16), (19, 17), (19, 18), (20, 18), (20, 19), (21, 19), (22, 19), (23, 19), (23, 20), (23, 21), (23, 22), (24, 22), (24, 23), (24, 24), (24, 25), (25, 25), (25, 26), (25, 27), (26, 27), (26, 28), (26, 29), (26, 30), (26, 31), (26, 32), (26, 33), (25, 33), (25, 34), (24, 34), (24, 35), (24, 36), (24, 37), (25, 37), (25, 38), (25, 39), (26, 39), (27, 39), (27, 40), (27, 41), (27, 42), (27, 43), (27, 44)]
my_maze.solution_path [(20, 0), (21, 0), (21, 1), (20, 1), (20, 2), (21, 2), (21, 3), (21, 4), (20, 4), (20, 5), (19, 5), (18, 5), (17, 5), (17, 6), (18, 6), (18, 7), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (18, 14), (18, 15), (19, 15), (19, 16), (19, 17), (19, 18), (20, 18), (20, 19), (21, 19), (22, 19), (23, 19), (23, 20), (23, 21), (23, 22), (24, 22), (24, 23), (24, 24), (24, 25), (25, 25), (25, 26), (25, 27), (26, 27), (26, 28), (26, 29), (26, 30), (26, 31), (26, 32), (26, 33), (25, 33), (25, 34), (24, 34), (24, 35), (24, 36), (24, 37), (25, 37), (25, 38), (25, 39), (26, 39), (27, 39), (27, 40), (27, 41), (27, 42), (27, 43), (27, 44)]
a_path [(20, 0), (21, 0), (21, 1), (20, 1), (20, 2), (21, 2), (21, 3), (21, 4), (20, 4), (20, 5), (19, 5), (18, 5), (17, 5), (17, 6), (18, 6), (18, 7), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (18, 14), (18, 15), (19, 15), (19, 16), (19, 17), (19, 18), (20, 18), (20, 19), (21, 19), (22, 19), (23, 19), (23, 20), (23, 21), (23, 22), (24, 22), (24, 23), (24, 24), (24, 25), (25, 25), (25, 26), (25, 27), (26, 27), (26, 28), (26, 29), (26, 30), (26, 31), (26, 32), (26, 33), (25, 33), (25, 34), (24, 34), (24, 35), (24, 36), (24, 37), (25, 37), (25, 38), (25, 39), (26, 39), (27, 39), (27, 40), (27, 41), (27, 42), (27, 43), (27, 44)]
█████████████████████████████████████████████████████████████████████ ███████████
█ █   █ █ █ █ █   █ █   █       █ █ █ █   █ █   █   █ █ █ █ █     █  *      █ █ █
█ █ █ █ █ █ █ █ ███ █ ███ ███████ █ █ ███ █ █ ███ ███ █ █ █ █████ ███ ███ ███ █ █
█ █ █   █       █   █ █       █ █             █   █   █ █   █ █   █  *  █     █ █
█ █████ █ ███████ ███ ███ █████ █████████ █ █ █ █ █ ███ █ ███ █ █ ███ █████████ █
█   █ █ █     █   █ █ █ █ █   █ █   █   █ █ █ █ █   █ █ █   █ █ █ █  *█   █   █ █
███ █ █ ███ █████ █ █ █ █ █ ███ ███ ███ █ █████████ █ █ █ ███ ███ ███ █ ███ ███ █
█   █   █ █ █ █ █ █         █ █ █   █   █   █     █         █ █   █ █*█ █ █ █ █ █
███ ███ █ █ █ █ █ ███ ███████ █ ███ █ ███ █ █████ █ █ █████ █ █ ███ █ █ █ █ █ █ █
█   █ █   █ █ █ █ █   █     █     █ █     █ █     █ █ █ █           █*█   █     █
███ █ ███ █ █ █ █ ███ █ ███ █████ █ █ ███ ███████ █████ █████ ███████ █ ███ █████
█ █ █   █         █     █ █ █       █   █ █ █       █ █ █         █* *█     █   █
█ █ ███ ███ ███████ ███ █ █████████ █ █████ █████ ███ █ █████ █████ █████ ███ ███
█     █   █ █   █   █     █   █           █   █   █ █       █   █ █* *█         █
███ █████ █ █ █████████ ███ █ ███ ███ █████ █████ █ ███████ █ ███ ███ █ █ █ █████
█ █ █   █ █     █   █     █ █ █ █ █   █ █ █     █ █ █   █   █   █ █ █*█ █ █     █
█ █ █ ███ █ ███ ███ ███ ███ ███ █████ █ █ █ █████ █ █ █████ █ █ █ █ █ ███ ███████
█ █ █   █   █ █ █ █     █   █   █ █       █   █   █   █     █ █ █ █ █*    █ █ █ █
█ █ ███ █ ███ ███ █████ █ █████ █ ███ █ █ █ ███ ███ █ ███ ███ ███ █ █ █████ █ █ █
█ █   █     █ █     █ █ █   █   █   █ █ █   █     █ █ █ █ █   █ █ █  * *        █
█ █ █████ █ █ █ █████ █ █ █████ █ █████████ █████ ███ █ █ █ ███ █ █████ ███ ███ █
█         █ █   █ █   █     █   █     █   █ █ █ █ █ █ █   █ █ █* * * * *  █   █ █
█████ █ █ █████ █ ███ █ ███████ ███ █████ █ █ █ █ █ █ ███ █ █ █ █████ █ █ █ ███ █
█     █ █     █   █ █     █ █ █ █   █   █   █ █     █         █*  █ █ █ █ █ █   █
█████ █ █ ███████ █ █ █████ █ █ ███ █ █████ █ █████ ███████ ███ ███ █████████████
█   █ █ █     █ █   █ █   █ █ █ █ █ █   █ █ █   █ █     █    * *  █     █       █
███ ███████ ███ █ ███ ███ █ █ █ █ █ ███ █ █ ███ █ █████ █████ ███ █ █████ ███████
█     █         █ █   █   █       █ █   █     █ █ █   █    * *█ █   █ █ █ █   █ █
█████ █████████ █ █ █████ █████ ███ ███ █████ █ █ ███ █████ ███ █████ █ █ █ ███ █
█   █ █ █ █ █         █ █     █   █   █     █ █   █ █ █   █*    █ █       █     █
███ █ █ █ █ █████ █████ ███ ███ █████ █ █████ █ ███ █ ███ █ █ ███ █ ███████ █ █ █
█     █       █   █   █     █ █   █     █ █ █ █ █ █ █ █* * *█             █ █ █ █
█████ █████ █████ █ ███████ █ ███ ███ ███ █ █ █ █ █ █ █ ███ █ ███ ███ █████ █████
█   █   █ █ █   █   █ █   █   █       █   █ █   █   █ █*  █ █ █ █ █       █   █ █
███ █ ███ █ ███ █ ███ ███ ███ █████ ███ ███ █ █████ █ █ ███████ ███████████ ███ █
█     █ █   █     █   █ █   █     █ █   █   █ █   █  * *      █ █   █   █ █   █ █
█ █ ███ ███ █████ █ ███ █ ███████ █ ███ █ ███ █ █████ █ ███ █ █ █ ███ ███ █ ███ █
█ █     █   █ █     █ █   █         █       █ █   █ █*█ █ █ █ █     █   █ █ █   █
█ █████ ███ █ ███ ███ ███ █████████ █ ███████ █ ███ █ ███ █████ █████ ███ █ ███ █
█ █ █         █       █ █   █ █ █       █ █ █ █ █ █ █*█ █ █ █     █   █ █   █   █
███ █████████ ███ █████ █ ███ █ █████ ███ █ █ █ █ █ █ █ █ █ █ █████ ███ ███ █ ███
█ █     █ █ █ █ █     █   █ █ █ █     █     █ █     █*█ █ █ █     █ █ █   █   █ █
█ ███ ███ █ █ █ █ █████ ███ █ █ ███ ███ █ █ █ █ ███ █ █ █ █ █ █████ █ █ █████ █ █
█   █   █     █   █ █ █ █   █       █ █ █ █ █   █ █ █*█ █   █ █   █ █ █ █ █     █
███ ███ █████ ███ █ █ █ █ █████████ █ █████ ███ █ ███ █ █ ███ █ ███ █ █ █ █ █████
█ █       █ █   █   █     █   █   █   █ █ █ █ █ █   █*█ █ █ █ █ █       █   █ █ █
█ ███ ███ █ █ ███ █ █████ █ █████ █ ███ █ █ █ █ █ ███ █ █ █ █ █ █ █████████ █ █ █
█ █   █   █ █   █ █ █ █ █ █   █     █   █ █   █ █* * *  █   █   █     █     █   █
█ ███████ █ █ █████ █ █ █ █ ███████ █ ███ █ ███ █ ███████ ███ ███ █████ ███████ █
█ █       █   █   █ █   █   █ █     █ █   █   █ █*█ █ █ █ █   █ █   █ █ █ █   █ █
█ ███ █ █████ ███ █ ███ █ ███ █████ █ ███ ███ █ █ █ █ █ █ █ ███ █ ███ █ █ ███ █ █
█ █ █ █   █ █   █ █     █ █     █ █   █ █   █   █*█   █ █   █ █       █     █   █
█ █ ███ ███ ███ █ ███ █ █ ███ ███ █ ███ █ █████ █ █ ███ █ ███ █ ███████ █████ ███
█           █ █   █   █ █   █   █ █ █   █   █ █  *          █   █ █ █ █     █   █
█████ ███ █ █ ███ ███ █ █ ███ ███ █ ███ ███ █ ███ ███████████ ███ █ █ █ █████ ███
█     █ █ █       █   █   █ █ █ █ █ █     █     █* *█   █     █ █ █ █ █ █       █
█ █████ █ ███████ █████ █ █ █ █ █ █ █████ ███ █████ ███ ███ ███ █ █ █ █ █ █████ █
█ █   █ █ █   █ █ █ █ █ █ █ █ █       █ █       █  *    █ █   █   █ █   █     █ █
█████ █ █████ █ █ █ █ ███ █ █ █████ ███ █████ █████ █████ █ █ █ █ █ █ ███ ███████
█                 █ █     █ █   █ █   █      * * * *    █ █ █ █ █ █ █ █ █       █
█ █████████████ █ █ ███ █ █ ███ █ █ ███████ █ ███ ███████ ███ █ ███ █ █ █ █ █████
█ █ █ █         █ █   █ █   █   █ █     █ █ █*█ █   █ █ █ █ █   █ █   █   █ █ █ █
███ █ ███ █████ █ ███ █████ ███ █ █ █████ ███ █ █████ █ █ █ ███ █ █ ███ █████ █ █
█         █   █ █   █   █   █ █ █ █ █   █ █ █*      █   █ █     █     █ █   █   █
█████████████ █████ █ █████ █ █ █ █ █ ███ █ █ █ █████ ███ ███ █ █ █████ █ █████ █
█     █               █       █ █ █       █* *█   █     █   █ █   █     █   █   █
█████ ███████████████ █████ ███ █ ███ █████ █████████ ███ █████ █ ███ ███ ███ ███
█ █   █ █   █ █ █ █       █ █ █   █   █ █* *█ █   █     █   █   █ █           █ █
█ ███ █ ███ █ █ █ ███████ █ █ ███ ███ █ █ ███ █ ███ ███████ █ ███████ █████ █ █ █
█     █         █ █     █ █ █     █ █   █* *      █ █   █ █ █             █ █   █
█████ █████████ █ █████ █ █ █████ █ ███ ███ ███████ █ ███ █ █ █ ███████ █ █ █████
█ █     █ █   █ █     █ █         █     █* *█     █ █     █   █ █ █   █ █ █     █
█ █████ █ ███ █ █ ███ █ █████████ █ █████ ███ █████ █ █████ █████ █ █████████████
█     █ █     █ █ █       █ █ █       █  *  █ █       █   █ █ █ █ █           █ █
█████ █ █████ █ █████████ █ █ █████ █ ███ ███ ███ ███████ █ █ █ █ █ █████ █████ █
█             █         █       █ █ █   █*█     █ █     █ █   █         █       █
█ █████ ███ ███████████ ███████ █ ███ ███ █ █████ ███ ███ █ ███ █████ █████ █████
█ █     █         █               █ █ █  *█     █ █   █ █           █ █   █     █
███ █ █ █ █ █████ █ █ █ █ █████ ███ █ ███ █ █████ ███ █ █ ███ █████████ █████████
█   █ █ █ █ █     █ █ █ █ █ █       █ █* *█ █ █   █   █     █ █ █ █   █       █ █
███ ███████ █ █ █ ███████ █ ███ ███ █ █ ███ █ █ ███ █████ █████ █ █ ███ ███ ███ █
█     █     █ █ █       █ █   █ █ █  * *█ █   █ █ █ █ █ █ █   █ █         █   █ █
█████ █ █ ███ █████████ █████ ███ ███ ███ █ ███ █ █ █ █ █ █ ███ █ █ ███████ ███ █
█ █ █ █ █ █ █ █         █ █         █*█ █   █ █ █   █       █     █   █ █ █ █   █
█ █ ███ █ █ ███████ ███ █ █ ███ ███ █ █ ███ █ █ ███ █ █████████ ███████ █ ███ ███
█       █ █         █       █   █    *█   █ █ █ █ █         █                   █
███████████ ███ █ ███ █ ███████ █████ ███ █ █ █ █ █ ███ █████ █ █████████████ █ █
█         █ █   █ █   █ █       █    * * *█   █     █ █ █     █             █ █ █
█ ███ ███ ███████████████████ █ █████ ███ █ █████ ███ ███ █ █ █████████ █████████
█   █ █                       █ █     █  *                █ █         █         █
█████████████████████████████████████████ ███████████████████████████████████████
d_path [(20, 0), (20, 1), (19, 1), (18, 1), (18, 2), (18, 3), (18, 4), (19, 4), (19, 5), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (21, 9), (21, 10), (20, 10), (20, 11), (21, 11), (21, 12), (22, 12), (22, 13), (22, 14), (22, 15), (23, 15), (24, 15), (25, 15), (25, 16), (25, 17), (24, 17), (24, 18), (24, 19), (24, 20), (24, 21), (25, 21), (26, 21), (26, 22), (26, 23), (26, 24), (26, 25), (26, 26), (26, 27), (27, 27), (27, 28), (27, 29), (28, 29), (29, 29), (29, 30), (29, 31), (30, 31), (30, 32), (31, 32), (31, 33), (31, 34), (32, 34), (33, 34), (34, 34), (35, 34), (35, 35), (34, 35), (34, 36), (34, 37), (34, 38), (33, 38), (33, 39), (34, 39), (34, 40), (34, 41), (34, 42), (34, 43), (34, 44)]
my_maze.solution_path [(20, 0), (20, 1), (19, 1), (18, 1), (18, 2), (18, 3), (18, 4), (19, 4), (19, 5), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (21, 9), (21, 10), (20, 10), (20, 11), (21, 11), (21, 12), (22, 12), (22, 13), (22, 14), (22, 15), (23, 15), (24, 15), (25, 15), (25, 16), (25, 17), (24, 17), (24, 18), (24, 19), (24, 20), (24, 21), (25, 21), (26, 21), (26, 22), (26, 23), (26, 24), (26, 25), (26, 26), (26, 27), (27, 27), (27, 28), (27, 29), (28, 29), (29, 29), (29, 30), (29, 31), (30, 31), (30, 32), (31, 32), (31, 33), (31, 34), (32, 34), (33, 34), (34, 34), (35, 34), (35, 35), (34, 35), (34, 36), (34, 37), (34, 38), (33, 38), (33, 39), (34, 39), (34, 40), (34, 41), (34, 42), (34, 43), (34, 44)]
a_path [(20, 0), (20, 1), (19, 1), (18, 1), (18, 2), (18, 3), (18, 4), (19, 4), (19, 5), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (21, 9), (21, 10), (20, 10), (20, 11), (21, 11), (21, 12), (22, 12), (22, 13), (22, 14), (22, 15), (23, 15), (24, 15), (25, 15), (25, 16), (25, 17), (24, 17), (24, 18), (24, 19), (24, 20), (24, 21), (25, 21), (26, 21), (26, 22), (26, 23), (26, 24), (26, 25), (26, 26), (26, 27), (27, 27), (27, 28), (27, 29), (28, 29), (29, 29), (29, 30), (29, 31), (30, 31), (30, 32), (31, 32), (31, 33), (31, 34), (32, 34), (33, 34), (34, 34), (35, 34), (35, 35), (34, 35), (34, 36), (34, 37), (34, 38), (33, 38), (33, 39), (34, 39), (34, 40), (34, 41), (34, 42), (34, 43), (34, 44)]
Dijkstra took 14.927 ms with Method.BIAS
A* took 14.892 ms with Method.BIAS

Process finished with exit code 0


"/Users/jiafei/Desktop/courses/SP 23/Sp23 CS F003C 02W Adv Data Struct Algorithm Python/bin/python" /Users/jiafei/Desktop/courses/SP 23/Sp23 CS F003C 02W Adv Data Struct Algorithm Python/maze_A_star.py 
Maze Size 5
Dijkstra took 0.407 ms with Method.STACK
A* took 0.424 ms with Method.STACK
Maze Size 5
Dijkstra took 0.339 ms with Method.RANDOM
A* took 0.340 ms with Method.RANDOM
Maze Size 5
Dijkstra took 0.328 ms with Method.BIAS
A* took 0.331 ms with Method.BIAS
Maze Size 10
Dijkstra took 0.894 ms with Method.STACK
A* took 1.069 ms with Method.STACK
Maze Size 10
Dijkstra took 0.931 ms with Method.RANDOM
A* took 1.019 ms with Method.RANDOM
Maze Size 10
Dijkstra took 1.114 ms with Method.BIAS
A* took 1.001 ms with Method.BIAS
Maze Size 20
Dijkstra took 3.290 ms with Method.STACK
A* took 4.669 ms with Method.STACK
Maze Size 20
Dijkstra took 3.426 ms with Method.RANDOM
A* took 4.475 ms with Method.RANDOM
Maze Size 20
Dijkstra took 3.125 ms with Method.BIAS
A* took 4.572 ms with Method.BIAS
Maze Size 40
Dijkstra took 16.986 ms with Method.STACK
A* took 16.357 ms with Method.STACK
Maze Size 40
Dijkstra took 16.327 ms with Method.RANDOM
A* took 18.465 ms with Method.RANDOM
Maze Size 40
Dijkstra took 15.889 ms with Method.BIAS
A* took 15.813 ms with Method.BIAS
Maze Size 80
Dijkstra took 60.824 ms with Method.STACK
A* took 66.973 ms with Method.STACK
Maze Size 80
Dijkstra took 56.517 ms with Method.RANDOM
A* took 57.312 ms with Method.RANDOM
Maze Size 80
Dijkstra took 55.837 ms with Method.BIAS
A* took 55.629 ms with Method.BIAS
Maze Size 160
Dijkstra took 238.663 ms with Method.STACK
A* took 272.674 ms with Method.STACK
Maze Size 160
Dijkstra took 225.493 ms with Method.RANDOM
A* took 262.013 ms with Method.RANDOM
Maze Size 160
Dijkstra took 229.762 ms with Method.BIAS
A* took 255.777 ms with Method.BIAS
   Maze Size     d_STACK     a_STACK  ...    a_RANDOM      d_BIAS      a_BIAS
0          5    0.407 ms    0.424 ms  ...     0.34 ms    0.328 ms    0.331 ms
1         10    0.894 ms    1.069 ms  ...    1.019 ms    1.114 ms    1.001 ms
2         20     3.29 ms    4.669 ms  ...    4.475 ms    3.125 ms    4.572 ms
3         40   16.986 ms   16.357 ms  ...   18.465 ms   15.889 ms   15.813 ms
4         80   60.824 ms   66.973 ms  ...   57.312 ms   55.837 ms   55.629 ms
5        160  238.663 ms  272.674 ms  ...  262.013 ms  229.762 ms  255.777 ms

[6 rows x 7 columns]

Process finished with exit code 0



"""
