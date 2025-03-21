import copy
import random


class MinHeap:

    class EmptyHeapError(Exception):
        pass

    def __init__(self, list_in=None):
        if list_in is None:
            self._heap_list = [None]
            self._size = 0
        else:
            self._heap_list = copy.deepcopy(list_in)
            self._size = len(list_in)
            self._heap_list.insert(0, 0)
            self._order_heap()

    @property
    def size(self):
        return self._size

    def insert(self, data):
        self._heap_list.append(None)
        self._size += 1
        child_index = self._size
        while child_index > 1 and data < self._heap_list[child_index >> 1]:
            self._heap_list[child_index] = self._heap_list[child_index >> 1]
            child_index = child_index >> 1
        self._heap_list[child_index] = data

    def _percolate_down(self, hole):
        val = self._heap_list[hole]
        while hole <= self._size // 2:
            left_idx = hole * 2
            right_idx = hole * 2 + 1
            small_idx = hole
            if left_idx <= self._size and val > self._heap_list[left_idx]:
                small_idx = left_idx
            if right_idx <= self._size and self._heap_list[small_idx] > self._heap_list[right_idx]:
                small_idx = right_idx
            self._heap_list[hole] = self._heap_list[small_idx]
            if hole == small_idx:
                break
            hole = small_idx
        self._heap_list[hole] = val

    def remove(self):
        if self._size == 0:
            raise MinHeap.EmptyHeapError
        return_value = self._heap_list[1]
        self._heap_list[1] = self._heap_list[self._size]
        self._size -= 1
        self._heap_list.pop()
        if self._size > 0:
            self._percolate_down(1)
        return return_value

    def _order_heap(self):
        for i in range(self._size // 2, 0, -1):
            self._percolate_down(i)


def main():
    # test of insert and remove
    print("Test of insert and remove.")
    sort_heap = MinHeap()
    sort_heap.insert(35)
    sort_heap.insert(29)
    sort_heap.insert(59)
    sort_heap.insert(11)
    sort_heap.insert(3)
    sort_heap.insert(46)
    sort_heap.insert(25)
    print(f"The size of heap_list is {sort_heap.size}.")
    print("Remove", sort_heap.remove())
    print(f"The size of heap_list is {sort_heap.size}.")
    print("Remove", sort_heap.remove())
    print(f"The size of heap_list is {sort_heap.size}.")

    # test of the list parameter of __init__() and remove
    print("\nTest of the list parameter of __init__() and remove.")
    sort_heap = MinHeap([13, 14, 16, 19, 21, 19, 68, 65, 26, 32, 31])
    print(f"The size of heap_list is {sort_heap.size}.")
    print("Remove", sort_heap.remove())
    print(f"The size of heap_list is {sort_heap.size}.")

    # test of the size of sort_heap is odd then remove all of them
    print("\nTest of the size of sort_heap is odd then remove all of them.")
    sort_heap = MinHeap()
    sort_heap.insert(10)
    sort_heap.insert(9)
    sort_heap.insert(8)
    sort_heap.insert(7)
    sort_heap.insert(6)
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)

    # test of the size of sort_heap is even then remove all of them
    print("\nTest of the size of sort_heap is even then remove all of them.")
    sort_heap = MinHeap()
    sort_heap.insert(9)
    sort_heap.insert(8)
    sort_heap.insert(7)
    sort_heap.insert(6)
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)
    print("Remove", sort_heap.remove())
    print(sort_heap._heap_list)

    # test of list parameter of size 2000 with duplicate numbers
    print("\nTest of list parameter of size 2000 with duplicate numbers.")
    seed_value = 1
    random.seed(seed_value)
    random_list = [random.randint(1, 1000) for _ in range(2000)]
    print(sorted(random_list)[:10])
    sort_heap = MinHeap(random_list)
    for _ in range(10):
        print("Remove", sort_heap.remove())

    # test of removing from an empty list
    print("\nTest of remove from an empty list.")
    sort_heap = MinHeap()
    print("Remove", sort_heap.remove())


if __name__ == "__main__":
    main()


