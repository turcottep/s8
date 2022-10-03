# Credit for this: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
from warnings import warn
import heapq
from swiplserver import PrologMQI


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


# this the new one
def astar(start, end):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """
    with PrologMQI() as mqi:
        with PrologMQI() as mqi_file:
            with mqi_file.create_thread() as prolog_thread:

                result = prolog_thread.query("[prolog/maze].")
                # print("[prolog/maze].", result)

                result = prolog_thread.query("[prolog/functions].")
                # print("[prolog/functions].", result)

                # Create start and end node
                start_node = Node(None, start)
                start_node.g = start_node.h = start_node.f = 0
                end_node = Node(None, end)
                end_node.g = end_node.h = end_node.f = 0

                # Initialize both open and closed list
                open_list = []
                closed_list = []

                # Heapify the open_list and Add the start node
                heapq.heapify(open_list)
                heapq.heappush(open_list, start_node)

                # Adding a stop condition
                outer_iterations = 0
                max_iterations = 25 * 15  # arbitrary value

                # Loop until you find the end
                while len(open_list) > 0:
                    outer_iterations += 1

                    if outer_iterations > max_iterations:
                        # if we hit this point return the path such as it is
                        # it will not contain the destination
                        warn("giving up on pathfinding too many iterations")
                        return return_path(current_node)

                    # Get the current node
                    current_node = heapq.heappop(open_list)
                    closed_list.append(current_node)

                    # Found the goal
                    if current_node == end_node:
                        return return_path(current_node)

                    # Generate children
                    children = []

                    query_str = (
                        "passage("
                        + str(current_node.position[0])
                        + ", "
                        + str(current_node.position[1])
                        + ", X, Y)."
                    )

                    # print("query_str", query_str)

                    all_moves = prolog_thread.query(query_str)
                    # print("all_moves", all_moves)
                    moves_clean = []
                    for move in all_moves:
                        moves_clean.append((move["X"], move["Y"]))
                    # print("moves_clean", moves_clean)

                    for node_position in moves_clean:  # Adjacent squares
                        # Create new node
                        new_node = Node(current_node, node_position)

                        # Append
                        children.append(new_node)

                    # Loop through children
                    for child in children:
                        # Child is on the closed list
                        if (
                            len(
                                [
                                    closed_child
                                    for closed_child in closed_list
                                    if closed_child == child
                                ]
                            )
                            > 0
                        ):
                            continue

                        # Create the f, g, and h values
                        child.g = current_node.g + 1
                        child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                            (child.position[1] - end_node.position[1]) ** 2
                        )
                        child.f = child.g + child.h

                        # Child is already in the open list
                        if (
                            len(
                                [
                                    open_node
                                    for open_node in open_list
                                    if child.position == open_node.position
                                    and child.g > open_node.g
                                ]
                            )
                            > 0
                        ):
                            continue

                        # Add the child to the open list
                        heapq.heappush(open_list, child)

                warn("Couldn't get a path to destination")
                return None


def example():

    # A-star
    starting_positon = (1, 0)
    ending_position = (22, 15)

    # raise Exception("stop")
    path = astar(starting_positon, ending_position)
    print(path)


if __name__ == "__main__":
    example()
