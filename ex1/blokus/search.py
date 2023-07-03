"""
In search.py, you will implement generic search algorithms
"""
import util
# import PCF.searchAgents as sa

SUCC_PROB_IDX = 0
ACTION_IDX = 1
COST_IDX = 2


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


class ActionTuple:
    def __init__(self, board_move_cost, parent_cost):
        self.board_state = board_move_cost[SUCC_PROB_IDX]
        self.move = board_move_cost[ACTION_IDX]
        self.priority = board_move_cost[COST_IDX] + parent_cost
        self.tuple = board_move_cost


    def __eq__(self, other):
        return self.priority == other.priority

    """
    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

    def __gt__(self, other):
        return self.key > other.key

    def __ge__(self, other):
        return self.key >= other.key
    """


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def search_helper(problem, fringe, parents_list, path, heuristic=0):
    visited_set = set()
    start_state = ActionTuple((problem.get_start_state(), None, 0), 0)
    if heuristic == 0:
        fringe.push(start_state)
    else:
        fringe.push(start_state, a_star_function(start_state, heuristic, problem))

    while not fringe.isEmpty():
        succ_state = fringe.pop()

        if succ_state.board_state not in visited_set:
            visited_set.add(succ_state.board_state)

            if problem.is_goal_state(succ_state.board_state):
                next_stage = succ_state.tuple
                while next_stage[ACTION_IDX]:
                    path.append(next_stage[ACTION_IDX])
                    next_stage = parents_list[next_stage].tuple
                return True

            all_successors = problem.get_successors(succ_state.board_state)
            # sort all_successors by the current cost of the board
            #if heuristic == 0:
            #    all_successors = all_successors.sort(key=lambda x: x[SUCC_PROB_IDX].scores)
            #    all_successors = all_successors.reverse()

            for new_succ in all_successors:
                action_new_succ = ActionTuple(new_succ, succ_state.priority)
                if heuristic == 0:
                    fringe.push(action_new_succ)
                else:
                    fringe.push(action_new_succ, a_star_function(action_new_succ, heuristic, problem))
                parents_list[action_new_succ.tuple] = succ_state
    return False

"""
def a_star_helper(problem, fringe, parents_list, path, heuristic):
    visited_set = set()
    start_state = ActionTuple((problem.get_start_state(), None, 0), 0)
    fringe.push(start_state, a_star_function(start_state, heuristic, problem))

    while not fringe.isEmpty():
        succ_state = fringe.pop()

        if succ_state.board_state not in visited_set:
            visited_set.add(succ_state.board_state)
            board_state = succ_state.board_state
            if problem.is_goal_state(board_state):
                next_stage = succ_state.tuple
                while next_stage[ACTION_IDX]:
                    path.append(next_stage[ACTION_IDX])
                    next_stage = parents_list[next_stage].tuple
                return True

            all_successors = problem.get_successors(succ_state.board_state)
            for new_succ in all_successors:
                action_new_succ = ActionTuple(new_succ, succ_state.priority)
                fringe.push(action_new_succ, a_star_function(action_new_succ, heuristic, problem))
                parents_list[action_new_succ.tuple] = succ_state
    return False
"""


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("helooooooooooooo")
    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.isGoalState(problem.get_start_state()))
    print("Start's successors:", problem.getSuccessors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    # init fringe
    state = problem.get_start_state
    funcArgs = state
    fringe = util.Stack()
    parents_list = {}
    path = []
    search_helper(problem, fringe, parents_list, path)
    # reversing the list of moves, such that, it will be a new list (not an iterator)
    return path[::-1]


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    # init fringe
    state = problem.get_start_state
    funcArgs = state
    fringe = util.Queue()
    parents_list = {}
    path = []
    search_helper(problem, fringe, parents_list, path)

    return path[::-1]
    # util.raiseNotDefined()

# *************************** COST FUNCTIONs *********************** #


def state_cost_function(state):
    """
    :param state:
    :return: the cost until this state (past)
    """
    return state.priority


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """

    "*** YOUR CODE HERE ***"
    state = problem.get_start_state
    funcArgs = state
    fringe = util.PriorityQueueWithFunction(state_cost_function)
    parents_list = {}
    path = []
    search_helper(problem, fringe, parents_list, path)

    return path[::-1]
    # util.raiseNotDefined()

def a_star_function(state, heuristic, problem):
    return state_cost_function(state) + heuristic(state.board_state, problem)


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # init fringe

    fringe = util.PriorityQueue()
    parents_list = {}
    path = []
    search_helper(problem, fringe, parents_list, path, heuristic)
    #a_star_helper(problem, fringe, parents_list, path, heuristic)

    return path[::-1]
    # util.raiseNotDefined()


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search