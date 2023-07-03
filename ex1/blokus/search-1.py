"""
In search.py, you will implement generic search algorithms
"""
import util
# import PCF.searchAgents as sa

SUCC_PROB_IDX = 0
ACTION_IDX = 1
COST_IDX = 2

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


def depth_first_search_recursive(problem, state, fringe, visited_set, moves_path):
    visited_set.add(state)
    all_successors = problem.get_successors(state)
    [fringe.push(succ) for succ in all_successors]

    while not fringe.isEmpty():
        succ_state = fringe.pop()
        if problem.is_goal_state(state):
            return True

        is_visited = False
        # if visited not in visited_set:
        for visited in visited_set:
            if visited == succ_state[SUCC_PROB_IDX]:
                is_visited = True
                break
        if not is_visited:
            if depth_first_search_recursive(problem, succ_state[SUCC_PROB_IDX], fringe, visited_set, moves_path):
                # insert only action to solution path
                moves_path.append(succ_state[ACTION_IDX])
                return True
    return False


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
    fringe = util.Stack()
    fringe.push(problem.get_start_state())
    visited_set = set()
    moves_path = []
    depth_first_search_recursive(problem, problem.get_start_state(), fringe, visited_set, moves_path)
    return reversed(moves_path)
    # util.raiseNotDefined()


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    # init fringe
    fringe = util.Queue()
    visited_set = set()
    moves_path = []
    depth_first_search_recursive(problem, problem.get_start_state(), fringe, visited_set, moves_path)

    return reversed(moves_path)
    # util.raiseNotDefined()


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # init fringe
    fringe = util.PriorityQueueWithFunction(heuristic(problem.get_start_state(), problem))
    visited_set = set()
    moves_path = []
    depth_first_search_recursive(problem, problem.get_start_state(), fringe, visited_set, moves_path)

    return reversed(moves_path)
    # util.raiseNotDefined()



# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
