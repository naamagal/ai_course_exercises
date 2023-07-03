import numpy as np
import abc
import util
from game import Agent, Action

MIN_AGENT = 0
MAX_AGENT = 1

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        numRows = successor_game_state._num_of_rows
        numCols = successor_game_state._num_of_columns

        "*** YOUR CODE HERE ***"
        board_size = numCols * numRows
        empty_tile_score = get_number_of_empty_tiles(board)/board_size
        adjacentScore = get_num_of_same_adjacent_tile(board, numRows, numCols, max_tile)

        return empty_tile_score*0.3 + adjacentScore*0.6 + score*0.1


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        return genericGetAction(game_state, self.depth, self.evaluation_function, True)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        return genericGetAction(game_state, self.depth, self.evaluation_function, False, True)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        return genericGetAction(game_state, self.depth, self.evaluation_function, True, False)


#  Algorithms for Search in a Tree

def genericGetAction(game_state, depth, evaluation_function, is_minimax=False, is_alpha_beta=False):
    max_depth = depth*2
    eval_func = evaluation_function
    legal_actions = game_state.get_agent_legal_actions()
    current_action_score = 0
    current_action = None

    for action in legal_actions:
        if is_alpha_beta:
            action_score = alpha_beta_helper(game_state.generate_successor(MIN_AGENT, action), max_depth - 1, eval_func,
                                             MIN_AGENT, alpha=-float('Inf'), beta=float('Inf'))
        elif is_minimax:
            action_score = minmax_helper(game_state.generate_successor(MIN_AGENT, action), max_depth-1, eval_func,
                                         MIN_AGENT)
        else:
            action_score = expectimax_helper(game_state.generate_successor(MIN_AGENT, action), max_depth - 1, eval_func,
                                             MIN_AGENT)

        if action_score >= current_action_score:
            current_action_score = action_score
            current_action = action
    return current_action


def expectimax_helper(game_state, max_depth, eval_func, agent_type, cur_score=0):
    if max_depth == 0:
        return eval_func(game_state)

    if agent_type:
        legal_actions = game_state.get_agent_legal_actions()
    else:
        legal_actions = game_state.get_opponent_legal_actions()

    if len(legal_actions) == 0:
        return 0  # ??

    exp_score = 0
    max_score = 0
    for action in legal_actions:
        score = expectimax_helper(game_state.generate_successor(abs(agent_type - 1), action), max_depth-1, eval_func, abs(agent_type - 1), max_score)
        exp_score += score
        if score > max_score:
            max_score = score

    exp_score = exp_score/len(legal_actions)
    if agent_type:
        return max(max_score, cur_score)  # return max value
    else:
        return exp_score   # should return the expected value


def minmax_helper(game_state, max_depth, eval_func, agent_type):
    if max_depth == 0:
        return eval_func(game_state)

    if agent_type:
        legal_actions = game_state.get_agent_legal_actions()
    else:
        legal_actions = game_state.get_opponent_legal_actions()

    if len(legal_actions) == 0:
        return 0

    childrens_score = []
    for action in legal_actions:
        score = minmax_helper(game_state.generate_successor(abs(agent_type - 1), action), max_depth-1,
                              eval_func, abs(agent_type - 1))
        childrens_score.append(score)

    sorted(childrens_score)
    if agent_type:
        return childrens_score[-1]
    else:
        return childrens_score[0]


def alpha_beta_helper(game_state, max_depth, eval_func, agent_type, alpha, beta):
    if max_depth == 0:
        return eval_func(game_state)
    if agent_type:
        legal_actions = game_state.get_agent_legal_actions()
    else:
        legal_actions = game_state.get_opponent_legal_actions()
    if len(legal_actions) == 0:
        return 0  #eval_func(game_state)  # or 0??

    # search with a-b pruning at a tree
    for action in legal_actions:
        action_score = alpha_beta_helper(game_state.generate_successor(abs(agent_type - 1), action), max_depth-1, eval_func, abs(agent_type - 1), alpha, beta)

        if max_depth % 2 == 0:  # the agent is max agent
            alpha = max(alpha, action_score)
            # exit the loop, to do not continue with this branch, when there is no need!
            if beta <= alpha:
                return alpha
        else:  # the agent is min agent
            beta = min(beta, action_score)
            if beta <= alpha:
                return beta

    if max_depth % 2 == 1: # the agent is min agent
        return beta
    # else:
    return alpha # the agent is max agent


########################    Evaluation Functions    ########################

# Evaluation #1 - score of a board
def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


# Evaluation #2 - board is more empty
def get_number_of_empty_tiles(np_board):
    num_of_empty_tiles = 0
    for row in np_board:
            for tile in row:
                if tile == 0:
                    num_of_empty_tiles += 1
    n_rows = len(np_board)
    n_cols = 1
    if n_rows > 0:
        n_cols = len(np_board[0])
    return num_of_empty_tiles


# Evaluation #3 - there is some corner, where the higher values are closer to it
def distance_to_corner(r_idx, c_idx, r_target, c_target):
    return min(abs(r_idx - r_target), abs(c_idx - c_target))


def degree_to_corners(r_idx, c_idx, n_rows, n_cols, r_target, c_target, weight=1):
    dist_to_corner = abs(r_idx - r_target) + abs(c_idx - c_target)
    max_val_degree_to_corner = max(n_rows, n_cols)*2 - 1
    # for example, when weight =1, board of size 3x3,
    # target cornet is [0,0], will return 'degree_to_corner'=
    # 5 4 3
    # 4 3 2
    # 3 2 1
    val_degree_to_corner = max_val_degree_to_corner - dist_to_corner
    return val_degree_to_corner * weight


def idx_max_board(np_board):
    max_tile_idx = np.argmax(np_board)
    n_rows = len(np_board)
    n_cols = 0
    if n_rows > 0:
        n_cols = len(np_board[0])
    r_idx = int(max_tile_idx / n_cols)
    c_idx = max_tile_idx - r_idx * n_cols
    return [r_idx, c_idx]


def get_weight_corner(np_board):
    # for two board with the same number of empty tiles, it will be a good measurement
    n_rows = len(np_board)
    n_cols = 0
    if n_rows > 0:
        n_cols = len(np_board[0])

    # find best corner
    corners = [[0, 0], [0, n_cols-1], [n_rows-1, 0], [n_rows-1, n_cols-1]]
    [r_max_idx, c_max_idx] = idx_max_board(np_board)
    closest_corner_dist = max(n_rows, n_cols)
    closest_corner = [0,0]
    for corner in corners:
        dist = distance_to_corner(r_max_idx, c_max_idx, corner[0], corner[1]);
        if dist < closest_corner_dist:
            closest_corner_dist = dist
            closest_corner = corner

    # find weight matrix to chosen (best) corner
    weigth_board = np.zeros([n_rows, n_cols])
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            weigth_board[r_idx][c_idx] = degree_to_corners(r_idx, c_idx, n_rows, n_cols, closest_corner[0], closest_corner[1])
    # normalize matrix
    weigth_board = weigth_board/sum(sum(weigth_board))
    return weigth_board


# Evaluation #4 - number of adjacent tiles that are similar.
def get_num_of_same_adjacent_tile(board, numRows, numCols, maxVal):
    # can change it to the degree of how similar they are...
    score = 0
    for row in board:
        prevVal = 0
        for column in row:
            if column == prevVal:
                #score += 1
                score += column/maxVal
            elif column != 0:
                prevVal = column
    for column in range(numCols):
        prevVal = 0
        for row in range(numRows):
            if board[row][column] == prevVal:
                #score += 1
                score += prevVal/maxVal
            elif board[row][column] != 0:
                prevVal = board[row][column]
    if score > 0:
        return score
    return 0

# Evaluation #5 - monoonic in a row/ column TODO
def get_num_of_non_monotonic(row, max_tile):
    num_of_non_monotonic = 0
    if len(row) < 3:
        return 0
    for num in range(2, len(row)):
        if (row[num] - row[num-1]) * (row[num - 1] - row[num - 2]) < 0:
            num_of_non_monotonic += max([row[num], row[num-1], row[num-2]])/max_tile

    return num_of_non_monotonic

def get_monotonic_score(board, numRows, numCols, board_size, max_tile):
    non_monotonic_tiles = 0
    row_vals = []
    for row in board:
        for tile in row:
            if tile != 0:
                row_vals.append(tile)
        non_monotonic_tiles += get_num_of_non_monotonic(row_vals, max_tile)
        row_vals = []
    col_vals = []
    for column in range(numCols):
        for row in range(numRows):
            if board[row][column] != 0:
                col_vals.append(board[row][column])
        non_monotonic_tiles += get_num_of_non_monotonic(col_vals, max_tile)
        col_vals = []
    return board_size - non_monotonic_tiles


# Evaluation #6 - one of the edges (row or column) of the higher value in the matrix is FULL TODO
def get_high_edge_score(board):
    return


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    numRows = current_game_state._num_of_rows
    numCols = current_game_state._num_of_columns
    board_size = numCols * numRows
    max_opt_val = 1024  # ???

    #cur_score = score_evaluation_function(current_game_state) / max_opt_val
    cur_score = 0
    empty_tile_score = get_number_of_empty_tiles(board) / board_size
    similar_adj_score = get_num_of_same_adjacent_tile(board, numRows, numCols, max_tile)/ board_size
    mono_score = get_monotonic_score(board, numRows, numCols, board_size, max_tile)/(numRows * numCols)
    #higher_at_corner_score = sum(sum(np.multiply(get_weight_corner(board), board)/max_tile))
    higher_at_corner_score = 0

    score = (empty_tile_score * .15) + (similar_adj_score * .15) + (mono_score * .7) + (higher_at_corner_score * 0) + (cur_score * 0)
    score = score
    return score


# Abbreviation
better = better_evaluation_function