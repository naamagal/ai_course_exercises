from board import Board
from search import SearchProblem, dfs, astar, ucs, null_heuristic
import util
import pieces
import numpy as np


X_POS = 0
Y_POS = 1

# (game_min_dist(start_pt, target), start_pt, target, current_board.deepcopy(), vacant_targets.deep_copy()
DIST = 0
START_PT = 1
TARGET_PT = 2
BOARD= 3
PIECE_LIST = 4
PATH = 5

def get_all_optional_next_start_points(state, player):
    """
    :return: all location which can be a start point for the nxt moves, from this current state
    """
    w = state.board_w - 1
    h = state.board_h - 1
    optional_pos_lst = []
    for x_idx in range(w):
        for y_idx in range(h):
            if state.check_tile_legal(player, x_idx, y_idx) and state.check_tile_attached(player, x_idx, y_idx):
                optional_pos_lst.append((x_idx, y_idx))
    return optional_pos_lst


def targets_to_assign(state, targets):
    t_to_assign = []
    for target in targets:
        if state.get_position(target[0], target[1]) == -1:
            t_to_assign.append(target)
    return t_to_assign


def all_targets_assigned(state, targets):
    if len(targets_to_assign(state, targets)) == 0:
            return True
    return False


def cost_of_action(actions, board, player):
    board_state = board.__copy__()
    for action in actions:
        # or maybe assuming all of these action are legal action after another??
        if not board_state.check_move_valid(player, action):
            util.raiseNotDefined()
        board_state = board_state.do_move(player, action)
    return board_state.score(player)


def game_min_dist(start_pt, target_pt):
    return max(abs(start_pt[X_POS] - target_pt[X_POS]), abs(start_pt[Y_POS] - target_pt[Y_POS])) + 1


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        # return list of successors
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.player = 0
        self.targets = []
        self.define_corners()
        self.is_uniform_cost = False
        # for heurstric:
        #  self.direction_min_moves = [0, 0, 0, 0]  # [up, down, left, right]
        #  self.is_corners_assigned = [False, False, False, False]  # [(0,0), (w,0), (0,h), (w,h)]

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def define_corners(self):
        """
        :param state: Board obj
        :return: list of the 4 corners (x,y) location on the board
        """
        self.targets.append((0, 0))
        self.targets.append((self.board.board_w-1, 0))
        self.targets.append((0, self.board.board_h-1))
        self.targets.append((self.board.board_w-1, self.board.board_h-1))

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        """
        in this problem, we want to return true only when all corners are not empty!
        """
        return all_targets_assigned(state, self.targets)
        # util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        if self.is_uniform_cost:
            return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]
        else:
            return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return cost_of_action(actions, self.board, self.player)


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    #  Longest path from closest point, option #1
    cur_start_points = get_all_optional_next_start_points(state, problem.player)

    #  Divide board to 4 squares, option #2
    board_squares_dist = 0
    square_w = int(problem.board.board_w / 2)
    square_h = int(problem.board.board_h / 2)
    square_size = min(square_w, square_h)
    max_value = max(problem.board.board_w, problem.board.board_h)
    t_to_assign = targets_to_assign(state, problem.targets)
    longest_path_dist = 0
    min_board_squares_dist = 0
    for target_pt in t_to_assign:
        min_path_dist = max_value
        board_squares_dist = max_value
        for start_pt in cur_start_points:
            cur_dist = game_min_dist(start_pt, target_pt)
            #  if inside the square- adding cur dist. if it is outside the square- that only the size of the square.
            board_squares_dist = min(cur_dist, square_size, board_squares_dist)
            #  find game_min_dist for each target
            if cur_dist < min_path_dist:
                min_path_dist = cur_dist
            if min_path_dist == 1 or board_squares_dist == 1:
                return 1
        min_board_squares_dist += board_squares_dist
        # Take the farther target, because anyway we will need to reach that one!
        if min_path_dist > longest_path_dist:
            longest_path_dist = min_path_dist
    # both are lower boundaries, but each can be more tight in different cases...
    return longest_path_dist  #max(longest_path_dist, min_board_squares_dist)


class BlokusCoverProblem(SearchProblem):

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.player = 0
        self.is_uniform_cost = False

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return all_targets_assigned(state, self.targets)
        #util.raiseNotDefined()

    def update_board(self, other_board):
        self.board = other_board.deep_copy()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        if self.is_uniform_cost:
            return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]
        else:
            return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return cost_of_action(actions, self.board, self.player)


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    return blokus_corners_heuristic(state, problem)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.player = 0
        self.starting_point = starting_point

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return all_targets_assigned(state, self.targets)

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def remaining_pieces(self, action_operated, piece_list):
        remaining_piece_list = piece_list.pieces.copy()
        for action in action_operated:
            piece = action.piece
            piece_index = action.piece_index
            del remaining_piece_list[piece_index]
            #remaining_piece_list.remove(piece)
        return remaining_piece_list

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []
        while ....
            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace

        "*** YOUR CODE HERE ***"
        """
        current_board = self.board.__copy__()
        targets = self.targets.copy() # should be a queue
        #vacant_targets = targets.deep_copy()
        board_w = current_board.board_w
        board_h = current_board.board_h

        final_actions_path = []
        fringe = util.Stack()  # to use Queue
        curr_path = []
        all_dist_from_starts = []
        for target in targets:
            dist = game_min_dist(self.starting_point, target)
            all_dist_from_starts.append((dist, self.starting_point, target, current_board.__copy__(), current_board.piece_list.pieces, curr_path))

        all_dist_from_starts_sorted = sorted(all_dist_from_starts, key=lambda x: x[0])
        all_dist_from_starts_sorted = all_dist_from_starts_sorted[::-1]
        #if current_board.get_position(target[0], target[1]) == -1:
        for item in all_dist_from_starts_sorted:
            fringe.push(item)

        while not fringe.isEmpty():
            state = fringe.pop()
            start_pt = state[START_PT]
            current_board = state[BOARD]
            cur_target = state[TARGET_PT]
            piece_list = state[PIECE_LIST]
            piece_list_obj = pieces.PieceList()
            piece_list_obj.pieces = piece_list

            cover_problem = BlokusCoverProblem(board_w, board_h, piece_list_obj, start_pt, [cur_target])
            cover_problem.is_uniform_cost = True  # ??
            moves_to_cover_target = dfs(cover_problem) # astar with some smart heuristic,plus, not searching for an optimal solution
            if moves_to_cover_target:  # when there is no solution- list is empty
                new_board = current_board.__copy__()
                for move in moves_to_cover_target:
                    new_board = new_board.do_move(0, move)
                remaining_piece_list = self.remaining_pieces(moves_to_cover_target, piece_list)

                new_path = state[PATH]
                for move in moves_to_cover_target:
                    new_path.append(move)
                final_actions_path = new_path

                if all_targets_assigned(new_board, targets):
                    return final_actions_path

                for new_target in targets:
                    if new_board.get_position(new_target[0], new_target[1]) == -1:
                        all_dist_from_starts = [(game_min_dist(new_start_pt, target), new_start_pt, new_target, new_board.__copy__(), remaining_piece_list, new_path) for new_start_pt in get_all_optional_next_start_points(new_board,self.player)]
                        all_dist_from_starts_sorted = sorted(all_dist_from_starts, key=lambda x: x[0])
                        all_dist_from_starts_sorted = all_dist_from_starts_sorted[::-1]
                        for item in all_dist_from_starts_sorted:
                            fringe.push(item)
            """
            #old
            if current_board.get_position(cur_target[0], cur_target[1]) == -1:
                all_dist_from_starts = [(game_min_dist(start_pt, target), start_pt) for start_pt in
                                        get_all_optional_next_start_points(current_board)]
                all_dist_from_starts_sorted = sorted(all_dist_from_starts, key=lambda x: x[0])
                for start_pt in all_dist_from_starts_sorted:
                    # save (board, path) each time we are trying to find a path for specific target    

                    cover_problem = BlokusCoverProblem(self.board.board_w, self.board.board_h, self.board.piece_list, start_pt, [target])
                    # use current board (update problem)
                    cover_problem.update_board(current_board)
                    # I don't care about the number of tiles in a piece.
                    cover_problem.is_uniform_cost = True

                    actions_to_cover_target = dfs(cover_problem)
                    if actions_to_cover_target:
                        for action in actions_to_cover_target:
                            backtrace.append(action)
                            current_state = current_board.do_move(0, action)
                        break  # out of start point
                if not actions_to_cover_target:
                    #target.push(target)
                    backtrace = [] # resart from the next option
            """
        return final_actions_path


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
