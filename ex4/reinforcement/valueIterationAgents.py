# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util
import numpy as np

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter()  # A Counter is a dict with default 0
    "*** YOUR CODE HERE ***"
     # On start: init V(s)=0 for each state s.
    for state in mdp.getStates():
        self.values[state] = 0.0
    # dictionary for each (state, action), tell me the value I can get out of it..
    self.q_values = util.Counter()
    # policy given a state is the action I should do, for each state..
    self.policy = util.Counter()

    # Updating self.value with givven number of iterations
    for i in range(0, self.iterations):
        new_vals = self.values.copy()
        for state in mdp.getStates():
            if self.mdp.isTerminal(state):
                self.values[state] = 0
                continue
            max_val = -float('inf')
            pos_actions = mdp.getPossibleActions(state)
            for action in pos_actions:
                # Returns list of (nextState, prob) pairs
                nextState_prob = mdp.getTransitionStatesAndProbs(state, action)
                val = 0.0
                q_val = 0.0
                for (next_state, prob) in nextState_prob:
                    # action and the second 'next_state' doesn't matters!
                    val += prob*self.values[next_state]
                    q_val += prob * (self.mdp.getReward(state, action, next_state) +
                                     self.discount * self.values[next_state])
                val *= self.discount
                self.q_values[(state, action)] = q_val
                if val > max_val:
                    max_val = val
                    best_act = action
                # choose the action with the maximum reward, do not consider the discount effect!
            self.policy[state] = best_act
            new_vals[state] = max_val + self.get_reward_from_state(state)
        self.values = new_vals


  def get_reward_from_state(self, state):
      if self.mdp.isTerminal(state):
          return 0
      next_actions = self.mdp.getPossibleActions(state)
      s_p_pairs = self.mdp.getTransitionStatesAndProbs(state, next_actions[0])
      s_p_pair = s_p_pairs[0]  # gets the first pair
      # getReward cares only about the cur state, why??
      return self.mdp.getReward(state, next_actions[0], s_p_pair[0])

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iterationv passes).
      Note that value iteration does not necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
        return 0
    return self.q_values[(state, action)]

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    return self.policy[state]

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
