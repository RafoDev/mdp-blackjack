"""
 
  Nombre : Rafael David Peñalva
  Codigo : 192-10-47133

"""

import math
import random
from collections import defaultdict
import util


# **********************************************************
# **            PART 01 Modeling BlackJack                **
# **********************************************************


class BlackjackMDP(util.MDP):
    """
    The BlackjackMDP class is a subclass of MDP that models the BlackJack game as a MDP
    """

    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    def startState(self):
        """
         Return the start state.
         Each state is a tuple with 3 elements:
           -- The first element of the tuple is the sum of the cards in the player's hand.
           -- If the player's last action was to peek, the second element is the index
              (not the face value) of the next card that will be drawn; otherwise, the
              second element is None.
           -- The third element is a tuple giving counts for each of the cards remaining
              in the deck, or None if the deck is empty or the game is over (e.g. when
              the user quits or goes bust).
        """
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    def actions(self, state):
        """
        Return set of actions possible from |state|.
        You do not must to modify this function.
        """
        return ['Take', 'Peek', 'Quit']

    def succAndProbReward(self, state, action):
        """
        Given a |state| and |action|, return a list of (newState, prob, reward) tuples
        corresponding to the states reachable from |state| when taking |action|.
        A few reminders:
         * Indicate a terminal state (after quitting, busting, or running out of cards)
           by setting the deck to None.
         * If |state| is an end state, you should return an empty list [].
         * When the probability is 0 for a transition to a particular new state,
           don't include that state in the list returned by succAndProbReward.
        """
        # BEGIN_YOUR_CODE
        # print(state)
        reachableStates = []
        total, indexOfSpiedCard, deckMultiplicity = state

        attempToSpyTwice = (action == 'Peek') and (indexOfSpiedCard is not None)
        isDeckEmpty = deckMultiplicity is None

        if isDeckEmpty or attempToSpyTwice:
            return []

        if action == 'Take':
            if indexOfSpiedCard is None:
                for i in range(len(self.cardValues)):
                    if deckMultiplicity[i] > 0:
                        newDeckMultiplicity = list(deckMultiplicity)
                        prob = newDeckMultiplicity[i] / float(sum(deckMultiplicity))
                        reward = 0
                        newTotal = total + self.cardValues[i]
                        if newTotal > self.threshold:
                            # [TERMINAL STATE] - Bankrupt Case
                            newState = (newTotal, None, None)
                        else:
                            newDeckMultiplicity[i] -= 1
                            if sum(newDeckMultiplicity) > 0:
                                newState = (newTotal, None, tuple(
                                    newDeckMultiplicity))
                            else:
                                # [TERMINAL STATE] - Empty Deck Case
                                reward = newTotal
                                newState = (newTotal, None, None)
                        reachableStates.append((newState, prob, reward))
            else:
                prob = 1
                reward = 0
                total += self.cardValues[indexOfSpiedCard]

                if total > self.threshold:
                    # [TERMINAL STATE] - Bankrupt Case
                    newState = (total, None, None)
                else:
                    newDeckMultiplicity = list(deckMultiplicity)
                    newDeckMultiplicity[indexOfSpiedCard] -= 1
                    if sum(newDeckMultiplicity) > 0:
                        newState = (total, None, tuple(newDeckMultiplicity))
                    else:
                        # [TERMINAL STATE] - Empty Deck Case
                        reward = total
                        newState = (total, None, None)

                reachableStates.append((newState, prob, reward))
        if action == 'Peek':
            for i in range(len(self.cardValues)):
                if deckMultiplicity[i] > 0:
                    newDeckMultiplicity = list(deckMultiplicity)
                    prob = newDeckMultiplicity[i] / float(sum(deckMultiplicity))
                    newState = (total, i, deckMultiplicity)
                    reachableStates.append((newState, prob, -self.peekCost))
        if action == 'Quit':
            prob = 1
            reward = total
            newState = (total, None, None)
            reachableStates.append((newState, 1, reward))

        # print(reachableStates)
        return reachableStates

        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        """
        Return the descount  that is 1
        """
        return 1

# **********************************************************
# **                    PART 02 Value Iteration           **
# **********************************************************


class ValueIteration(util.MDPAlgorithm):
    """ Asynchronous Value iteration algorithm """

    def __init__(self):
        self.pi = {}
        self.V = {}

    def solve(self, mdp, epsilon=0.001):
        """
        Solve the MDP using value iteration.  Your solve() method must set
        - self.V to the dictionary mapping states to optimal values
        - self.pi to the dictionary mapping states to an optimal action
        Note: epsilon is the error tolerance: you should stop value iteration when
        all of the values change by less than epsilon.
        The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
        """
        mdp.computeStates()

        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            return sum(prob * (reward + mdp.discount() * V[newState])
                       for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action)
                                for action in mdp.actions(state))[1]
            return pi
        V = defaultdict(float)  # state -> value of state

        # Implement the main loop of Asynchronous Value Iteration Here:
        # BEGIN_YOUR_CODE

        def checkConvergence(states, V, newV):
            for state in states:
                if abs(newV[state] - V[state]) > epsilon:
                    return False
            return True

        hasConverged = False

        while not hasConverged:
            newV = {}
            for state in mdp.states:
                newV[state] = max((computeQ(mdp, V, state, action))
                                  for action in mdp.actions(state))

            hasConverged = checkConvergence(mdp.states, V, newV)
            V = newV
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

        # Extract the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        # print("ValueIteration: %d iterations" % numIters)
        self.pi = pi
        self.V = V


# First MDP
MDP1 = BlackjackMDP(cardValues=[1, 5],
                    multiplicity=2, threshold=10, peekCost=1)

# Second MDP
MDP2 = BlackjackMDP(cardValues=[1, 5],
                    multiplicity=2, threshold=15, peekCost=1)

# Solver

# MDPSolver = ValueIteration()
# MDPSolver.solve(MDP1)
# print("MDP1 - pi")
# print(MDPSolver.pi)
# print("MDP1 - V")
# print(MDPSolver.V)


def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action for at least 10% of the states.
    """
    # BEGIN_YOUR_CODE
    return BlackjackMDP(cardValues=[5, 20],
                    multiplicity=10, threshold=20, peekCost=1)
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE


# MDPSolver = ValueIteration()
# MDPSolver.solve(peekingMDP())
# print(MDPSolver.pi)



# **********************************************************
# **                    PART 03 Q-Learning                **
# **********************************************************

class QLearningAlgorithm(util.RLAlgorithm):
    """
    Performs Q-learning.  Read util.RLAlgorithm for more information.
    actions: a function that takes a state and returns a list of actions.
    discount: a number between 0 and 1, which determines the discount factor
    featureExtractor: a function that takes a state and action and returns a
    list of (feature name, feature value) pairs.
    explorationProb: the epsilon value indicating how frequently the policy
    returns a random action
    """

    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    def getQ(self, state, action):
        """
         Return the Q function associated with the weights and features
        """
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        """
        Produce an action given a state, using the epsilon-greedy algorithm: with probability
        |explorationProb|, take a random action.
        """
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    def getStepSize(self):
        """
        Return the step size to update the weights.
        """
        return 1.0 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state, action, reward, newState):
        """
         We will call this function with (s, a, r, s'), which you should use to update |weights|.
         You should update the weights using self.getStepSize(); use
         self.getQ() to compute the current estimate of the parameters.

         HINT: Remember to check if s is a terminal state and s' None.
        """
        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


def identityFeatureExtractor(state, action):
    """
    Return a single-element list containing a binary (indicator) feature
    for the existence of the (state, action) pair.  Provides no generalization.
    """
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


# Large test case
largeMDP = BlackjackMDP(
    cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

# **********************************************************
# **        PART 03-01 Features for Q-Learning             **
# **********************************************************


def blackjackFeatureExtractor(state, action):
    """
    You should return a list of (feature key, feature value) pairs.
    (See identityFeatureExtractor() above for a simple example.)
    """
    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
