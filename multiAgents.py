# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currentGhostStates = currentGameState.getGhostStates()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newFood = newFood.asList()
        "*** YOUR CODE HERE ***"
        reflexTotal = successorGameState.getScore()
        currGhostDist = 0
        newGhostDist = 0
        closestFood = float('inf')
        
        # If the proposed successor is a win, return current score plus score for a win
        if successorGameState.isWin():
          return currentGameState.getScore() + 500

        # If the proposed successor is a loss, return zero
        if successorGameState.isLose():
          return 0
        
        # Calculate distance to ghosts in current and successor states:
        # Trying to "get away from the ghosts", so return higher value if successor is farther away
        for i in range(1,len(currentGhostStates)):
          currGhostDist += util.manhattanDistance(currentGameState.getPacmanPosition(),currentGameState.getGhostPosition(i))
        
        for i in range(1,len(newGhostStates)):
          newGhostDist += util.manhattanDistance(newPos,successorGameState.getGhostPostion())

        if newGhostDist >= currGhostDist:
          reflexTotal += 500

        # Find the closest food and subtract it from total:
        # this is equivalent to rewarding states which move closer to the food
        for food in newFood:
          foodDist = util.manhattanDistance(newPos,food)
          if foodDist < closestFood:
            closestFood = foodDist

        reflexTotal -= 10*closestFood

        # Heavily penalize pacman for doing nothing; time is $$
        if action == Directions.STOP:
          reflexTotal -= 500

        # Give greater value to successors if pacman eats a food in that step
        if len(newFood)< len((currentGameState.getFood()).asList()):
          reflexTotal += 500

        return reflexTotal

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        firstPlayer = 0
        firstDepth = 0 
        action = self.maxFunc(gameState,firstDepth,firstPlayer)[1]
        return action
        util.raiseNotDefined()

    # Function for max nodes
    def maxFunc(self,state,depth,player):
      bestEval = [float("-inf"),Directions.STOP]
      for action in state.getLegalActions(player):
        successorState = state.generateSuccessor(player,action)
        successorDepth = depth + 1
        successorAgent = (depth + 1) % state.getNumAgents()
        successorEval = self.stateEvaluate(successorState,successorDepth,successorAgent)
        if max(bestEval[0],successorEval) == successorEval:
          bestEval[0] = successorEval
          bestEval[1] = action
      return bestEval

    # Function for min nodes
    def minFunc(self,state,depth,player):
      bestEval = [float("inf"),Directions.STOP]
      for action in state.getLegalActions(player):
        successorState = state.generateSuccessor(player,action)
        successorDepth = depth + 1
        successorAgent = (depth + 1) % state.getNumAgents()
        successorEval = self.stateEvaluate(successorState,successorDepth,successorAgent)
        if min(bestEval[0],successorEval) == successorEval:
          bestEval[0] = successorEval
          bestEval[1] = action
      return bestEval

    # This handles the partitioning of min/maxes amoung ghosts
    def stateEvaluate(self,state,depth,player):
      if self.cutoff(state,depth):
        return self.evaluationFunction(state)
      elif player != 0:
        return self.minFunc(state,depth,player)[0]
      else:
        return self.maxFunc(state,depth,player)[0]

    # Tests whether a state is terminal 
    def cutoff(self,state,depth):
      if (depth >= state.getNumAgents()*self.depth) or state.isWin() or state.isLose():
        return True
      return False


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        firstPlayer = 0
        firstDepth = 0 
        alpha = float("-inf")
        beta = float("inf")
        action = self.maxFuncAB(gameState,firstDepth,firstPlayer,alpha,beta)[1]
        return action

    # Function for max nodes
    def maxFuncAB(self,state,depth,player,alpha,beta):
      bestEval = [float("-inf"),Directions.STOP]
      for action in state.getLegalActions(player):
        successorState = state.generateSuccessor(player,action)
        successorDepth = depth + 1
        successorAgent = (depth + 1) % state.getNumAgents()
        successorEval = self.stateEvaluate(successorState,successorDepth,successorAgent,alpha,beta)
        if max(bestEval[0],successorEval) == successorEval:
          bestEval[0] = successorEval
          bestEval[1] = action
        if successorEval > beta:
          return bestEval
        alpha = max(alpha,successorEval)
      return bestEval

    # Function for min nodes
    def minFuncAB(self,state,depth,player,alpha,beta):
      bestEval = [float("inf"),Directions.STOP]
      for action in state.getLegalActions(player):
        successorState = state.generateSuccessor(player,action)
        successorDepth = depth + 1
        successorAgent = (depth + 1) % state.getNumAgents()
        successorEval = self.stateEvaluate(successorState,successorDepth,successorAgent,alpha,beta)
        if min(bestEval[0],successorEval) == successorEval:
          bestEval[0] = successorEval
          bestEval[1] = action
        if successorEval < alpha:
          return bestEval
        beta = min(beta,successorEval)
      return bestEval

    # This handles the partitioning of min/maxes amoung ghosts
    def stateEvaluate(self,state,depth,player,alpha,beta):
      if self.cutoff(state,depth):
        return self.evaluationFunction(state)
      elif player != 0:
        return self.minFuncAB(state,depth,player,alpha,beta)[0]
      else:
        return self.maxFuncAB(state,depth,player,alpha,beta)[0]

    # Tests whether a state is terminal 
    def cutoff(self,state,depth):
      if (depth >= state.getNumAgents()*self.depth) or state.isWin() or state.isLose():
        return True
      return False  

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        firstPlayer = 0
        firstDepth = 0 
        action = self.maxFunc(gameState,firstDepth,firstPlayer)[1]
        return action
        util.raiseNotDefined()

    # Function for max nodes
    def maxFunc(self,state,depth,player):
      bestEval = [float("-inf"),Directions.STOP]
      for action in state.getLegalActions(player):
        successorState = state.generateSuccessor(player,action)
        successorDepth = depth + 1
        successorAgent = (depth + 1) % state.getNumAgents()
        successorEval = self.stateEvaluate(successorState,successorDepth,successorAgent)
        if max(bestEval[0],successorEval) == successorEval:
          bestEval[0] = successorEval
          bestEval[1] = action
      return bestEval

    # Expectation function for minagents
    def expectaMinFunc(self,state,depth,player):
      expectaMin = 0
      for action in state.getLegalActions(player):
        successorState = state.generateSuccessor(player,action)
        successorDepth = depth + 1
        successorAgent = (depth + 1) % state.getNumAgents()
        successorEval = self.stateEvaluate(successorState,successorDepth,successorAgent)
        expectaMin += successorEval
      expectaVal = expectaMin/len(state.getLegalActions(player)) 
      return expectaVal

    # This handles the partitioning of min/maxes amoung ghosts
    def stateEvaluate(self,state,depth,player):
      if self.cutoff(state,depth):
        return self.evaluationFunction(state)
      elif player != 0:
        return self.expectaMinFunc(state,depth,player)
      else:
        return self.maxFunc(state,depth,player)[0]

    # Tests whether a state is terminal 
    def cutoff(self,state,depth):
      if (depth >= state.getNumAgents()*self.depth) or state.isWin() or state.isLose():
        return True
      return False
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    newGhostStates = successorGameState.getGhostStates()
    newCapsules = successorGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newFood = newFood.asList()
    if successorGameState.isWin():
      winPoint = 500
    
    for i in range(1,len(currentGhostStates)):
      currGhostDist += util.manhattanDistance(currentGameState.getPacmanPosition(),currentGameState.getGhostPosition(i))
        
    for i in range(1,len(newGhostStates)):
      newGhostDist += util.manhattanDistance(newPos,successorGameState.getGhostPostion())

    ghostPoints = -(currentGhostDist - newGhostDist)
    
      
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

