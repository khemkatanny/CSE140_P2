import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        # keep track of total score in the game
        # feep track of food distance
        # keep track of number of capsules left
        # keep track of food left

        total_score = 0
        food_List = successorGameState.getFood().asList()
        food_left = len(food_List)

        active_ghost = currentGameState.getGhostStates()
        ghost_list = []
        for ghostState in active_ghost:
            ghost_position = ghostState.getPosition()
            # print("ghost_position: ", ghost_position)
            # print("newPosition: ", newPosition)
            ghost_list.append((ghost_position[0], ghost_position[1]))
            ghost_list.append((ghost_position[0], ghost_position[1] - 1))
            ghost_list.append((ghost_position[0], ghost_position[1] + 1))
            ghost_list.append((ghost_position[0] - 1, ghost_position[1]))
            ghost_list.append((ghost_position[0] + 1, ghost_position[1]))
        if food_left > 0:
            total_score -= abs(newPosition[0] - food_List[0][0])
            + abs(newPosition[1] - food_List[0][1])
        # if food left is less than old food
        if food_left < len(oldFood):
            total_score = 0
        if newPosition in ghost_list:
            total_score -= 1000
        return total_score

        return successorGameState.getScore()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    # CHECK > DEPTH, GAME IS A WIN AND GAME IS A LOSS
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        scores = []
        # legalMoves = gameState.getLegalActions()
        for action in gameState.getLegalActions(0):
            # add as tuple not list
            scores.append((self.minimax_helper(1, 0,
            gameState.generateSuccessor(0, action), 0), action))
        best_score = max(scores)
        # print("best_score: ", best_score)
        return best_score[1]

    def minimax_helper(self, max_agent, max_index, gameState, depth):
        if depth >= self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)

        if max_agent == gameState.getNumAgents():
            max_agent = 0
            max_index += 1
        if max_agent == gameState.getNumAgents() - 1:
            depth += 1
        # print("depth: ", depth)

        actions = gameState.getLegalActions(max_agent)
        # print("actions: ", actions)
        if len(actions) == 0:
            return self.getEvaluationFunction()(gameState)
        v = -float("inf") if max_agent == 0 else float("inf")
        for action in actions:
            if max_agent == 0:
                v = max(v, self.minimax_helper(max_agent + 1, max_index,
                gameState.generateSuccessor(max_agent, action), depth))
            else:
                v = min(v, self.minimax_helper(max_agent + 1, max_index,
                gameState.generateSuccessor(max_agent, action), depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        scores = []
        alpha, beta = -float("inf"), float("inf")
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        for action in gameState.getLegalActions(0):
            result = ((self.ab_helper(alpha, beta, 1, 0,
            gameState.generateSuccessor(0, action), 0), action))
            scores.append((result[0], action))
            alpha, beta = result, result
        best_score = max(scores)
        # print("best_score: ", best_score)
        return best_score[1]

    def ab_helper(self, alpha, beta, max_agent, max_index, gameState, depth):
        if depth >= self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)

        if max_agent == gameState.getNumAgents():
            max_agent = 0
            max_index += 1
        if max_agent == gameState.getNumAgents() - 1:
            depth += 1
        # print("depth: ", depth)

        actions = gameState.getLegalActions(max_agent)
        if len(actions) == 0:
            return self.getEvaluationFunction()((gameState), alpha, beta)
        v = -float("inf") if max_agent == 0 else float("inf")
        for action in actions:
            if max_agent == 0:
                v = max(v, self.ab_helper(alpha, beta, max_agent + 1, max_index,
                gameState.generateSuccessor(max_agent, action), depth))
            else:
                v = min(v, self.ab_helper(alpha, beta, max_agent + 1, max_index,
                gameState.generateSuccessor(max_agent, action), depth))
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        scores = []
        for action in gameState.getLegalActions(0):
            # add as tuple not list
            scores.append((self.expmax_helper(1, 0,
            gameState.generateSuccessor(0, action), 0), action))
        best_score = max(scores)
        # print("best_score: ", best_score)
        return best_score[1]

    def expmax_helper(self, max_agent, max_index, gameState, depth):
        if depth >= self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)

        if max_agent == gameState.getNumAgents():
            max_agent = 0
            max_index += 1
        if max_agent == gameState.getNumAgents() - 1:
            depth += 1
        # print("depth: ", depth)

        actions = gameState.getLegalActions(max_agent)
        # print("actions: ", actions)
        if len(actions) == 0:
            return self.getEvaluationFunction()(gameState)
        v = -float("inf") if max_agent == 0 else float("inf")
        for action in actions:
            if max_agent == 0:
                v = max(v, self.expmax_helper(max_agent + 1, max_index,
                gameState.generateSuccessor(max_agent, action), depth))
            else:
                v = min(v, self.expmax_helper(max_agent + 1, max_index,
                gameState.generateSuccessor(max_agent, action), depth))
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    total_score = 0
    newPosition = currentGameState.getPacmanPosition()
    food_List = currentGameState.getFood().asList()
    food_left = len(food_List)

    if food_left > 0:
        total_score -= abs(newPosition[0] - food_List[0][0])
        + abs(newPosition[1] - food_List[0][1])
        # print("total_score: ", total_score)
    return currentGameState.getScore() + total_score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
