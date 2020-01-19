# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # specify which data structure to use in DFS
    stack = util.Stack()
    # get the starting state
    startingState = problem.getStartState()
    # as the question required, nodes need to contain other necessary elements, not just state (coordinate)
    # since getSuccessors() function returns states in the form of (coord., act., cost), nodes will be indicated this way
    # however, for uninformed search, we do not need the cost information
    # coordinates of the starting node is startingState, list to keep actions taken is empty
    actionsTaken = []
    startingNode = (startingState, actionsTaken)
    # keep nodes visited in a variable
    visitedNodes = []
    # push starting node to the stack
    stack.push(startingNode)
    
    while not stack.isEmpty():
        # pop the fist node in the stack
        currentNode = stack.pop()
        # update actions taken by the agent
        # keep actions taken in a variable
        actionsTaken = currentNode[1]
        # visited nodes kept. To be able to check whether a node is visited, states need to be compared
        visitedStates = list(map(lambda x: x[0], visitedNodes))
        if currentNode[0] not in visitedStates:
            # mark it as visited by adding to the list of visited nodes
            visitedNodes.append(currentNode)
            # check if currentNode is the goal node
            if not problem.isGoalState(currentNode[0]):
                # get the adjacent nodes of the current node
                adjacentNodes = problem.getSuccessors(currentNode[0])
                # push adjacents onto the stack
                for adjacentNode in adjacentNodes:
                    if adjacentNode[0] not in visitedStates:
                        adjacentNode = (adjacentNode[0], actionsTaken + [adjacentNode[1]])
                        stack.push(adjacentNode)
            else:
                return actionsTaken
                
    
    return ["Search could not be completed"]
    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Same steps in DFS -except the data structure being used- are taken for BFS, as well
    queue = util.Queue()
    startingState = problem.getStartState()
    actionsTaken = []
    startingNode = (startingState, actionsTaken)
    visitedNodes = []
    queue.push(startingNode)
    
    while not queue.isEmpty():
        currentNode = queue.pop()
        actionsTaken = currentNode[1]
        visitedStates = list(map(lambda x: x[0], visitedNodes))
        if currentNode[0] not in visitedStates:
            visitedNodes.append(currentNode)
            if not problem.isGoalState(currentNode[0]):
                adjacentNodes = problem.getSuccessors(currentNode[0])
                for adjacentNode in adjacentNodes:
                    if adjacentNode[0] not in visitedStates:
                        adjacentNode = (adjacentNode[0], actionsTaken + [adjacentNode[1]])
                        queue.push(adjacentNode)
            else:
                return actionsTaken
    
    return ["Search could not be completed"]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # For UCS, priority queue is used. Costs of actions will be assigned as priorities to nodes
    priorityQueue =  util.PriorityQueue()
    startingState = problem.getStartState()
    actionsTaken = []
    totalCost = 0
    # For this search problem state information (for start 'A'), action list and costs will be kept in a tuple
    startingNode = (startingState, actionsTaken, totalCost)
    visitedNodes = []
    priorityQueue.push(startingNode, startingNode[-1])
    
    while not priorityQueue.isEmpty():
        currentNode = priorityQueue.pop()
        actionsTaken = currentNode[1]
        totalCost = currentNode[-1]
        visitedStates = list(map(lambda x: x[0], visitedNodes))
        if currentNode[0] not in visitedStates:
            visitedNodes.append(currentNode)
            if not problem.isGoalState(currentNode[0]):
                adjacentNodes = problem.getSuccessors(currentNode[0])
                for adjacentNode in adjacentNodes:
                    if adjacentNode[0] not in visitedStates:
                        adjacentNode = (adjacentNode[0], actionsTaken + [adjacentNode[1]], totalCost + adjacentNode[-1])
                        priorityQueue.push(adjacentNode, adjacentNode[-1])
            else:
                 return actionsTaken
             
    return ["Search could not be completed"]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # frontier is again priority queue, but state priorities are now total cost + heuristic of a state
    priorityQueue =  util.PriorityQueue()
    startingState = problem.getStartState()
    actionsTaken = []
    totalCost = 0
    currentHeuristic = heuristic(problem.getStartState(), problem)
    priority = totalCost + currentHeuristic
    # For this search problem state information (for start 'A'), action list and costs will be kept in a tuple
    startingNode = (startingState, actionsTaken, totalCost, currentHeuristic, priority)
    visitedNodes = []
    priorityQueue.push(startingNode, startingNode[-1])
    
    while not priorityQueue.isEmpty():
        currentNode = priorityQueue.pop()
        actionsTaken = currentNode[1]
        totalCost = currentNode[2]
        currentHeuristic = heuristic(currentNode[0], problem)
        priority = totalCost + currentHeuristic
        visitedStates = list(map(lambda x: x[0], visitedNodes))
        if currentNode[0] not in visitedStates:
            visitedNodes.append(currentNode)
            if not problem.isGoalState(currentNode[0]):
                adjacentNodes = problem.getSuccessors(currentNode[0])
                for adjacentNode in adjacentNodes:
                    if adjacentNode[0] not in visitedStates:
                        adjacentNode = (adjacentNode[0],
                                        actionsTaken + [adjacentNode[1]],
                                        totalCost + adjacentNode[2],
                                        heuristic(adjacentNode[0], problem),
                                        totalCost + adjacentNode[2] + heuristic(adjacentNode[0], problem))
                        priorityQueue.push(adjacentNode, adjacentNode[-1])
            else:
                 return actionsTaken
             
    return ["Search could not be completed"]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
