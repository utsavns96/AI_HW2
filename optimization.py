from asyncio import constants
import constraint
import math
import random
from simanneal import Annealer
import cvxpy as cp
import numpy as np

############################## PROBLEM 1 ######################################
# In problem 1, you are going to implement CSP for Sudoku problem. Implement cstAdd,
# which adds the constraints.  It takes a problem object (problem), a matrix of variable
# names (grid), a list of legal values (domains), and the side length of the inner squares
# (psize, which is 3 in an ordinary sudoku and 2 in the smaller version we provide as
# the easier test case).

""" A helper function to visualize ouput.  You do not need to change this """
""" output: the output of your solver """
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
def sudokuCSPToGrid(output,psize):
    if output is None:
        return None
    dim = psize**2
    return np.reshape([[output[str(dim*i+j+1)] for j in range(dim)] for i in range(dim)],(dim,dim))

""" helper function to add variables to the CSP """
""" you do not need to change this"""
""" Note how we initialize the domains to the supplied values on the marked line """
def addVar(problem, grid, domains, init):
    numRow = grid.shape[0]
    numCol = grid.shape[1]
    for rowIdx in range(numRow):
        for colIdx in range(numCol):
            if grid[rowIdx, colIdx] in init: #use supplied value
                problem.addVariable(grid[rowIdx,colIdx], [init[grid[rowIdx, colIdx]]])
            else:
                problem.addVariable(grid[rowIdx,colIdx], domains)

                    
""" here you want to add all of the constraints needed.
    problem: the CSP problem instance we have created for you
    grid: a psize ** 2 by psize ** 2 array containing the CSP variables
    domains: the domain for the variables representing non-pre-filled squares
    psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku)
    # Hint: Use loops!
    #       Remember problem.addConstraint() to add constraints
    #       Example syntax for adding a constraint that two variable are not equal:
    #       problem.addConstraint(lambda a, b: a !=b, (variable1,variable2)
    #       See the example file for more"""
def cstAdd(problem, grid, domains,psize):
    # --------------------
    # 1. Alldiff for rows
    # 2. Alldiff for columns
    # 3. All diff for psize*psize box
    v=0
    for i in range(psize*psize):
        row_index=[grid[v,col] for  col in range(psize*psize)] # Alldiff for rows
        problem.addConstraint(constraint.AllDifferentConstraint(), row_index)
        col_index=[grid[row,v] for row in range(psize*psize)] # Alldiff for columns
        problem.addConstraint(constraint.AllDifferentConstraint(), col_index)
        v+=1
    
    for r in range(0,psize**2,psize): #Alldiff for small boxes
        for c in range(0,psize**2,psize):
            small_block = []
            for r1 in range(r,r+psize):
                for c1 in range(c,c+psize):
                    small_block.append(grid[r1,c1])
            problem.addConstraint(constraint.AllDifferentConstraint(),small_block)

            

    # --------------------

""" Implementation for a CSP Sudoku Solver """
""" positions: list of (row,column,value) triples representing the already filled in cells"""
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
""" You do not need to change this """
def sudokuCSP(positions,psize):
    sudokuPro = constraint.Problem()
    dim = psize ** 2
    numCol = dim
    numRow = dim
    domains = list(range(1,dim+1))
    init = {str(dim*p[0]+p[1]+1):p[2] for p in positions}
    sudokuList = [str(i) for i in range(1,dim**2+1)]
    sudoKuGrid = np.reshape(sudokuList, [numRow, numCol])
    addVar(sudokuPro, sudoKuGrid, domains, init)
    cstAdd(sudokuPro, sudoKuGrid, domains,psize)
    return sudokuPro.getSolution()

############################## PROBLEM 2 ######################################
# In the fractional knapsack problem you have a knapsack with a fixed weight capacity
# and want to fill it with valuable items so that we maximize the total value in it
# while ensuring the weight does notexceed the capacity. Fractions of items are allowed
#

""" Frational Knapsack Problem
    c: the capacity of the knapsack
    Hint: Think carefully about the range of values your variables can be, and include them in the constraints"""
def fractionalKnapsack(c):
    # -------------------
    # Your code
    # First define some variables
    weights = np.array([5,3,1])
    values = np.array([2,3,1])
    x = cp.Variable(3)
    # Put your constraints here
    # 1. We need to keep the weights below c
    # 2. We can only use one item each

    constraints = [cp.sum(weights @ x) <= c,0 <= x, x <= 1]
    # Fix this to be the correct objective function
    obj = cp.Maximize(values @ x)

    # End of your code
    # ------------------
    prob = cp.Problem(obj, constraints)
    return prob.solve()

############################## PROBLEM 3 ######################################
# Integer Programming: Sudoku
# We have provided most of an IP implementation.
# Again, you just need to implement the constraints.  Note however, unlike in the CSP version,
# we have not already “prefilled” the squares for you.  You’ll need to add those constraints yourself.

""" A helper function to visualize ouput.  You do not need to change this """
""" binary: the output of your solver """
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
def sudokuIPToGrid(binary,psize):
    if binary is None:
        return None
    dim = psize**2
    x = np.zeros((dim,dim),dtype=int)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if binary[dim*i+j][k] >= 0.99:
                    x[i][j] = k+1
    return x

""" Implementation for a IP Sudoku Solver """
""" positions: list of (row,column,value) triples representing the already filled in cells"""
""" psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
""" the library does not support 3D variables, so M[i][j] should be your indicator variable """
""" for the ith square having value j where in a 4x4 grid i ranges from 0 to 15 """
def sudokuIP(positions,psize):
    # Define the variables - see comment above about interpretation
    print(positions)
    dim = psize**2
    M = cp.Variable((dim**2,dim),integer=True) #Sadly we cannot do 3D Variables
    constraints = []
    # --------------------
    # Your code
    # It should define the constraints needed
    # We've given you one to get you started
    constraints.extend([0 <= M[x][k] for x in range(dim**2) for k in range (dim)])
    constraints.extend([M[x][k] <= 1 for x in range(dim**2) for k in range (dim)])
    constraints.extend([M[(dim*r)+c][v-1] == 1 for r,c,v in positions])
    for x in range(dim**2):
        t = []
        for k in range(dim):
            t.append(M[x][k])
        constraints.append(cp.sum(t) == 1)
    for k in range(dim):
        for r in range(0,dim**2,dim):
            t = [] 
            for r1 in range(r,r+dim):
                t.append(M[r1][k])
            constraints.append(cp.sum(t) == 1)
    
    for k in range(dim):
        for c in range(0,dim):
            t2 = []
            for c1 in range(c,dim**2,dim):
                t2.append(M[c1][k])
            constraints.append(cp.sum(t2) == 1)   

    for k in range(dim):
        for r in range(0,dim,psize): #Alldiff for small boxes
                for c in range(0,dim,psize):
                    small_block = []
                    for r1 in range(r,r+psize):
                        for c1 in range(c,c+psize):
                            small_block.append(M[(dim*r1)+c1][k])
                    constraints.append(cp.sum(small_block) == 1)
            





    # End your code
    # -------------------

    # Form dummy objective - we only care about feasibility
    obj = cp.Minimize(M[0][0])

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()
    #Uncomment the version below instead if you want more detailed information from the solver to see what might be going wrong
    #Please leave commented out when submitting.
    # prob.solve(verbose=True)
    #For debugging you may want to look at some of the information contained in prob before returning
    #See the example file
    return M.value
    # --------------------

############################## PROBLEM 4 ######################################
# Local Search: TSP
# We have provided most of a simulated annealing implementation of the famous traveling salesman problem,
# where you seek to visit a list of cities while minimizing the total distance traveled.
# You need to implement move and energy.
# The former is the operation for finding nearby candidate solutions while the latter
# evaluates how good the current candidate solution is.
# Move should generate a random local move without regard for whether it is beneficial.
# Similarly, to receive credit energyshould calculate the total euclidean distance of the current candidate tour.
# There is a distance function you may wish to implement to help with this.

class TravellingSalesmanProblem(Annealer):

    """problem specific data"""
    # latitude and longitude for the twenty largest U.S. cities
    cities = {
        'New York City': (40.72, 74.00),
        'Los Angeles': (34.05, 118.25),
        'Chicago': (41.88, 87.63),
        'Houston': (29.77, 95.38),
        'Phoenix': (33.45, 112.07),
        'Philadelphia': (39.95, 75.17),
        'San Antonio': (29.53, 98.47),
        'Dallas': (32.78, 96.80),
        'San Diego': (32.78, 117.15),
        'San Jose': (37.30, 121.87),
        'Detroit': (42.33, 83.05),
        'San Francisco': (37.78, 122.42),
        'Jacksonville': (30.32, 81.70),
        'Indianapolis': (39.78, 86.15),
        'Austin': (30.27, 97.77),
        'Columbus': (39.98, 82.98),
        'Fort Worth': (32.75, 97.33),
        'Charlotte': (35.23, 80.85),
        'Memphis': (35.12, 89.97),
        'Baltimore': (39.28, 76.62)
    }

    """problem-specific helper function"""
    """you may wish to implement this """
    def distance(self, a, b):
        """Calculates distance between two latitude-longitude coordinates."""
        # -----------------------------
        # Your code
        return math.sqrt(((a[0]-b[0])**2)+((a[1]-b[1])**2))
        # -----------------------------



    """ make a local change to the solution"""
    """ a natural choice is to swap to cities at random"""
    """ current state is available as self.state """
    """ Note: This is just making the move (change) in the state,
              Worry about whether this is a good idea elsewhere. """
    """ Make sure there is a way for enough of your local changes to
              reach a solution """
    def move(self):

        # --------------------
        # Your code
        random.shuffle(self.state)
        # -------------------------


    """ how good is this state? """
    """ lower is better """
    """ current state is available as self.state """
    """ Use self.cities to find a city's coordinates"""
    def energy(self):
        # Initialize the value to be returned
        e = 0
        
        #-----------------------
        # Your code
        for i in range(0,len(self.state)-1): # For all cities except 1st and last
            e += self.distance(self.cities[self.state[i]],
                          self.cities[self.state[i+1]])
        
        e+= self.distance(self.cities[self.state[-1]],self.cities[self.state[0]]) # Accounting for distance between first and last city.

        #-----------------------

        return e

# Execution part, please don't change it!!!
def annealTSP(initial_state):
        # initial_state is a list of starting cities
        tsp = TravellingSalesmanProblem(initial_state)
        return tsp.anneal()

############################## PROBLEM 5 ######################################
# Local Search: Sudoku
# Now we have the skeleton of a simulated annealing implemen-tation of Sudoku.
# You need to design the move and energy functions and will receive credit based on
# how many of 10 runs succeed in finding a correct answer:  to achieve k points 2k−1 runs need to pass

class SudokuProblem(Annealer):

    """ positions: list of (row,column,value) triples representing the already filled in cells"""
    """ psize: the problem size (e.g. 3 for a 3x3 grid of 3x3 squares in a standard Sudoku) """
    def __init__(self,initial_state,positions,psize):
        self.psize = psize
        self.positions = positions
        super(SudokuProblem, self).__init__(initial_state)

    """ make a local change to the solution"""
    """ current state is available as self.state """
    """ Hint: Remember this is sudoku, just make one local change
              print self.state may help to get started"""
    """ Note that the initial state we give you is purely random
              and may not even respect the filled in squares. """
    """ Make sure there is a way for enough of your local changes to
              reach a solution """
    def move(self):

        # --------------------
        # Your code
        pass
        # random.shuffle(self.state)
        # -------------------------


    """ how good is this state? """
    """ lower is better """
    """ current state is available as self.state """
    """ Remember what we talked about in class for the energy function for a CSP """
    def energy(self):
        # Initialize the value to be returned
        e = 0
        
        #-----------------------
        # Your code


        #-----------------------

        return e

# Execution part, please don't change it!!!
def annealSudoku(positions, psize):
        # initial_state of starting values:
        initial_state = [random.randint(1,psize**2) for i in range(psize ** 4)]
        sudoku = SudokuProblem(initial_state,positions,psize)
        sudoku.steps = 100000
        sudoku.Tmax = 100.0
        sudoku.Tmin = 1.0
        return sudoku.anneal()
