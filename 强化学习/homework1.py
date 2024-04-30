# Step 2: Define the uniform random policy
def uniform_Random_policy(gridworld):
    policy = {}
    for state in gridworld.allstates:
        policy[state] = {'north':0.25,'south':0.25,'west':0.25,'east':0.25}
    return policy
    
# Step 3: Policy Evaluation
def policy_evaluation(policy,gridworld,discount_factor=0.99,theta=0.01):
    # Initialization state value
    V ={state: 0 for state in gridworld.allstates}
    while True:
        delta = 0
        for state in gridworld.allstates:
            v = 0
            for action,action_prob in policy[state].items():
                nextState,reward = gridworld.move(state,action)
                v+=action_prob*(reward+discount_factor*V[nextState])
            delta = max(delta,abs(v-V[state]))
            V[state]=v
        if delta < theta:
            break
    return V

def policy_improvement(V,gridworld,discount_factor=0.99):
    policy_stable = True
    new_policy = {}
    for state in gridworld.allstates:
        actions_values = {}
        for action in {'north','south','west','east'}:
            nextState,reward = gridworld.move(state,action)
            actions_values[action] = reward +discount_factor*V[nextState]
        best_action = max(actions_values,key=actions_values.get)
        new_policy[state]={action:(1 if action==best_action else 0)
                            for action in ['north','south','west','east']}
    return new_policy,policy_stable

def policy_iteration(gridworld):
    policy = uniform_Random_policy(gridworld)
    while True:
        V = policy_evaluation(policy,gridworld)
        new_policy,policy_stable = policy_improvement(V,gridworld)
        if policy_stable:
            return new_policy, V
        policy = new_policy

# Step 4 : Value Iteration
def value_iteration(gridworld,discount_factor = 0.99, theta = 0.01):
    V={state: 0 for state in gridworld.allstates}
    while True:
        delta = 0
        for state in gridworld.allstates:
            if gridworld.isTerminal(state):
                continue
            v = V[state]
            V[state] = max(gridworld.move(state,action)[1]+discount_factor*V[gridworld.move(state,action)[0]] for action in ['north','south','west','east'])
            delta = max(delta,abs(v-V[state]))
        if delta < theta:
            break
    policy = {}
    for state in gridworld.allstates:
        action_values = {}
        for action in ['north','south','west','east']:
            next_state,reward = gridworld.move(state,action)
            action_values[action] = reward + discount_factor * V[next_state]
        best_action = max(action_values,key=action_values.get)
        policy[state] = {action:(1 if action == best_action else 0) for action in ['north','south','west','east']}
    return policy,V

# Step 1: Bulid the environment of Gridworld
class Gridworld:
    def __init__(self,width,height,start,ends,obstacles): #initialization
        self.width = width
        self.height = height
        self.start = start
        self.ends = ends
        self.obstacles = obstacles
        self.allstates = [(i,j) for i in range(width) for j in range(height) if (i,j) not in obstacles]
    
    def reset(self): #reset function
        self.state = self.start
        return self.state
    
    def isTerminal (self,state): # Terminal Jugement function
        return state in self.ends
    
    def move(self,state,action): #Movement function
        nextState = state
        if self.isTerminal(state):
            return state,0
        if action =='north':
            nextState = (state[0],max(state[1]-1,0))
        elif action =='south':
            nextState = (state[0],min(state[1]+1,self.height-1))
        elif action == 'west':
            nextState = (max(state[0]-1,0),state[1])
        elif action == 'east':
            nextState = (min(state[0]+1,self.width-1),state[1])
        
        if nextState in self.obstacles:
            nextState = state
        return nextState, -1 if nextState not in self.ends else 0
