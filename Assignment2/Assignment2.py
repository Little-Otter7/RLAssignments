import random
from collections import defaultdict
import numpy as np
# Step 2: Define the uniform random policy
def uniform_random_policy(gridworld):
    return random.choice(gridworld.actions)
    
# Step 3: Definition of the generate_episode
def generate_episode(gridworld,policy):
    episode = []
    current_state=random.choice([s for s in gridworld.states if not gridworld.isTerminal(s)])
    while not gridworld.isTerminal(current_state):
        action=policy(gridworld)
        next_state , reward = gridworld.step(current_state,action)
        episode.append((current_state,action,reward))
        current_state = next_state
    return episode

# Step 4: Definition of first visit MC prediction
def first_visit_mc_prediction(gridworld,policy,num_episodes,discount_factor=1.0):
    V = defaultdict(float)
    returns = defaultdict(list)
    for _ in range(num_episodes):
        episode = generate_episode(gridworld,policy)
        visited_states = set()
        for i, (state,_,_) in enumerate(episode):
            if state not in visited_states:
                visited_states.add(state)
                G=sum([x[2]*(discount_factor**j) for j,x in enumerate(episode[i:])])
                returns[state].append(G)
                V[state]=np.mean(returns[state])
    return V

# Step 5: Build the TD(0) prediction model
def td_0_prediction(gridworld,policy,num_episodes,alpha=0.01,discount_factor=1):
    V = defaultdict(float)
    for _ in range(num_episodes):
        current_state = random.choice([s for s in gridworld.states if not gridworld.isTerminal(s)])
        while not gridworld.isTerminal(current_state):
            action = policy(gridworld)
            next_state,reward = gridworld.step(current_state,action)
            td_target = reward + discount_factor*V[next_state]
            td_delta = td_target - V[current_state]
            V[current_state] += alpha*td_delta
            current_state=next_state
    return V
# Step 6: Build the every_visit_mc prediction Model
def every_visit_mc_prediction(gridworld,policy,num_episodes,discount_factor=1.0):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = generate_episode(gridworld,policy)
        for i , (state,_,_) in enumerate(episode):
            G = sum(x[2]*discount_factor**j for j,x in enumerate(episode[i:]))
            returns[state].append(G)
            V[state] = np.mean(returns[state])
    return V

# Step 1: Build the environment of Gridworld with the code 'Assignment1'
class Gridworld:
    def __init__(self,width=6,height=6): #initialization
        self.width = width
        self.height = height
        self.states = [(i,j) for i in range(width) for j in range(height)]
        self.terminal_states = [(0,0),(width-1,height-1)]
        self.actions = ['north','south','west','east']
        self.state_rewards={(i,j):-1 for i in range(width) for j in range(height)}
    
    def isTerminal (self,state): # Terminal Jugement function
        return state in self.terminal_states
    
    def step(self,state,action): #Movement function
        if self.isTerminal(state):
            return state,0
        next_state =list(state)
        if action =='north' and state[1]>0:
            next_state[1]-=1
        elif action =='south' and state[1]< self.height-1:
            next_state[1]+=1
        elif action == 'west' and state[0]>0:
            next_state[0]-=1
        elif action == 'east' and state[0] < self.width-1:
            next_state[0]+=1
        
        return tuple(next_state) , self.state_rewards.get(tuple(next_state),0)