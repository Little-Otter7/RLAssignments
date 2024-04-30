import numpy as np
import matplotlib.pyplot as plt
# Definition of CliffwalkingEnv
class CliffWalkingEnv:
    def __init__(self, width=12, height=4): #Create a 12*4 grid
        self.width = width
        self.height = height
        self.start = (self.height - 1, 0)
        self.goal = (self.height - 1, self.width - 1)
        self.state = self.start
        self.end_states = [self.goal]
        self.cliff = [(self.height - 1, i) for i in range(1, self.width - 1)]

    def reset(self): #Definition of the reset function
        self.state = self.start
        return self.state

    def step(self, action): # Action
        if self.state in self.end_states:
            return self.state, 0, True, {}
        
        next_state = list(self.state)
        
        if action == 'U':
            next_state[0] = max(0, next_state[0] - 1)
        elif action == 'D':
            next_state[0] = min(self.height - 1, next_state[0] + 1)
        elif action == 'L':
            next_state[1] = max(0, next_state[1] - 1)
        elif action == 'R':
            next_state[1] = min(self.width - 1, next_state[1] + 1)
        
        next_state = tuple(next_state)
        
        if next_state in self.cliff:
            reward = -100
            next_state = self.start
        else:
            reward = -1

        self.state = next_state
        done = self.state in self.end_states
        return next_state, reward, done, {}

    def render(self):
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.state:
                    print('S', end=' ')
                elif (i, j) in self.cliff:
                    print('C', end=' ')
                elif (i, j) == self.goal:
                    print('G', end=' ')
                else:
                    print('.', end=' ')
            print()

# Definition of epsilon_greedy_policy
def epsilon_greedy_policy(Q, state, epsilon=0.05):
    if np.random.rand() < epsilon:
        return np.random.choice(['U', 'D', 'L', 'R'])  
    else:
        
        actions = np.array(['U', 'D', 'L', 'R'])
        q_values = np.array([Q.get((state, a), 0) for a in actions])
        return actions[np.argmax(q_values)]

# Definition of sarsa method
def sarsa(env, episodes, alpha=0.5, gamma=1.0, epsilon=0.05, Q=None):
    if Q is None:
        Q = {}  

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)

        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            Q[(state, action)] = Q.get((state, action), 0) + alpha * (
                reward + gamma * Q.get((next_state, next_action), 0) - Q.get((state, action), 0)
            )

            state, action = next_state, next_action

    return Q

# Definition of Q_Learning
def q_learning(env, episodes, alpha=0.5, gamma=1.0, epsilon=0.0, Q=None):
    if Q is None:
        Q = {}  

    for episode in range(episodes):
        state = env.reset()

        done = False
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            next_max = max([Q.get((next_state, a), 0) for a in ['U', 'D', 'L', 'R']])

            Q[(state, action)] = Q.get((state, action), 0) + alpha * (
                reward + gamma * next_max - Q.get((state, action), 0)
            )

            state = next_state

    return Q



def run_experiment(env, method_function, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.05):
    Q = {}  
    total_rewards = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if method_function == sarsa:
                next_action = epsilon_greedy_policy(Q, next_state, epsilon)
                Q[(state, action)] = Q.get((state, action), 0) + alpha * (
                    reward + gamma * Q.get((next_state, next_action), 0) - Q.get((state, action), 0)
                )
                state, action = next_state, next_action
            elif method_function == q_learning:
                next_max = max([Q.get((next_state, a), 0) for a in ['U', 'D', 'L', 'R']])
                Q[(state, action)] = Q.get((state, action), 0) + alpha * (
                    reward + gamma * next_max - Q.get((state, action), 0)
                )
                state = next_state
        total_rewards[i] = total_reward
    return Q, total_rewards  

def print_optimal_path(Q, env):
    state = env.reset()
    optimal_path = [state]
    done = False
    while not done:
        action = epsilon_greedy_policy(Q, state, 0)  # 使用epsilon=0来选择最优动作
        next_state, _, done, _ = env.step(action)
        if next_state == state: 
            print("No optimal path found.")
            return
        state = next_state
        optimal_path.append(state)
        if state == env.goal:
            done = True

    
    for i in range(env.height):
        for j in range(env.width):
            if (i, j) == env.start:
                print('S', end=' ')
            elif (i, j) == env.goal:
                print('G', end=' ')
            elif (i, j) in env.cliff:
                print('C', end=' ')
            elif (i, j) in optimal_path:
                print('*', end=' ') 
            else:
                print('.', end=' ')
        print()


env = CliffWalkingEnv()
episodes = 500

Q_sarsa, sarsa_rewards = run_experiment(env, sarsa, episodes)
Q_q_learning, q_learning_rewards = run_experiment(env, q_learning, episodes)

sarsa_avg_reward = np.mean(sarsa_rewards)
q_learning_avg_reward = np.mean(q_learning_rewards)

print(f"Sarsa平均奖励: {sarsa_avg_reward}")
print(f"Q-Learning平均奖励: {q_learning_avg_reward}")
print("\nSarsa Optimal Path:")
print_optimal_path(Q_sarsa, env)
print("\nQ-Learning Optimal Path:")
print_optimal_path(Q_q_learning, env)
plt.plot(np.arange(episodes), sarsa_rewards, label='Sarsa')
plt.plot(np.arange(episodes), q_learning_rewards, label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Sarsa vs Q-Learning in CliffWalkingEnv')
plt.legend()
plt.show()


