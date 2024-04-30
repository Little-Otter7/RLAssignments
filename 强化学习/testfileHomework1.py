from homework1 import Gridworld,policy_iteration,value_iteration

def test():
    gridworld = Gridworld(width=6, height=6, start=(0, 0), ends=[(5, 5)], obstacles=[(1, 1), (2, 2)])

    pi_policy, _ = policy_iteration(gridworld)
    print("策略迭代得到的策略:")
    for state, actions in pi_policy.items():
        print(state, actions)

    # 使用价值迭代
    vi_policy, _ = value_iteration(gridworld)
    print("\n价值迭代得到的策略:")
    for state, actions in vi_policy.items():
        print(state, actions)

if __name__=='__main__':ff
    test()