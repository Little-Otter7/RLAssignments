from Assignment2 import Gridworld, uniform_random_policy, first_visit_mc_prediction, every_visit_mc_prediction, td_0_prediction

def test_fvmc():
    gridworld = Gridworld()
    V = first_visit_mc_prediction(gridworld,uniform_random_policy,num_episodes = 1000)
    for state,value in V.items():
        print(f'State:{state},Value:{value}')

def test_evmc():
    gridworld = Gridworld()
    V = every_visit_mc_prediction(gridworld,uniform_random_policy,num_episodes = 1000)
    for state,value in V.items():
        print(f'State:{state},Value:{value}') 

def test_td0():
    gridworld = Gridworld()
    V = td_0_prediction(gridworld,uniform_random_policy,num_episodes = 1000)
    for state,value in V.items():
        print(f'State:{state},Value:{value}')   







if __name__ =='__main__':
    test_fvmc()
    test_evmc()
    test_td0()