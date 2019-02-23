"""
MRP class for realizing state action model
"""
import numpy as np

class MRP():
    def __init__(self,n):
        #n=number of state
        self.n=n

    def startState(self):
        #states go from 0,1,....,n-1->Terminal ::: total self.n states
        return 1

    def isEnd(self, state):
        return state==self.n

    def next_state(self,state):
        #transition from state to next state: no action
        return state+1

    def get_successor_state_reward(self, state):
        if self.isEnd(state): return (state,0)
        next_state=self.next_state(state)
        if state%2==0:#even state
            return (next_state,np.random.randn())
        else:#odd state
            return (next_state,0.1)


def td_learning(traj_list, num_states, td_update_fn, lamda, gamma):
    Vs=np.zeros(num_states)
    Es=np.zeros(num_states)
    for episode in traj_list:
        Vs, Es, lamda_s, gamma_s=td_update_fn(episode,Vs, Es, lamda_s, gamma_s)
    return Vs, Es, lamda_s, gamma_s


num_states=10
problem=MRP(num_states)
s=problem.startState()
history=[]
while s!=num_states:
    state_reward=problem.get_successor_state_reward(s)
    s=state_reward[0]
    history.append(state_reward)

print("------------history------------------")
print(history)

