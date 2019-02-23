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
            return (state,np.random.randn())
        else:#odd state
            return (state,0.1)

#function for td lambda learning
def td_learning(traj_list, num_states, td_update_fn, lambd, gamma, alpha):
    #initializations
    Vs      =np.zeros(num_states)
    gamma_s =np.ones(num_states)
    lambd_s =np.ones(num_states)

    #update V, E, lambd, gamma for all states from the td_update_fn
    for episode in traj_list:
        Vs, lambd_s, gamma_s=td_update_fn(episode,Vs, lambd_s, gamma_s, alpha)
    return Vs, lambd_s, gamma_s


def generate_trajectories(problem, n_states, n_trajectories):
    traj_list=[]
    for n in range(n_trajectories):
        s= problem.startState()
        h=[]
        while s!=n_states:
            s_r=problem.get_successor_state_reward(s)
            #next state
            s=problem.next_state(s)
            h.append(s_r)
        traj_list.append(h)
    return traj_list

n_traj= 50
num_states=10
problem=MRP(num_states)
traj_list=generate_trajectories(problem, num_states, n_traj)
print("------------trajectories list------------")
for item in traj_list:
    print(item)
s=problem.startState()
history=[]
while s!=num_states:
    state_reward=problem.get_successor_state_reward(s)
    s=problem.next_state(s)
    history.append(state_reward)

print("------------history------------------")
print(history)

