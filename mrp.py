"""
MRP class for realizing state action model
"""
import numpy as np
import matplotlib.pyplot as plt

class MRP():
	def __init__(self,n):
		#n=number of state
		self.n=n
	
	def startState(self):
		#states go from 0,1,....,n-1->Terminal ::: total self.n states
		return 0
	
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
		
	def get_successor_state_reward_negative(self, state, prev_reward=0):
		if self.isEnd(state): return (state,0)
		#if state%2==0:#even state
		#	return (state,np.random.randn())
		#else:#odd state
		#	return (state, -prev_reward)
		if state == 0:
			return (state,np.random.randn())
		else:
			return (state, -prev_reward)
		
	def meta_online_TD_lambda_update(self, h, Vs, lambd, gamma, alpha, beta):
		
		rewards = [item[1] for item in h]
		T = len(h)
		Vnew = Vs.copy()
		Es = np.zeros(self.n)
		Enew = np.zeros(self.n)
		for (t, (s, r)) in enumerate(h):
			# ordinary TD update
			Enew = Es*lambd*gamma
			Enew[s] += 1.
			TD_error = r + gamma[s] * Vs[next_state(self, s)] - Vs[s]
			Vnew += alpha * TD_error * Enew
			# meta parameter update
			lambdnew = lambd + beta # not done
			Es = Enew.copy()
			
	def meta_offline_TD_lambda_update(self, h, hh, Vs, lambd, gamma, alpha, beta):
		states = [item[0] for item in h]
		rewards = [item[1] for item in h]
		T = len(h)
		Vnew = Vs.copy()
		for (t, (s, r)) in enumerate(h):
			# calculate lambda return
			Gt = []
			Gt_lambd = 0
			gammaR = [rewards[_t]*gamma[s]**(_t - t) for _t in np.arange(t,T)]
			for _t in np.arange(t+1, T-1):
				_Gt = sum(gammaR[:_t - t]) + Vs[states[_t]]*gamma[s]**(_t-t)
				Gt.append(_Gt)
				Gt_lambd += (1 - lambd[s])*lambd[s]**(_t-t-1)*_Gt
			_Gt = sum(gammaR)
			Gt.append(_Gt)
			Gt_lambd += _Gt*lambd[s]**(T-t-1)
			# update V
			Vnew[s] += alpha*(Gt_lambd - Vs[s])
		Vs = Vnew.copy()
		
		# update lambda and gamma (double sampling)
		states = [item[0] for item in hh]
		rewards = [item[1] for item in hh]
		lambdnew = lambd.copy()
		for (t, (s, r)) in enumerate(h):
			Gt = []
			gammaR = [rewards[_t]*gamma[s]**(_t - t) for _t in np.arange(t,T)]
			for _t in np.arange(t+1, T-1):
				Gt.append(sum(gammaR[:_t - t]) + Vs[states[_t]]*gamma[s]**(_t-t))
			Gt.append(sum(gammaR))	
			
			d = -Gt[0]
			for _t in np.arange(1, T-t-1):
				d += lambd[s]**(_t - 1) * (_t*(1 - lambd[s])-lambd[s])*Gt[_t]
			d += (T-t-1)*lambd[s]**(T-t-2)*Gt[-1]
			lambdnew[s] += beta*d
		
		return Vs, lambdnew, gamma
			
		
			

#function for td lambda learning
def td_learning(traj_list, traj_list_double, num_states, td_update_fn, alpha=0.1, beta=0.003):
	#initializations
	n_traj = len(traj_list)
	Vs      =np.zeros(num_states)
	gamma_s =np.ones((n_traj+1, num_states))
	lambd_s =np.ones((n_traj+1, num_states))*0.5

    #update V, E, lambd, gamma for all states from the td_update_fn
	for i_traj in range(n_traj):
		Vs, lambd_s[i_traj+1,:], gamma_s[i_traj+1,:]=td_update_fn(traj_list[i_traj], traj_list_double[i_traj], Vs, lambd_s[i_traj,:], gamma_s[i_traj,:], alpha, beta)
	return Vs, lambd_s, gamma_s


def generate_trajectories(problem, n_states, n_trajectories):
	traj_list=[]
	for n in range(n_trajectories):
		s= problem.startState()
		h=[]
		s_r=(0,0)
		while s!=n_states:
			s_r=problem.get_successor_state_reward_negative(s, s_r[1])
			#next state
			s=problem.next_state(s)
			h.append(s_r)
		traj_list.append(h)
	return traj_list


if __name__ == '__main__':
	n_traj= 5000
	num_states=9
	problem=MRP(num_states)
	traj_list=generate_trajectories(problem, num_states, n_traj)
	traj_list_double=generate_trajectories(problem, num_states, n_traj)
	Vs, lambd_s, gamma_s=td_learning(traj_list, traj_list_double, num_states, problem.meta_offline_TD_lambda_update)
	
	plt.figure()
	for s in range(num_states):
		plt.plot(lambd_s[:,s], label='lambda_'+str(s))
	plt.xlabel('number of trajectories')
	plt.ylabel('lambda')
	#print("------------trajectories list------------")
	#for item in traj_list:
	#    print(item)
	#s=problem.startState()
	#history=[]
	#while s!=num_states:
	#    state_reward=problem.get_successor_state_reward(s)
	#    s=problem.next_state(s)
	#    history.append(state_reward)
	
	#print("------------history------------------")
	#print(history)
