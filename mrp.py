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
	
	def meta_offline_TD_update(self, h, Vs, lambd, gamma, alpha):
		states = [item[0] for item in h]
		rewards = [item[1] for item in h]
		T = len(h)
		Vnew = Vs.copy()
		for (t, (s, r)) in enumerate(h):
			# calculate lambda return
			Gt = []
			Gt_lambd = 0
			gammaR = [rewards[_t]*gamma[s]**(_t - t) for _t in np.arange(t,T)]
			for _t in np.arange(t+1, T):
				_Gt = sum(gammaR[:_t - t]) + Vs[states[_t]]*gamma[s]**(_t-t)
				Gt.append(_Gt)
				Gt_lambd += (1 - lambd[s])*lambd[s]**(_t-t-1)*_Gt
			_Gt = sum(gammaR)
			Gt.append(_Gt)
			Gt_lambd += _Gt*lambd[s]**(T-t-1)
			# update V
			Vnew[s] += alpha*(Gt_lambd - Vs[s])
		return Vnew
	
	def meta_offline_lambda_update(self, h, Vs, lambd, gamma, beta):
		states = [item[0] for item in h]
		rewards = [item[1] for item in h]
		T = len(h)
		lambdnew = lambd.copy()
		for (t, (s, r)) in enumerate(h):
			Gt = []
			Gt_lambd = 0
			gammaR = [rewards[_t]*gamma[s]**(_t - t) for _t in np.arange(t,T)]
			for _t in np.arange(t+1, T):
				_Gt = sum(gammaR[:_t - t]) + Vs[states[_t]]*gamma[s]**(_t-t)
				Gt.append(_Gt)
				Gt_lambd += (1 - lambd[s])*lambd[s]**(_t-t-1)*_Gt
			_Gt = sum(gammaR)
			Gt.append(_Gt)
			Gt_lambd += _Gt*lambd[s]**(T-t-1)	
			
			d = -Gt[0]
			for _t in np.arange(1, T-t-1):
				d += lambd[s]**(_t - 1) * (_t*(1 - lambd[s])-lambd[s])*Gt[_t]
			d += (T-t-1)*lambd[s]**(T-t-2)*Gt[-1]
			lambdnew[s] += beta*d

		return lambdnew
    
	def meta_offline_gamma_update(self, h, Vs, lambd, gamma, beta):
		states = [item[0] for item in h]
		rewards = [item[1] for item in h]
		T = len(h)
		gammanew = gamma.copy()
		for (t, (s, r)) in enumerate(h):
			if t == T-1:
				break
			d = (1 - lambd[s])*Vs[states[t+1]]
			for _t in np.arange(2, T-t):
				tmp = np.sum([m*gamma[s]**(m-1)*rewards[t+m] for m in np.arange(1,_t)])
				tmp += _t*gamma[s]**(_t - 1)*Vs[states[t+_t]]
				d += (1-lambd[s])*lambd[s]**(_t - 1)*tmp
			tmp = np.sum([m*gamma[s]**(m-1)*rewards[t+m] for m in np.arange(1,T-t)])
			d += gamma[s]**(T-t-1)*tmp

			gammanew[s] += beta*d
		
		return gammanew
            
	def meta_offline_gamma_update_simplify(self, h, Vs, lambd, gamma, beta):
		rewards = [item[1] for item in h]
		T = len(h)
		gammanew = gamma.copy()
		for (t, (s, r)) in enumerate(h):
			if t == T-1:
				break
			d = np.sum([gamma[s]**(_t-1)*rewards[t+_t] for _t in np.arange(1,T-t)])
			gammanew[s] += beta*d

		return gammanew
			
class MRP1(MRP):
	def __init__(self, n):
		super().__init__(n)
	def get_state_reward(self, state, prev_reward=0):
		if self.isEnd(state): 
			return state, 0., state
		next_state=self.next_state(state)
		if state%2==0: #even state is signal state
			return state, 0.1, next_state
		else: #odd state is noise state
			return state, np.random.randn(), next_state

class MRP2(MRP):
	def __init__(self, n):
		super().__init__(n)
	def get_state_reward(self, state, prev_reward=0):
		if self.isEnd(state): 
			return state, 0, state
		next_state=self.next_state(state)
		if state==0:
			return state, np.random.randn(), next_state
		else:
			return state, -prev_reward, next_state

#function for td lambda learning
def td_learning(traj_list, traj_list_double, num_states, 
				td_update_fn, lambd_update_fn=None, gamma_update_fn=None, 
				init_gamma=1., init_lambd=1.,
				alpha=0.01, beta=0.003):
	
	#initializations
	n_traj = len(traj_list)
	Vs      =np.zeros(num_states)
	gamma_s =np.ones((n_traj+1, num_states))*init_gamma
	lambd_s =np.ones((n_traj+1, num_states))*init_lambd

    #update V, E, lambd, gamma for all states from the td_update_fn
	for i_traj in range(n_traj):
		Vs = td_update_fn(traj_list[i_traj], Vs, lambd_s[i_traj,:], gamma_s[i_traj,:], alpha)
		if lambd_update_fn is not None:
			lambd_s[i_traj+1,:] = lambd_update_fn(traj_list_double[i_traj], Vs, 
												  lambd_s[i_traj,:], gamma_s[i_traj,:], 
												  beta)
		if gamma_update_fn is not None:
			gamma_s[i_traj+1,:] = gamma_update_fn(traj_list_double[i_traj], Vs, 
												  lambd_s[i_traj,:], gamma_s[i_traj,:], 
												  beta/(np.ceil((i_traj+1)/50)))
		
	return Vs, lambd_s, gamma_s


def generate_trajectories(problem, n_states, n_trajectories):
	traj_list = []
	for n in range(n_trajectories):
		s = problem.startState()
		r = 0
		h = []
		while s!=n_states:
			s_prev, r, s = problem.get_state_reward(s, r)
			h.append((s_prev, r))
		traj_list.append(h)
	return traj_list


if __name__ == '__main__':
	n_traj= 2000
	
	# first MRP
	num_states=10
	problem=MRP1(num_states)
	traj_list=generate_trajectories(problem, num_states, n_traj)
	traj_list_double=generate_trajectories(problem, num_states, n_traj)
	Vs, lambd_s, gamma_s=td_learning(traj_list, traj_list_double, num_states, 
								  problem.meta_offline_TD_update, 
								  gamma_update_fn=problem.meta_offline_gamma_update_simplify, 
								  init_gamma=0.2,
								  beta=0.01)
	
	
	plt.figure()
	ax = plt.subplot(111)
	for s in range(num_states-1):
		ax.plot(gamma_s[:,s], label='gamma_'+str(s))
	plt.xlabel('number of trajectories')
	plt.ylabel('gamma')
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.grid()
	print('final gamma: ', gamma_s[-1,:])
	
	
	# second MRP
	num_states=9
	problem=MRP2(num_states)
	traj_list=generate_trajectories(problem, num_states, n_traj)
	traj_list_double=generate_trajectories(problem, num_states, n_traj)
	Vs, lambd_s, gamma_s=td_learning(traj_list, traj_list_double, num_states, 
								  problem.meta_offline_TD_update, 
								  lambd_update_fn=problem.meta_offline_lambda_update,
								  init_lambd=0.5)
	
	plt.figure()
	ax = plt.subplot(111)
	for s in range(num_states):
		ax.plot(lambd_s[:,s], label='lambda_'+str(s))
	plt.xlabel('number of trajectories')
	plt.ylabel('lambda')
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.grid()
	print('final lambda: ', lambd_s[-1,:])
    