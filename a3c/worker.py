import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
from scipy.linalg import toeplitz

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

# from lib import plotting
from lib.atari.state_processor import StateProcessor
from lib.atari import helpers as atari_helpers
from estimators import ValueEstimator, PolicyEstimator

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
  """
  Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
  The ordering of the variables in the lists must be identical.
  """
  v1_list = list(sorted(v1_list, key=lambda v: v.name))
  v2_list = list(sorted(v2_list, key=lambda v: v.name))

  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)

  return update_ops

def make_train_op(local_estimator, global_estimator, _optimizer):
  """
  Creates an op that applies local estimator gradients
  to the global estimator.
  """
  local_grads, _ = zip(*local_estimator.grads_and_vars)
  # Clip gradients
  local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
  _, global_vars = zip(*global_estimator.grads_and_vars)
  local_global_grads_and_vars = list(zip(local_grads, global_vars))
  return _optimizer.apply_gradients(local_global_grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())


class Worker(object):
  """
  An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

  Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    policy_net: Instance of the globally shared policy net
    value_net: Instance of the globally shared value net
    global_counter: Iterator that holds the global step
    discount_factor: Reward discount factor
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    max_global_steps: If set, stop coordinator when global_counter > max_global_steps
  """
  def __init__(self, name, env, policy_net, value_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
    self.name = name
    # self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.contrib.framework.get_global_step()
    self.global_policy_net = policy_net
    self.global_value_net = value_net
    self.global_counter = global_counter
    self.local_counter = itertools.count()
    self.sp = StateProcessor()
    self.summary_writer = summary_writer
    self.env = env

    # Create local policy/value nets that are not updated asynchronously
    with tf.variable_scope(name):
      self.policy_net = PolicyEstimator(policy_net.num_outputs)
      self.value_net = ValueEstimator(reuse=True)

    # Op to copy params from global policy/valuenets
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    self.vnet_train_op = make_train_op(self.value_net, self.global_value_net, self.global_value_net.optimizer)
    self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net, self.global_policy_net.optimizer)
    # TODO
    # add operator to update gamma
    self.gamma_train_op = make_train_op(self.policy_net, self.global_policy_net, self.global_policy_net.optimizer_gamma)

    self.state = None

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Initial state
      self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
      try:
        while not coord.should_stop():
          # Copy Parameters from the global networks
          sess.run(self.copy_params_op)

          # Collect some experience
          transitions, local_t, global_t = self.run_n_steps(t_max, sess)

          if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            coord.request_stop()
            return
        
          new_grads_and_vars = self.update(transitions, sess, update_case="local")
          sess.run(self.copy_params_op)
          
          self.update(transitions, sess, update_case="gamma", grads_and_vars=new_grads_and_vars)
          
          
          # Update the global networks
          self.update(transitions, sess, update_case="global")

      except tf.errors.CancelledError:
        return

  def _policy_net_predict(self, state, sess):
    gamma = sess.run(self.policy_net.gamma_var)
    gamma_vector = gamma*np.ones((1, 1))
    feed_dict = { self.policy_net.states: [state], 
                 self.policy_net.gamma_ph: gamma_vector }
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

  def _value_net_predict(self, state, sess):
    gamma = sess.run(self.value_net.gamma_var)
    gamma_vector = gamma*np.ones((1, 1))
    feed_dict = { self.value_net.states: [state],
                 self.value_net.gamma_ph: gamma_vector}
    preds = sess.run(self.value_net.predictions, feed_dict)
    return preds["logits"][0]

  def run_n_steps(self, n, sess):
    transitions = []
    for _ in range(n):
      # Take a step
      action_probs = self._policy_net_predict(self.state, sess)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      next_state, reward, done, _ = self.env.step(action)
      next_state = atari_helpers.atari_make_next_state(self.state, self.sp.process(next_state))

      # Store transition
      transitions.append(Transition(
        state=self.state, action=action, reward=reward, next_state=next_state, done=done))

      # Increase local and global counters
      local_t = next(self.local_counter)
      global_t = next(self.global_counter)

      if local_t % 100 == 0:
        tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

      if done:
        self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
        break
      else:
        self.state = next_state
    return transitions, local_t, global_t

  def calc_placeholders(self, transitions, sess):
    reward = 0.
    T = len(transitions)
    if not transitions[-1].done:
      reward = self._value_net_predict(transitions[-1].next_state, sess)
    rew_matrix = np.diag([reward]*(T + 1))
    rew_matrix = rew_matrix[1:,:]
    col_list = [trans.reward for trans in transitions[::-1]]
    row_list = [0.]*(T + 1)
    row_list[0] = col_list[0]
    row_list[1] = reward
    rew_matrix = rew_matrix + toeplitz(col_list, row_list)
    powers = np.arange(0,T + 1)

    # Accumulate minibatch exmaples
    states = np.array([trans.state for trans in transitions[::-1]])
    actions = [trans.action for trans in transitions[::-1]]
    
    gamma = sess.run(self.value_net.gamma_var)
    gamma_vector = gamma*np.ones((T, 1))
    
    Vtest = sess.run(self.value_net.predictions, feed_dict={self.value_net.states: states, self.value_net.gamma_ph: gamma_vector})
    Vtest = Vtest["logits"]
    return rew_matrix, powers, states, actions, gamma_vector, Vtest

  def update(self, transitions, sess, update_case='global', grads_and_vars=None):
    """
    Updates global policy and value networks based on collected experience

    Args:
      transitions: A list of experience transitions
      sess: A Tensorflow session
    """
    """
    # If we episode was not done we bootstrap the value from the last state
    reward = 0.
    T = len(transitions)
    if not transitions[-1].done:
      reward = self._value_net_predict(transitions[-1].next_state, sess)
    rew_matrix = np.diag([reward]*(T + 1))
    rew_matrix = rew_matrix[1:,:]
    col_list = [trans.reward for trans in transitions[::-1]]
    row_list = [0.]*(T + 1)
    row_list[0] = col_list[0]
    row_list[1] = reward
    rew_matrix = rew_matrix + toeplitz(col_list, row_list)
    powers = np.arange(0,T + 1)

    # Accumulate minibatch exmaples
    states = np.array([trans.state for trans in transitions[::-1]])
    actions = [trans.action for trans in transitions[::-1]]
    
    gamma = sess.run(self.value_net.gamma_var)
    gamma_vector = gamma*np.ones((T, 1))
    
    Vtest = sess.run(self.value_net.predictions, feed_dict={self.value_net.states: states, self.value_net.gamma_ph: gamma_vector})
    Vtest = Vtest["logits"]
    """
    rew_matrix, powers, states, actions, gamma_vector, Vtest = self.calc_placeholders(transitions, sess)
    
    
    feed_dict = {
      #self.gamma_ph: gamma_vector,
      self.policy_net.states: states,
      self.policy_net.actions: actions,
      self.policy_net.reward_matrix: rew_matrix,
      self.policy_net.powers: powers,
      self.policy_net.Vtest: Vtest,
      self.policy_net.gamma_ph: gamma_vector,
      self.policy_net.entropy_weight: 0.1,
      self.value_net.states: states,
      self.value_net.reward_matrix: rew_matrix,
      self.value_net.powers: powers,
      self.value_net.gamma_ph: gamma_vector,
    }
    if update_case == "global":
      # Train the global estimators using local gradients
      global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run([
        self.global_step,
        self.policy_net.loss,
        self.value_net.loss,
        self.pnet_train_op,
        self.vnet_train_op,
        self.policy_net.summaries,
        self.value_net.summaries
      ], feed_dict)

      # Write summaries
      if self.summary_writer is not None:
        self.summary_writer.add_summary(pnet_summaries, global_step)
        self.summary_writer.add_summary(vnet_summaries, global_step)
        self.summary_writer.flush()

      return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
    elif update_case == "local": # only update local network to generate auxilliary trajectories, then take derivative wrt new params at new loss function

      feed_dict_new = {
        #self.gamma_ph: gamma_vector,
        self.policy_net.states: states,
        self.policy_net.actions: actions,
        self.policy_net.reward_matrix: rew_matrix,
        self.policy_net.powers: powers,
        self.policy_net.Vtest: Vtest,
        self.policy_net.gamma_ph: gamma_vector,
        self.policy_net.entropy_weight: 0.,
        self.value_net.states: states,
        self.value_net.reward_matrix: rew_matrix,
        self.value_net.powers: powers,
        self.value_net.gamma_ph: gamma_vector,
      }
      sess.run([self.policy_net.train_op, self.value_net.train_op], feed_dict)
      new_transitions, _, _ = self.run_n_steps(5, sess)
      new_rew_matrix, new_powers, new_states, new_actions, new_gamma_vector, new_Vtest = self.calc_placeholders(new_transitions, sess)
      return sess.run(self.policy_net.grads_and_vars, feed_dict_new)
    else: # update gamma
      feed_dict_new = feed_dict
      _i = 0
      for grad, var in grads_and_vars:
        feed_dict_new[self.policy_net.external_grad[_i]]=grad
        _i += 1
      sess.run(self.gamma_train_op, feed_dict_new)
    return
