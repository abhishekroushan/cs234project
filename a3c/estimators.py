import numpy as np
import tensorflow as tf
import collections

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

def build_shared_network(X, add_summaries=False):
  """
  Builds a 3-layer network conv -> conv -> fc as described
  in the A3C paper. This network is shared by both the policy and value net.

  Args:
    X: Inputs
    gamma_ph: placeholder of gamma, batch_size*emb_size
    add_summaries: If true, add layer summaries to Tensorboard.

  Returns:
    Final layer activations.
  """
  # TODO
  # may need to add embed(gamma) as input
  # Three convolutional layers
  conv1 = tf.contrib.layers.conv2d(
    X, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
  conv2 = tf.contrib.layers.conv2d(
    conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")

  # Fully connected layer
  fc1 = tf.contrib.layers.fully_connected(
    inputs=tf.contrib.layers.flatten(conv2),
    num_outputs=256,
    scope="fc1")
  
  #fc1 = tf.concat([fc0, gamma_ph], axis=1)

  if add_summaries:
    tf.contrib.layers.summarize_activation(conv1)
    tf.contrib.layers.summarize_activation(conv2)
    tf.contrib.layers.summarize_activation(fc1)

  return fc1

def _value_net_predict(self, state, sess):
  feed_dict = { self.value_net.states: [state] }
  preds = sess.run(self.value_net.predictions, feed_dict)
  return preds["logits"][0]
'''
def calc_targets(rew_matrix, gamma_var, states, actions, rewards, next_state, done_mask, vnet):
  """
  Caltulate targets based on collected experience
  
  Args:
    gamma_var: variable gamma
    transitions: a list of experience transitions
    vnet: value estimator
  """
  reward = 0.0
  # If the episode was not done we bootstrap the value from the last state
  if not done_mask:
    reward = vnet.logits[transitions[-1].next_state]

  # Accumulate minibatch exmaples
  states = []
  policy_targets = []
  value_targets = []
  actions = []
  
  for transition in transitions[::-1]:
    reward = transition.reward + gamma_var * reward
    policy_target = (reward - vnet.logits[transition.state])
    states.append(transition.state)
    actions.append(transition.action)
    policy_targets.append(policy_target)
    value_targets.append(reward)
  
  return policy_targets, value_targets
'''

class PolicyEstimator():
  """
  Policy Function approximator. Given a observation, returns probabilities
  over all possible actions.

  Args:
    num_outputs: Size of the action space.
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
  """

  def __init__(self, num_outputs, gamma_init=0.99, gamma_emb_size=1, reuse=False, trainable=True):
    self.num_outputs = num_outputs

    # Placeholders for our input
    # Our input are 4 RGB frames of shape 160, 160 each
    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
    # Integer id of which action was selected
    self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
    self.reward_matrix = tf.placeholder(shape=[None, None], dtype=tf.float32, name="rewards")
    self.powers = tf.placeholder(shape=[None], dtype=tf.float32, name="powers")
    self.Vtest = tf.placeholder(shape=[None], dtype=tf.float32, name="V_st")
    #self.next_state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="next_X")
    # The TD target value
    # TODO
    # change to expression of variable gamma
    #self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
    
    # placeholder for embedding of gamma (as part of input)
    self.gamma_ph = tf.placeholder(shape=[None, gamma_emb_size], dtype=tf.float32, name="gamma_ph")
    
    
    self.entropy_weight = tf.placeholder(shape=[], dtype=tf.float32, name="entropy_weight")

    # Normalize
    X = tf.to_float(self.states) / 255.0
    self.batch_size = tf.shape(self.states)[0]

    # Graph shared with Value Net
    with tf.variable_scope("shared", reuse=reuse):
      self.gamma_var = tf.get_variable("gamma_var", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(gamma_init))
      fc1 = build_shared_network(X, add_summaries=(not reuse))
      fc2 = tf.concat([fc1, self.gamma_ph], axis=1)


    with tf.variable_scope("policy_net"):
      self.logits = tf.contrib.layers.fully_connected(fc2, num_outputs, activation_fn=None)
      self.probs = tf.nn.softmax(self.logits) + 1e-8

      self.predictions = {
        "logits": self.logits,
        "probs": self.probs
      }

      # We add entropy to the loss to encourage exploration
      self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
      self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

      # Get the predictions for the chosen actions only
      gather_indices = tf.range(self.batch_size) * tf.shape(self.probs)[1] + self.actions
      self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)
      
      self.targets = self.calc_value_targets(self.reward_matrix, self.powers)
      self.advantage = self.targets - self.Vtest
      # TODO
      # check loss, change targets
      self.losses = - (tf.log(self.picked_action_probs) * self.advantage + self.entropy_weight * self.entropy)
      self.loss = tf.reduce_sum(self.losses, name="loss")

      tf.summary.scalar(self.loss.op.name, self.loss)
      tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
      tf.summary.histogram(self.entropy.op.name, self.entropy)

      if trainable:
        # TODO
        # add gamma and second order derivative
        # self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars_all = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars_all if grad is not None and 'gamma_var' not in var.name]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())
        # TODO
        # a bug here
        self.second_grads = []
        self.train_op_gamma = []

    # Merge summaries from this network and the shared network (but not the value net)
    var_scope_name = tf.get_variable_scope().name
    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
    sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
    sumaries = [s for s in summary_ops if var_scope_name in s.name]
    self.summaries = tf.summary.merge(sumaries)
    
  def calc_value_targets(self, rew_matrix, powers):
    #rew_matrix = rew_matrix
    #rew_matrix = rew_matrix[1:,:]
    gamma_vector = tf.pow(self.gamma_var, powers)
    #gamma_vector = tf.stack([tf.pow(self.gamma_var, _i) for _i in powers])
    gamma_vector = tf.reshape(gamma_vector, [-1, 1])
    return tf.matmul(rew_matrix, gamma_vector)
        


class ValueEstimator():
  """
  Value Function approximator. Returns a value estimator for a batch of observations.

  Args:
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
  """

  def __init__(self, gamma_emb_size=1, reuse=False, trainable=True):
    # Placeholders for our input
    # Our input are 4 RGB frames of shape 160, 160 each
    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
    self.gamma_ph = tf.placeholder(shape=[None, gamma_emb_size], dtype=tf.float32, name="gamma_ph")
    self.reward_matrix = tf.placeholder(shape=[None, None], dtype=tf.float32, name="rewards")
    self.powers = tf.placeholder(shape=[None], dtype=tf.float32, name="powers")
    

    X = tf.to_float(self.states) / 255.0

    # Graph shared with Value Net
    with tf.variable_scope("shared", reuse=reuse):
      self.gamma_var = tf.get_variable("gamma_var", shape=[], dtype=tf.float32)
      fc1 = build_shared_network(X, add_summaries=(not reuse))
      fc2 = tf.concat([fc1, self.gamma_ph], axis=1)

    with tf.variable_scope("value_net"):
      self.logits = tf.contrib.layers.fully_connected(
        inputs=fc2,
        num_outputs=1,
        activation_fn=None)
      self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")
      
      self.targets = self.calc_value_targets(self.reward_matrix, self.powers)

      self.losses = tf.squared_difference(self.logits, self.targets)
      self.loss = tf.reduce_sum(self.losses, name="loss")

      self.predictions = {
        "logits": self.logits
      }

      # Summaries
      prefix = tf.get_variable_scope().name
      tf.summary.scalar(self.loss.name, self.loss)
      tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
      tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
      tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
      tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
      tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
      tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
      tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
      tf.summary.histogram("{}/values".format(prefix), self.logits)

      if trainable:
        # self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None and 'gamma_var' not in var.name]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.contrib.framework.get_global_step())

    var_scope_name = tf.get_variable_scope().name
    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
    sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
    sumaries = [s for s in summary_ops if var_scope_name in s.name]
    self.summaries = tf.summary.merge(sumaries)
  def calc_value_targets(self, rew_matrix, powers):
    #rew_matrix = rew_matrix
    #rew_matrix = rew_matrix[1:,:]
    gamma_vector = tf.pow(self.gamma_var, powers)
    #gamma_vector = tf.stack([tf.pow(self.gamma_var, _i) for _i in powers])
    gamma_vector = tf.reshape(gamma_vector, [-1, 1])
    return tf.matmul(rew_matrix, gamma_vector)