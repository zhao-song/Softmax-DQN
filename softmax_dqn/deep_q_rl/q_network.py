"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
Modifications: Zhao Song
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop
import scipy.stats
import logging
import sys

class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, eta,  
		 params_share=True, double_learning=False, 
		 annealing=False, temp=1.0, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
	self.eta = eta
	self.params_share = params_share
	self.double_learning = double_learning
	self.annealing = annealing
	self.temp0 = temp

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.l_out, self.l_feature, self.l_init = self.build_network(network_type, input_width, input_height,
                                        num_actions, num_frames, batch_size)

        if self.freeze_interval > 0:
            self.next_l_out, self.next_l_feature, self.next_l_init = self.build_network(network_type, input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size)
            self.reset_q_hat_share()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
	exp_temp = T.scalar('exploration tuning')

        # Shared variables for training from a minibatch of replayed
        # state transitions, each consisting of num_frames + 1 (due to
        # overlap) images, along with the chosen action and resulting
        # reward and terminal status.
        self.imgs_shared = theano.shared(
            np.zeros((batch_size, num_frames + 1, input_height, input_width),
                     dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
	self.exp_temp_shared = theano.shared(np.float32(self.temp0)) # default without annealing

        # Shared variable for a single state, to calculate q_vals.
        self.state_shared = theano.shared(
            np.zeros((num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)
        feature_vals = lasagne.layers.get_output(self.l_feature, states / input_scale)
        q_params = lasagne.layers.get_all_params(self.l_out)
        q_params_vals = lasagne.layers.get_all_param_values(self.l_out)
 	if self.params_share:
	    w_pi = q_params[-2]
	    b_pi = q_params[-1]
	else:
            params_init = lasagne.layers.get_all_param_values(self.l_init)
	    w_pi = theano.shared(params_init[-2])
	    b_pi = theano.shared(params_init[-1])

        pi_vals = T.nnet.softmax(exp_temp * (T.dot(feature_vals, w_pi) + b_pi))
        
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale)
	    if self.double_learning:
	        next_feature_vals = lasagne.layers.get_output(self.l_feature,
                                                    next_states / input_scale)
                next_q_params = lasagne.layers.get_all_params(self.l_out)
                next_q_params_vals = lasagne.layers.get_all_param_values(self.l_out)
 	        if self.params_share:
	            next_w_pi = next_q_params[-2]
	            next_b_pi = next_q_params[-1]
	        else:
                    next_params_init = lasagne.layers.get_all_param_values(self.l_init)
	            next_w_pi = theano.shared(next_params_init[-2])
	            next_b_pi = theano.shared(next_params_init[-1])
                next_pi_vals = T.nnet.softmax(exp_temp * (T.dot(next_feature_vals, next_w_pi) + next_b_pi)) 
		next_pi_vals = theano.gradient.disconnected_grad(next_pi_vals)
	    else:
	        next_feature_vals = lasagne.layers.get_output(self.next_l_feature,
                                                    next_states / input_scale)
                next_q_params = lasagne.layers.get_all_params(self.next_l_out)
                next_q_params_vals = lasagne.layers.get_all_param_values(self.next_l_out)
 	        if self.params_share:
	            next_w_pi = next_q_params[-2]
	            next_b_pi = next_q_params[-1]
	        else:
                    next_params_init = lasagne.layers.get_all_param_values(self.next_l_init)
	            next_w_pi = theano.shared(next_params_init[-2])
	            next_b_pi = theano.shared(next_params_init[-1])

                next_pi_vals = T.nnet.softmax(exp_temp * (T.dot(next_feature_vals, next_w_pi) + next_b_pi))       
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        terminalsX = terminals.astype(theano.config.floatX)
        actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                          actions.reshape((-1, 1))).astype(theano.config.floatX)

        target = (rewards + (T.ones_like(terminalsX) - terminalsX) *
                 self.discount * T.sum(next_q_vals * next_pi_vals, axis=1, keepdims=True))
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

	if self.params_share:
            params = lasagne.layers.helper.get_all_params(self.l_out)  
	else:
	    params = lasagne.layers.helper.get_all_params(self.l_out)
	    params.append(next_w_pi)
	    params.append(next_b_pi)

        train_givens = {
            states: self.imgs_shared[:, :-1],
            next_states: self.imgs_shared[:, 1:],
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared,
	    exp_temp: self.exp_temp_shared
        }
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss], updates=updates,
                                      givens=train_givens)
        q_givens = {
            states: self.state_shared.reshape((1,
                                               self.num_frames,
                                               self.input_height,
                                               self.input_width))
        }

        pi_givens = {
            states: self.state_shared.reshape((1,
                                               self.num_frames,
                                               self.input_height,
                                               self.input_width)),
	    exp_temp: self.exp_temp_shared
        }

        self._q_vals = theano.function([], q_vals[0], givens=q_givens)
        self._pi_vals = theano.function([], pi_vals[0], givens=pi_givens)

	grad_fc_w = T.grad(loss, self.l_out.W)
	self._grad = theano.function([], outputs=grad_fc_w,
				    givens=train_givens)

    def build_network(self, network_type, input_width, input_height,
                      output_dim, num_frames, batch_size):
        if network_type == "nature_cuda":
            return self.build_nature_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        if network_type == "nature_dnn":
            return self.build_nature_network_dnn(input_width, input_height,
                                                 output_dim, num_frames,
                                                 batch_size)
        elif network_type == "nips_cuda":
            return self.build_nips_network(input_width, input_height,
                                           output_dim, num_frames, batch_size)
        elif network_type == "nips_dnn":
            return self.build_nips_network_dnn(input_width, input_height,
                                               output_dim, num_frames,
                                               batch_size)
        elif network_type == "linear":
            return self.build_linear_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def grad_fc(self, imgs, actions, rewards, terminals):
	"""
	Arguments:
        imgs - b x (f + 1) x h x w numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: gradients for the fully connected linear layer
        """

        self.imgs_shared.set_value(imgs)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        return self._grad()

    def train(self, imgs, actions, rewards, terminals, q_var=None):
        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x h x w numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        self.imgs_shared.set_value(imgs)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
	if self.annealing:
	    if q_var is None:
	        self.exp_temp_shared.set_value(1.0)
	    else:
	        self.exp_temp_shared.set_value( np.float32(self.eta * np.floor(np.log(self.update_counter)) * np.sqrt(self.num_actions / q_var)))

        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            # self.reset_q_hat()
            self.reset_q_hat_share()
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

    def pi_vals(self, state):
        self.state_shared.set_value(state)
	if self.annealing:
            self.exp_temp_shared.set_value(1.0 + np.floor(self.eta * np.sqrt(self.update_counter)))
        return self._pi_vals()

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def choose_action_softmax(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions), -1, -1, -1, None
        pi_vals = self.pi_vals(state)
        q_vals = self.q_vals(state)
        return np.argmax(pi_vals), scipy.stats.entropy(pi_vals, base=2.0), np.max(pi_vals), np.min(pi_vals), q_vals

    def print_params(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.next_l_pi)
	return np.linalg.norm(all_params[0]), np.linalg.norm(all_params[1]), np.linalg.norm(all_params[-2]), np.linalg.norm(all_params[-1]) 
 
    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    def reset_q_hat_share(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

	if not self.params_share:
	    lasagne.layers.helper.set_all_param_values(self.next_l_features, all_params[:-2])
	    logging.info("{}".format("params not share"))

    def build_nature_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import cuda_convnet

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv3 = cuda_convnet.Conv2DCCLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out


    def build_nature_network_dnn(self, input_width, input_height, output_dim,
                                 num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv3 = dnn.Conv2DDNNLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_init = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out, l_hidden1, l_init

    def build_nips_network(self, input_width, input_height, output_dim,
                           num_frames, batch_size):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        from lasagne.layers import cuda_convnet
        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out


    def build_nips_network_dnn(self, input_width, input_height, output_dim,
                               num_frames, batch_size):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        # Import it here, in case it isn't installed.
        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )


        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

	if self.params_share:
            l_pi = lasagne.layers.DenseLayer(
            	l_hidden1,
            	num_units=output_dim,
            	nonlinearity=lasagne.nonlinearities.softmax,
            	#W=lasagne.init.HeUniform(),
            	W=l_out.W,
            	b=l_out.b
            )
	else:
            l_pi = lasagne.layers.DenseLayer(
            	l_hidden1,
            	num_units=output_dim,
            	nonlinearity=lasagne.nonlinearities.softmax,
            	#W=lasagne.init.HeUniform(),
                W=lasagne.init.Normal(.01),
                b=lasagne.init.Constant(.1)
            )

        return l_out, l_pi


    def build_linear_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=None
        )

        return l_out

def main():
    net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000,
                       32, 'nature_cuda')


if __name__ == '__main__':
    main()
