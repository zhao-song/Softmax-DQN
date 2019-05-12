"""
The NeuralAgent class wraps a deep Q-network for training and testing
in the Arcade learning environment.

Author: Nathan Sprague
Modifications: Zhao Song

"""

import os
import cPickle
import time
import logging

import numpy as np

import ale_data_set
import theano

import sys
sys.setrecursionlimit(10000)

class NeuralAgent(object):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref,
                 replay_start_size, update_frequency, rng,
		 grad_frequency, grad_times):

        self.network = q_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_pref = exp_pref
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.rng = rng
	self.grad_frequency = grad_frequency
	self.grad_times = grad_times

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.network.lr).replace(".", "p") + "_" \
                       + "{}".format(self.network.discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.num_actions = self.network.num_actions


        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()
	self._open_gradient_file()

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        # Exponential moving average of runtime performance.
        self.steps_sec_ema = 0.

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,max_reward,min_reward,mean_q\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon,entropy,max_prob,min_prob,max_q,min_q\n')
        self.learning_file.flush()

    def _open_gradient_file(self):
	self.gradient_file = open(self.exp_dir + '/gradient.csv', 'w', 0)
	self.gradient_file.write('iteration, variance, mean\n')
	self.gradient_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{:d}, {:d}, {}, {:.4f}, {}, {}, {:.4f}\n".format(epoch, num_episodes, self.total_reward,
                                        self.total_reward / float(num_episodes), max(self.episode_reward_set),
                                        min(self.episode_reward_set), holdout_sum)
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
	if not self.entropy_averages:
	    entropy_avg = -1.0;
	else:
	    entropy_avg = np.mean(self.entropy_averages)

	if not self.max_prob_averages:
	    max_prob_avg = -1.0;
	else:
	    max_prob_avg = np.mean(self.max_prob_averages)

	if not self.min_prob_averages:
	    min_prob_avg = -1.0;
	else:
	    min_prob_avg = np.mean(self.min_prob_averages)

	if not self.max_q_vals_averages:
	    max_q_vals_avg = np.inf;
	else:
	    max_q_vals_avg = np.mean(self.max_q_vals_averages)

	if not self.min_q_vals_averages:
	    min_q_vals_avg = np.inf;
	else:
	    min_q_vals_avg = np.mean(self.min_q_vals_averages)

        out = "{:.4f}, {:f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(np.mean(self.loss_averages),
                               self.epsilon, entropy_avg, max_prob_avg, min_prob_avg, max_q_vals_avg, min_q_vals_avg)
        self.learning_file.write(out)
        self.learning_file.flush()

    def _update_gradient_file(self, iter_num, grad_var, grad_mean):
	out = "{}, {}, {}\n".format(iter_num, grad_var, grad_mean)
	self.gradient_file.write(out)
        self.gradient_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []
        self.entropy_averages = []
        self.max_prob_averages = []
        self.min_prob_averages = []
        self.max_q_vals_averages = []
        self.min_q_vals_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action


    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1

        #TESTING---------------------------
        if self.testing:
            self.episode_reward += reward
            action, entropy, max_prob, min_prob, q_vals = self._choose_action_softmax(self.test_data_set, .05,
                                         observation, np.clip(reward, -1, 1))

        #NOT TESTING---------------------------
        else:

            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                action, entropy, max_prob, min_prob, q_vals = self._choose_action_softmax(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
		    if q_vals is None:
		        q_var = None
		    else:
		        q_var = np.var(q_vals)
                    loss = self._do_training(q_var)

                    self.batch_counter += 1
                    self.loss_averages.append(loss)
		    if entropy != -1:
                        self.entropy_averages.append(entropy)
		    if max_prob != -1:
                        self.max_prob_averages.append(max_prob)
		    if min_prob != -1:
                        self.min_prob_averages.append(min_prob)

		    if q_vals is not None:
                        self.max_q_vals_averages.append(max(q_vals))
                        self.min_q_vals_averages.append(min(q_vals))

		    if self.network.update_counter % self.grad_frequency == 0:
		        grad_var, grad_mean = self._do_grad()
                        self._update_gradient_file(self.network.update_counter, grad_var, grad_mean)
		    # print self.network.print_params()
		    # logging.info("{:.4f},{:.4f},{:.4f},{:.4f}".format(self.network.print_params())) 
            else: # Still gathering initial random data...
                action, entropy, max_prob, min_prob, q_vals = self._choose_action_softmax(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))


        self.last_action = action
        self.last_img = observation

        return action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _choose_action_softmax(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False)
	entropy = -1.0
	max_prob = -1.0
	min_prob = -1.0
	q_vals = None
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action, entropy, max_prob, min_prob, q_vals = self.network.choose_action_softmax(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action, entropy, max_prob, min_prob, q_vals

    def _do_training(self, q_var):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        imgs, actions, rewards, terminals = \
                                self.data_set.random_batch(
                                    self.network.batch_size)
        return self.network.train(imgs, actions, rewards, terminals, q_var)

    def _do_grad(self):
	"""
	Compute the gradients in the last layer.
	"""
	grad_cache = np.zeros([512, self.num_actions, self.grad_times], dtype=theano.config.floatX)
	for i in xrange(self.grad_times):
	    imgs, actions, rewards, terminals = \
	                            self.data_set.random_batch(
				    self.network.batch_size)
	    grad_cache[:, :, i] = self.network.grad_fc(imgs, actions, rewards, terminals)
        grad_var = np.var(grad_cache, axis=2)
        grad_mean = np.mean(grad_cache, axis=2)
        return np.mean(grad_var), np.mean(grad_mean)

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
		self.episode_reward_set.append(self.episode_reward)
        else:

            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)

            rho = 0.98
            self.steps_sec_ema *= rho
            self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)

            logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
                self.step_counter/total_time, self.steps_sec_ema))

            if self.batch_counter > 0:
                self._update_learning_file()
                # logging.info("average loss: {:.4f}".format(\
                #                 np.mean(self.loss_averages)))


    def finish_epoch(self, epoch):
        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0
	self.episode_reward_set = []

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            imgs, _, _, _ = self.data_set.random_batch(holdout_size)
            self.holdout_data = imgs[:, :self.phi_length]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)


if __name__ == "__main__":
    pass
