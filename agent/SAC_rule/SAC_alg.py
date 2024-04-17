import argparse
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model
from agent import Agent
from interface import Environment

Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'Plane'  # environment id
RANDOM_SEED = 2  # random seed
RENDER = True  # render while training

# RL training
ALG_NAME = 'SAC'
TRAIN_EPISODES = 100  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 200  # total number of steps for each episode
EXPLORE_STEPS = 100  # 500 for random action sampling in the beginning of training

BATCH_SIZE = 256  # update batch size
HIDDEN_DIM = 32  # size of hidden layers for networks
UPDATE_ITR = 3  # repeated updates for single step
SOFT_Q_LR = 3e-4  # q_net learning rate
POLICY_LR = 3e-4  # policy_net learning rate
ALPHA_LR = 3e-4  # alpha learning rate
POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed update for the policy network and target networks
REWARD_SCALE = 1.  # value range of reward
REPLAY_BUFFER_SIZE = 5e5  # size of the replay buffer

AUTO_ENTROPY = True  # automatically updating variable alpha for entropy

###############################  SAC  ####################################


class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, BATCH_SIZE):
        batch = random.sample(self.buffer, BATCH_SIZE)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SoftQNetwork(Model):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    """ the network for generating non-deterministic (Gaussian distributed) action from the state input """

    def __init__(
            self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-5, log_std_min=-20, log_std_max=2
    ):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.mean_linear = Dense(
            n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w),
            in_channels=hidden_dim, name='policy_mean'
        )
        self.log_std_linear = Dense(
            n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w),
            in_channels=hidden_dim, name='policy_logstd'
        )

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """ generate action with state for calculating gradients """
        state = state.astype(np.float32)
        mean, log_std = self.forward(state)
        std = tf.math.exp(log_std)  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = tf.math.tanh(mean + std * z)  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        # according to original paper, with an extra last term for normalizing different action range
        log_prob = Normal(mean, std).log_prob(mean + std * z) - tf.math.log(1. - action_0**2 +
                                                                            epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the dim of actions to get 1 dim probability; or else use Multivariate Normal.
        log_prob = tf.reduce_sum(log_prob, axis=1)[:, np.newaxis]  # expand dim as reduce_sum causes 1 dim reduced

        return action, log_prob, z, mean, log_std

    def get_action(self, state, greedy=False):
        """ generate action with state for interaction with envronment """
        mean, log_std = self.forward([state])
        std = tf.math.exp(log_std)

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        # print('z:', z)
        action = self.action_range * tf.math.tanh(
            mean + std * z
        )  # TanhNormal distribution as actions; reparameterization trick
        print('action:', mean + std * z)
        action = self.action_range * tf.math.tanh(mean) if greedy else action
        # print('mean:',tf.math.tanh(mean))
        return action.numpy()[0]

    def sample_action(self, ):
        """ generate random actions for exploration """
        a = tf.random.uniform([self.num_actions], -1, 1)
        print('调用')
        return self.action_range * a.numpy()


class SAC:

    def __init__(
            self, state_dim, action_dim, action_range, hidden_dim, replay_buffer, SOFT_Q_LR=3e-5, POLICY_LR=3e-5,
            ALPHA_LR=3e-5
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()

        self.log_alpha = tf.Variable(0, dtype=np.float32, name='log_alpha')
        self.alpha = tf.math.exp(self.log_alpha)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)
        # set mode
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()

        # initialize weights of target networks
        self.target_soft_q_net1 = self.target_ini(self.soft_q_net1, self.target_soft_q_net1)
        self.target_soft_q_net2 = self.target_ini(self.soft_q_net2, self.target_soft_q_net2)

        self.soft_q_optimizer1 = tf.optimizers.Adam(SOFT_Q_LR)
        self.soft_q_optimizer2 = tf.optimizers.Adam(SOFT_Q_LR)
        self.policy_optimizer = tf.optimizers.Adam(POLICY_LR)
        self.alpha_optimizer = tf.optimizers.Adam(ALPHA_LR)

    def target_ini(self, net, target_net):
        """ hard-copy update for initializing target networks """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        """ update all networks in SAC """
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # print('normal_reward:', reward)
        # Training Q Function
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(
            self.target_soft_q_net1(target_q_input), self.target_soft_q_net2(target_q_input)
        ) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # the dim 0 is number of samples

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.soft_q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value1, target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.soft_q_net1.trainable_weights)
        self.soft_q_optimizer1.apply_gradients(zip(q1_grad, self.soft_q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.soft_q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value2, target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.soft_q_net2.trainable_weights)
        self.soft_q_optimizer2.apply_gradients(zip(q2_grad, self.soft_q_net2.trainable_weights))

        # Training Policy Function
        with tf.GradientTape() as p_tape:
            new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
            new_q_input = tf.concat([state, new_action], 1)  # the dim 0 is number of samples
            """ implementation 1 """
            predicted_new_q_value = tf.minimum(self.soft_q_net1(new_q_input), self.soft_q_net2(new_q_input))
            # """ implementation 2 """
            # predicted_new_q_value = self.soft_q_net1(new_q_input)
            policy_loss = tf.reduce_mean(self.alpha * log_prob - predicted_new_q_value)
        p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

        # Updating alpha w.r.t entropy
        # alpha: trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean((self.log_alpha * (log_prob + target_entropy)))
            alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha = tf.math.exp(self.log_alpha)
        else:  # fixed alpha
            self.alpha = 1.
            alpha_loss = 0

        # Soft update the target value nets
        self.target_soft_q_net1 = self.target_soft_update(self.soft_q_net1, self.target_soft_q_net1, soft_tau)
        self.target_soft_q_net2 = self.target_soft_update(self.soft_q_net2, self.target_soft_q_net2, soft_tau)

    def save(self):  # save trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.save_npz(self.soft_q_net1.trainable_weights, extend_path('model_q_net1.npz'))
        tl.files.save_npz(self.soft_q_net2.trainable_weights, extend_path('model_q_net2.npz'))
        tl.files.save_npz(self.target_soft_q_net1.trainable_weights, extend_path('model_target_q_net1.npz'))
        tl.files.save_npz(self.target_soft_q_net2.trainable_weights, extend_path('model_target_q_net2.npz'))
        tl.files.save_npz(self.policy_net.trainable_weights, extend_path('model_policy_net.npz'))
        np.save(extend_path('log_alpha.npy'), self.log_alpha.numpy())  # save log_alpha variable

    def load_weights(self):  # load trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        extend_path = lambda s: os.path.join(path, s)
        tl.files.load_and_assign_npz(extend_path('model_q_net1.npz'), self.soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_q_net2.npz'), self.soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net1.npz'), self.target_soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net2.npz'), self.target_soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_policy_net.npz'), self.policy_net)
        self.log_alpha.assign(np.load(extend_path('log_alpha.npy')))  # load log_alpha variable


if __name__ == '__main__':
    # initialization of env
    # env = gym.make(ENV_ID).unwrapped
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # action_range = env.action_space.high  # scale action, [-action_range, action_range]

    # 环境初始化

    blue_agent = Agent()
    red_agent_obs_ind = 'raw'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    MAP_PATH = 'C:\\Users\\admin\\Desktop\\MaCA\\maps\\1000_1000_fighter1v1.map'
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    print('blue_num', blue_detector_num, blue_fighter_num)
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    state_dim = 6
    action_dim = 1
    action_range = [2.]


    # reproducible
    # env.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # initialization of buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    # initialization of trainer
    # agent = SAC(state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, SOFT_Q_LR, POLICY_LR, ALPHA_LR)
    agent = SAC(state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, SOFT_Q_LR, POLICY_LR, ALPHA_LR)

    t0 = time.time()
    # training loop
    if args.train:
        frame_idx = 0
        all_episode_reward = []

        # need an extra call here to make inside functions be able to use model.forward
        env.reset()
        state_org = env.get_obs_raw()
        # print('state:', state_org)
        state_buffer = []
        state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_x']))
        state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_y']))
        state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['course']))
        state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_x']))
        state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_y']))
        state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['course']))
        # print('state_buffer', state_buffer)
        np.array(state_buffer).astype(np.float32)
        agent.policy_net([state_buffer])

        for episode in range(TRAIN_EPISODES):
            state_buffer = []
            state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_x']))
            state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_y']))
            state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['course']))
            state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_x']))
            state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_y']))
            state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['course']))
            # print('state_buffer', state_buffer)
            state = np.array(state_buffer).astype(np.float32)

            episode_reward = 0
            for step in range(MAX_STEPS):
                if step == 0:
                    red_obs_dict, blue_obs_dict = env.get_obs()
                blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step)
                if RENDER:
                    # env.render()
                    pass
                if frame_idx > EXPLORE_STEPS:
                    action = agent.policy_net.get_action(state)
                else:
                    action = agent.policy_net.sample_action()
                action_list = [action[0]*90, 1, 0, 0]
                # print(np.array([red_detector_action]).astype(np.int32), np.array([action_list]), np.array([blue_detector_action]).astype(np.int32), np.array(blue_fighter_action).astype(np.int32))
                # print('action:',action)
                env.step(np.array([red_detector_action]).astype(np.int32), np.array([action_list]), np.array([blue_detector_action]).astype(np.int32), np.array(blue_fighter_action).astype(np.int32))
                state_buffer = []
                state_org = env.get_obs_raw()
                state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_x']))
                state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_y']))
                state_buffer.append(np.float(state_org[0]['fighter_obs_list'][0]['course']))
                state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_x']))
                state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_y']))
                state_buffer.append(np.float(state_org[1]['fighter_obs_list'][0]['course']))
                # print('state_buffer', state_buffer)
                state = np.array(state_buffer).astype(np.float32)
                next_state = state
                reward = np.array(env.get_reward()[1]).astype(np.float32)
                done = env.get_done()
                red_obs_dict, blue_obs_dict = env.get_obs()

                next_state = next_state.astype(np.float32)
                # print('next_state:', next_state)
                done = 1 if done is True else 0

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                frame_idx += 1

                if len(replay_buffer) > BATCH_SIZE:
                    for i in range(UPDATE_ITR):
                        agent.update(
                            BATCH_SIZE, reward_scale=REWARD_SCALE, auto_entropy=AUTO_ENTROPY,
                            target_entropy=-1. * action_dim
                        )

                if done:
                    break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print('episode_reward:', episode_reward)
            # print(
            #     'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
            #         episode + 1, TRAIN_EPISODES, episode_reward,
            #         time.time() - t0
            #     )
            # )
        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        agent.load_weights()

        # need an extra call here to make inside functions be able to use model.forward
        state = env.reset().astype(np.float32)
        agent.policy_net([state])

        for episode in range(TEST_EPISODES):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                # env.render()
                state, reward, done, info = env.step(agent.policy_net.get_action(state, greedy=True))
                state = state.astype(np.float32)
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0
                )
            )