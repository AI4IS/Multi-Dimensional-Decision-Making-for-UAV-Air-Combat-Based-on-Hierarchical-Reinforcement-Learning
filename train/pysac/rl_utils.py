from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from agent.fix_rule_no_att.agent import Agent

MAX_STEP = 300


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    red_detector_action = []
    blue_agent = Agent()
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                # state: [-0.22665706  0.97397465 -0.44414896]
                # state = env.reset() # 状态
                env.reset()
                state_org = env.get_obs_raw()
                state_buffer = []
                x1 = state_org[0]['fighter_obs_list'][0]['pos_x']
                x2 = state_org[1]['fighter_obs_list'][0]['pos_x']
                y1 = state_org[0]['fighter_obs_list'][0]['pos_y']
                y2 = state_org[1]['fighter_obs_list'][0]['pos_y']
                angle2 = np.float(state_org[0]['fighter_obs_list'][0]['course'])
                angle2 = angle2 * np.pi / 180
                k = (y2 - y1) / (x2 - x1 + 1e-6)
                # 计算angle1
                if x2 > x1 and y2 > y1:
                    angle = np.arctan(k)
                elif x2 > x1 and y2 < y1:
                    angle = np.arctan(k) + np.pi * 2
                elif x2 > x1 and y2 == y1:
                    angle = 0
                elif x1 > x2:
                    angle = np.arctan(k) + np.pi
                    print('x1>x2:',angle)
                elif x1 == x2 and y1 > y2:
                    angle = np.pi * 3 / 2
                elif x1 == x2 and y1 < y2:
                    angle = np.pi / 2

                elif x1 == x2 and y1 == y2:
                    angle = angle2
                else:
                    print('warning!')

                d_std = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) / 10

                state_buffer.append(np.cos(angle - angle2))
                state_buffer.append(np.sin(angle - angle2))
                # state_buffer.append(angle - angle2)
                # state_buffer.append(d_std)
                state = state_buffer
                # print('state:',state)
                done = False
                # print('check1')
                # print('angle:',angle)
                # print('angle2:',angle2)
                # print('state:',state)

                for j in range(MAX_STEP):
                    action = np.array(agent.take_action(state))  # 策略网络输入状态，输出动作[-1,1]
                    # env.render = True                    # print('action:',action)                    ##动作预处理
                    obs_list = []
                    next_obs_list = []
                    r_action_list = []
                    b_action_list = []
                    # print('check2')
                    # print('state:',state)
                    # print('action:',action)
                    if j == 0:
                        red_obs_dict, blue_obs_dict = env.get_obs()
                    blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, j)
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
                    # print('red_obs_dict:',red_obs_dict)
                    # print('blue_obs_dict:',blue_obs_dict)
                    # get red action
                    x1 = red_obs_dict['fighter_obs_list'][0]['pos_x']
                    y1 = red_obs_dict['fighter_obs_list'][0]['pos_y']
                    x2 = blue_obs_dict['fighter_obs_list'][0]['pos_x']
                    y2 = blue_obs_dict['fighter_obs_list'][0]['pos_y']
                    # print('坐标：',x1,x2,y1,y2)
                    k = (y2 - y1) / (x2 - x1 + 1e-6)
                    # print('y:',y1,y2)
                    # print('敌机坐标:',x2,y2)                    angle2 = red_obs_dict['fighter_obs_list'][0]['course']
                    angle2 = angle2 * np.pi / 180
                    # 计算angle
                    if x2 > x1 and y2 > y1:
                        angle = np.arctan(k)
                    elif x2 > x1 and y2 < y1:
                        angle = np.arctan(k) + np.pi * 2
                    elif x2 > x1 and y2 == y1:
                        angle = 0
                    elif x1 > x2:
                        angle = np.arctan(k) + np.pi
                        print('x1>x2:b ', angle)

                    elif x1 == x2 and y1 > y2:
                        angle = np.pi * 3 / 2
                    elif x1 == x2 and y1 < y2:
                        angle = np.pi / 2

                    elif x1 == x2 and y1 == y2:
                        angle = angle2
                    else:
                        print('warning!')
                        # 计算angle2
                    # print('目标方位：',angle)                    # print('angle2:',angle2)                    d_std = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) / 10
                    temp_angle = angle

                    obs_list.append(np.cos(angle - angle2))
                    obs_list.append(np.sin(angle - angle2))
                    # obs_list.append(angle - angle2)
                    env.step(np.array([red_detector_action]),np.array([[float((action+1)*180),1,0,0]]),
                    np.array([blue_detector_action]), np.array(b_action_list))
                    ###################################################
                    red_obs_dict, blue_obs_dict = env.get_obs()
                    x1 = red_obs_dict['fighter_obs_list'][0]['pos_x']
                    y1 = red_obs_dict['fighter_obs_list'][0]['pos_y']
                    x2 = blue_obs_dict['fighter_obs_list'][0]['pos_x']
                    y2 = blue_obs_dict['fighter_obs_list'][0]['pos_y']

                    # print('采取动作后航向：',red_obs_dict['fighter_obs_list'][0]['course'])
                    # print('动作值：',action*180)
                    k = (y2 - y1) / (x2 - x1 + 1e-6)
                    # 计算angle
                    angle2 = red_obs_dict['fighter_obs_list'][0]['course']
                    angle2 = angle2 * np.pi / 180
                    # 计算angle
                    if x2 > x1 and y2 > y1:
                        angle = np.arctan(k)
                    elif x2 > x1 and y2 < y1:
                        angle = np.arctan(k) + np.pi * 2
                    elif x2 > x1 and y2 == y1:
                        angle = 0
                    elif x1 > x2:
                        angle = np.arctan(k) + np.pi
                        print('x1>x2:a ', angle)

                    elif x1 == x2 and y1 > y2:
                        angle = np.pi * 3 / 2
                    elif x1 == x2 and y1 < y2:
                        angle = np.pi / 2

                    elif x1 == x2 and y1 == y2:
                        angle = angle2
                    else:
                        print('warning!')
                    d_std = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) / 10
                    temp_delta = temp_angle - angle2
                    # print('temp_delta:',temp_delta)
                    next_obs_list.append(np.cos(angle - angle2))
                    next_obs_list.append(np.sin(angle - angle2))
                    # next_obs_list.append(temp_delta)
                    # next_obs_list.append(d_std)
                    done = env.get_done()

                    r = -temp_delta ** 2
                    ################################################
                    # next_state, reward, done, _ = env.step(action) # 环境采取动作返还下一步状态，奖励，是否结束
                    print('info:',state, action, r, next_obs_list, done)
                    replay_buffer.add(state, action, r, next_obs_list, done)
                    # print('info:',state, action, reward, next_state, done)
                    # sample #                    # [-0.87174714  0.48995608 -0.22461547] [0.02978992648422718] -6.919596472865825 [-0.8753325   0.48352158  0.14732009] False
                    state = next_obs_list
                    episode_return += r
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)  # 更新

                return_list.append(episode_return)
                    # ep_reward += reward
                    # print('Episode:', episode, 'Reward:%i' % int(episode_return))
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})

                pbar.update(1)


    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)