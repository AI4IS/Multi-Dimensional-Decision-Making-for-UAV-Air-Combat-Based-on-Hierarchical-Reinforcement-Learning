from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from agent import Agent

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
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
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
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
#     return_list = []
#     red_detector_action = []
#     blue_agent = Agent()
#     blue_agent_obs_ind = blue_agent.get_obs_ind()
#     size_x, size_y = env.get_map_size()
#     red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
#     # set map info to blue agent
#     blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)
#     for i in range(10):
#         with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):
#                 episode_return = 0
#                 # state: [-0.22665706  0.97397465 -0.44414896]
#                 # state = env.reset() # 状态
#                 env.reset()
#                 state_org = env.get_obs_raw()
#                 state = []
#                 state.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_x']))
#                 state.append(np.float(state_org[0]['fighter_obs_list'][0]['pos_y']))
#                 state.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_x']))
#                 state.append(np.float(state_org[1]['fighter_obs_list'][0]['pos_y']))
#
#                 print('state:',state)
#                 done = False
#                 j = 0
#                 while not done:
#                     action = agent.take_action(state) # 策略网络输入状态，输出动作
#                     print('action:',action)
#                     ##动作预处理
#                     obs_list = []
#                     next_obs_list = []
#                     r_action_list = []
#                     b_action_list = []
#                     red_fighter_action = []
#                     true_action = np.array([0, 1, 0, 0], dtype=np.int32)
#                     true_action[0] = action[0]
#                     red_fighter_action.append(true_action)
#                     red_fighter_action = np.array(red_fighter_action)
#                     r_action_list.append(red_fighter_action)
#                     if j == 0:
#                         red_obs_dict, blue_obs_dict = env.get_obs()
#                     blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, j)
#
#                     b_action_list.append([blue_fighter_action[0][0], 1, 0, 0]) ##改
#                     # get red action
#                     obs_list.append(np.float(red_obs_dict['fighter_obs_list'][0]['pos_x']))
#                     obs_list.append(np.float(red_obs_dict['fighter_obs_list'][0]['pos_y']))
#                     # obs_list.append(np.float(red_obs_dict['fighter_obs_list'][0]['course']))
#                     obs_list.append(np.float(blue_obs_dict['fighter_obs_list'][0]['pos_x']))
#                     obs_list.append(np.float(blue_obs_dict['fighter_obs_list'][0]['pos_y']))
#                     # obs_list.append(np.float(blue_obs_dict['fighter_obs_list'][0]['course']))
#                     obs_list = np.array(obs_list).astype(np.int32)
#                     ##################################################
#                     # 采取step
#                     env.step(np.array([red_detector_action]).astype(np.int32), r_action_list[0],
#                              np.array([blue_detector_action]).astype(np.int32),
#                              np.array(b_action_list).astype(np.int32))
#                     ###################################################
#                     red_detector_reward, reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
#                     # detector_reward = red_detector_reward + red_game_reward
#                     # fighter_reward = r + red_game_reward
#                     red_obs_dict, blue_obs_dict = env.get_obs()
#                     next_obs_list.append(np.float(red_obs_dict['fighter_obs_list'][0]['pos_x']))
#                     next_obs_list.append(np.float(red_obs_dict['fighter_obs_list'][0]['pos_y']))
#                     # next_obs_list.append(np.float(red_obs_dict['fighter_obs_list'][0]['course']))
#                     next_obs_list.append(np.float(blue_obs_dict['fighter_obs_list'][0]['pos_x']))
#                     next_obs_list.append(np.float(blue_obs_dict['fighter_obs_list'][0]['pos_y']))
#                     next_state = next_obs_list
#                     # next_obs_list.append(np.float(blue_obs_dict['fighter_obs_list'][0]['course']))
#                     # print('next:',j,next_obs_list)
#                     next_obs_list = np.array(next_obs_list).astype(np.int32)
#                     done = env.get_done()
#                     ################################################
#                     # next_state, reward, done, _ = env.step(action) # 环境采取动作返还下一步状态，奖励，是否结束
#                     replay_buffer.add(state, action, reward, next_state, done)
#                     # print('info:',state, action, reward, next_state, done)
#                     # sample #
#                     # [-0.87174714  0.48995608 -0.22461547] [0.02978992648422718] -6.919596472865825 [-0.8753325   0.48352158  0.14732009] False
#                     state = next_state
#                     episode_return += reward
#                     if replay_buffer.size() > minimal_size:
#                         b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
#                         transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
#                         agent.update(transition_dict) # 更新
#                     j += 1
#
#                 return_list.append(episode_return)
#                 if (i_episode+1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#     return return_list
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
                elif x1 == x2 and y1 > y2:
                    angle = np.pi * 3 / 2
                elif x1 == x2 and y1 < y2:
                    angle = np.pi / 2

                elif x1 == x2 and y1 == y2:
                    angle = angle2
                else:
                    print('warning!')

                # T = np.abs(angle - angle2)
                T = angle-angle2
                # state_buffer.append(np.cos(abs(T - np.pi)))
                state_buffer.append(T)
                state_buffer.append(angle2)
                # state_buffer.append(np.sin(angle - angle2))
                # state_buffer.append(abs(T - np.pi))
                state = state_buffer
                # print('state:',state)
                done = False

                for j in range(500):
                    action = np.array(agent.take_action(state))  # 策略网络输入状态，输出动作[-1,1]
                    # env.render = True                    # print('action:',action)                    ##动作预处理
                    obs_list = []
                    next_obs_list = []
                    r_action_list = []
                    b_action_list = []
                    if j == 0:
                        red_obs_dict, blue_obs_dict = env.get_obs()
                    blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, j)
                    b_action_list.append([blue_fighter_action[0][0], 1, 0, 0])

                    # get red action
                    x1 = red_obs_dict['fighter_obs_list'][0]['pos_x']
                    y1 = red_obs_dict['fighter_obs_list'][0]['pos_y']
                    x2 = blue_obs_dict['fighter_obs_list'][0]['pos_x']
                    y2 = blue_obs_dict['fighter_obs_list'][0]['pos_y']
                    angle2 = red_obs_dict['fighter_obs_list'][0]['course']
                    # print('坐标：',x1,x2,y1,y2)
                    k = (y2 - y1) / (x2 - x1 + 1e-6)
                    # print('y:',y1,y2)
                    # print('敌机坐标:',x2,y2)                    angle2 = red_obs_dict['fighter_obs_list'][0]['course']
                    angle2 = angle2 * np.pi / 180
                    # print('angle2:',angle2)
                    # 计算angle
                    if x2 > x1 and y2 > y1:
                        angle = np.arctan(k)
                    elif x2 > x1 and y2 < y1:
                        angle = np.arctan(k) + np.pi * 2
                    elif x2 > x1 and y2 == y1:
                        angle = 0
                    elif x1 > x2:
                        angle = np.arctan(k) + np.pi
                    elif x1 == x2 and y1 > y2:
                        angle = np.pi * 3 / 2
                    elif x1 == x2 and y1 < y2:
                        angle = np.pi / 2

                    elif x1 == x2 and y1 == y2:
                        angle = angle2
                    else:
                        print('warning!')
                        # 计算angle2

                    temp_angle = angle
                    # T = np.abs(angle - angle2)
                    T = angle - angle2
                    # print('angle:',angle)
                    # obs_list.append(np.cos(abs(T - np.pi)))
                    obs_list.append(T)
                    obs_list.append(angle2)
                    d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
                    if d <= 120:
                        attack = blue_obs_dict['fighter_obs_list'][0]['id']
                        print('attack:',attack)
                    else:
                        attack = 0
                    # state_buffer.append(np.sin(angle - angle2))
                    # obs_list.append(abs(T - np.pi))
                    # obs_list.append(angle - angle2)

                    env.step(np.array([red_detector_action]),np.array([[float((action+1)*180),1,0,0]]),
                    np.array([blue_detector_action]), np.array(b_action_list))
                    ###################################################

                    red_obs_dict, blue_obs_dict = env.get_obs()
                    blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, j+1)
                    x1 = red_obs_dict['fighter_obs_list'][0]['pos_x']
                    y1 = red_obs_dict['fighter_obs_list'][0]['pos_y']
                    x2 = blue_obs_dict['fighter_obs_list'][0]['pos_x']
                    y2 = blue_obs_dict['fighter_obs_list'][0]['pos_y']

                    # print('采取动作后航向：',red_obs_dict['fighter_obs_list'][0]['course'])
                    # print('动作值：',action*180)
                    k = (y2 - y1) / (x2 - x1 + 1e-6)
                    # 计算angle
                    angle2 = (action+1)*180
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
                    elif x1 == x2 and y1 > y2:
                        angle = np.pi * 3 / 2
                    elif x1 == x2 and y1 < y2:
                        angle = np.pi / 2

                    elif x1 == x2 and y1 == y2:
                        angle = angle2
                    else:
                        print('warning!')
                    # d_std = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) / 10
                    # T = np.abs(angle - angle2)
                    T = angle - angle2
                    # temp_delta = temp_angle - angle2
                    # print('temp_delta:',temp_delta)
                    next_obs_list.append(float(T))
                    next_obs_list.append(float(angle2))

                    # state_buffer.append(np.sin(angle - angle2))
                    # next_obs_list.append(abs(T - np.pi))
                    # next_obs_list.append(temp_delta)
                    # next_obs_list.append(d_std)
                    done = env.get_done()

                    r = -(angle2-temp_angle) ** 2
                    ################################################
                    # next_state, reward, done, _ = env.step(action) # 环境采取动作返还下一步状态，奖励，是否结束
                    replay_buffer.add(obs_list, action, r[0], next_obs_list, done)
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