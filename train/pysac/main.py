from SACModel import *
import gym
import numpy as np
import matplotlib.pyplot as plt
from agent.fix_rule_no_att.agent import Agent
from interface import Environment
import importlib
import interface

MAP_PATH = 'maps/1000_1000_fighter10v10.map'

RENDER = True               # 是否显示地图画面
MAX_EPISODE = 5000           # 最大回合数量
MAX_STEP = 500             # 每个回合最大步数

update_every = 50   # target update frequency 每100步更新神经网络参数一次



if __name__ == '__main__':
    agent1_module = importlib.import_module('fix_rule.' + 'agent')
    blue_agent = agent1_module.Agent()
    red_agent_obs_ind = 'raw'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    MAP_PATH = 'C:\\Users\\admin\\Desktop\\MaCA\\maps\\1000_1000_fighter1v1.map'
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)


    red_detector_action = []

    obs_dim = 3
    act_dim = 1
    act_bound = [-1, 1]

    sac = SAC(obs_dim, act_dim, act_bound)

    batch_size = 100
    rewardList = []

    # create blue agent
    blue_agent = Agent()
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []

    # execution
    for episode in range(MAX_EPISODE):
        step_cnt = 0
        env.reset()
        state_org = env.get_obs_raw()
        state_buffer = []
        x1 = state_org[0]['fighter_obs_list'][0]['pos_x']
        x2 = state_org[1]['fighter_obs_list'][0]['pos_x']
        y1 = state_org[0]['fighter_obs_list'][0]['pos_y']
        y2 = state_org[1]['fighter_obs_list'][0]['pos_y']
        k = (y1-y2)/(x2-x1)
        # 计算angle1
        if x2 > x1:
            angle = -np.arctan(k)
        elif x1 > x2 and k > 0:
            angle = -np.arctan(k) + np.pi
        else:
            angle = -np.arctan(k) - np.pi
        angle2 = np.float(state_org[0]['fighter_obs_list'][0]['course'])
        if angle2 > 180:
            angle2 -= 360
        angle2 = angle2*np.pi/180
        d_std = np.sqrt((y2-y1)**2+(x2-x1)**2)/10

        state_buffer.append(np.cos(angle-angle2))
        state_buffer.append(np.sin(angle-angle2))
        state_buffer.append(d_std)
        o = state_buffer
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 20:
                action = sac.get_action(o)
            else:
                action = sac.sample_action()

            obs_list = []
            next_obs_list = []
            r_action_list = []
            b_action_list = []
            if j == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, j)
            b_action_list.append([blue_fighter_action[0]['course'], 1, 0, 0])
            # get red action
            x1 = red_obs_dict['fighter_obs_list'][0]['pos_x']
            y1 = red_obs_dict['fighter_obs_list'][0]['pos_y']
            x2 = blue_obs_dict['fighter_obs_list'][0]['pos_x']
            y2 = blue_obs_dict['fighter_obs_list'][0]['pos_y']
            k = (y1 - y2) / (x2 - x1 + 3e-5)
            # print('zuobiao:', x2, y2)
            # 计算angle
            if x2 > x1:
                angle = -np.arctan(k)
            elif x1 > x2 and k > 0:
                angle = -np.arctan(k) + np.pi
            else:
                angle = -np.arctan(k) - np.pi
            # 计算angle2
            angle2 = red_obs_dict['fighter_obs_list'][0]['course']
            if angle2 > 180:
                angle2 -= 360
            angle2 = angle2 * np.pi / 180
            d_std = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) / 10

            obs_list.append(np.cos(angle-angle2))
            obs_list.append(np.sin(angle-angle2))
            obs_list.append(d_std)
            temp_delta = angle - angle2

            env.step(np.array([red_detector_action]),np.array([[float(action*180),1,0,0]]),
                     np.array([blue_detector_action]), np.array(b_action_list))

            # red_detector_reward, r, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            # detector_reward = red_detector_reward + red_game_reward
            # fighter_reward = r + red_game_reward
            red_obs_dict, blue_obs_dict = env.get_obs()
            x1 = red_obs_dict['fighter_obs_list'][0]['pos_x']
            y1 = red_obs_dict['fighter_obs_list'][0]['pos_y']
            x2 = blue_obs_dict['fighter_obs_list'][0]['pos_x']
            y2 = blue_obs_dict['fighter_obs_list'][0]['pos_y']
            k = (y1 - y2) / (x2 - x1 + 3e-5)
            # 计算angle
            if x2 > x1:
                angle = -np.arctan(k)
            elif x1 > x2 and k > 0:
                angle = -np.arctan(k) + np.pi
            else:
                angle = -np.arctan(k) - np.pi
            # 计算angle2
            angle2 = red_obs_dict['fighter_obs_list'][0]['course']
            if angle2 > 180:
                angle2 -= 360
            angle2 = angle2 * np.pi / 180
            d_std = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) / 10

            next_obs_list.append(np.cos(angle - angle2))
            next_obs_list.append(np.sin(angle - angle2))
            next_obs_list.append(d_std)
            done = env.get_done()

            r = -temp_delta**2*d_std

            sac.replay_buffer.store(obs_list, np.array(action), r, next_obs_list, done)
            o2 = next_obs_list
            d = done

            if episode >=10 and j % update_every == 0:
                for _ in range(update_every):
                    batch = sac.replay_buffer.sample_batch(batch_size)
                    sac.update(data=batch)

            o = o2
            ep_reward += r

            if d:
                break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        rewardList.append(ep_reward)

    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()
