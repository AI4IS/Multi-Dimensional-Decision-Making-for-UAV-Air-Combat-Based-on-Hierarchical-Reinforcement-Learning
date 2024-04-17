#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from base_agent import BaseAgent
import interface
from world import config
import copy
import random
import numpy as np


class Agent(BaseAgent):
    def __init__(self):
        # 初始化
        BaseAgent.__init__(self)
        self.obs_ind = 'raw'

        # self.left_border = True
        # self.right_border = False

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        # 根据需要自行选择函数实现形式
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num
        self.rand = np.zeros(self.fighter_num)
        self.pre = np.zeros(self.fighter_num)
        self.find_step = -1
        self.turn = False
        pass

# 当前存在的问题：同一目标多架飞机发射导弹（任务分配问题）
    def get_action(self, obs_dict, step_cnt):
        # obs_dict为状态，step_cnt为当前步数 从1开始
        if step_cnt == 0:
            self.pre = np.zeros(self.fighter_num)
            self.search = False
            self.find_step = -1

        detector_action = []
        fighter_action = []
        for y in range(self.fighter_num):
            if obs_dict['fighter_obs_list'][y]['last_reward'] != 0:
                pass
                 # print("上一步奖励:",obs_dict['fighter_obs_list'][y]['last_reward'])

            # true_action = np.array([0, 1, 0, 0], dtype=np.int32)
            if obs_dict['fighter_obs_list'][y]['alive']:
                # print('y:', obs_dict['fighter_obs_list'][y]['pos_y'])
                true_action = np.array([0, 1, 2, 0], dtype=np.int32)
                true_action[0] = self.pre[y]
                # if y == 1:
                #     true_action[2] = 0
                if step_cnt % 50 == 0 and self.search == True:
                        true_action[0] = self.rand[y]
                        self.pre[y] = true_action[0]
                        self.rand[y] = random.randint(0,359)

                # if obs_dict['fighter_obs_list'][y]['striking_list']:
                #     true_action[3] = list(set(obs_dict['fighter_obs_list'][y]['striking_list']))[0]

                # 被动雷达发现目标(测试结果距离180发现)
                if (obs_dict['fighter_obs_list'][y]['j_recv_list']):
                    print('attacker:%d:被动雷达发现目标' % obs_dict['fighter_obs_list'][y]['id'])
                    print('打击列表：',obs_dict['fighter_obs_list'][y]['striking_list'])
                    # print('attacker发现列表：', obs_dict['fighter_obs_list'][y]['j_recv_list'])
                    true_action[3] = obs_dict['fighter_obs_list'][y]['j_recv_list'][0]['id']
                    self.find_step = step_cnt

                    bj_id = obs_dict['fighter_obs_list'][y]['j_recv_list'][0]['id']

                    true_action[0] = obs_dict['fighter_obs_list'][y]['j_recv_list'][0]['direction']
                    # if step_cnt
                    # true_action[3] = bj_id
                    self.search = True
                    # print("x位置：",obs_dict['fighter_obs_list'][y]['pos_x'])
                    # print("")
                    # 攻击条件


                # 雷达探测到目标(180,远距离导弹距离120时发射有效,近距离导弹距离50时发射有效)
                if (obs_dict['fighter_obs_list'][y]['r_visible_list']):

                    print('attacker:%d:主动雷达发现目标' % obs_dict['fighter_obs_list'][y]['id'])
                    # print('attacker:发现列表：', obs_dict['fighter_obs_list'][y]['r_visible_list'])

                    r_id = obs_dict['fighter_obs_list'][y]['id']
                    r_x = obs_dict['fighter_obs_list'][y]['pos_x']
                    r_y = obs_dict['fighter_obs_list'][y]['pos_y']

                    b_id = obs_dict['fighter_obs_list'][y]['r_visible_list'][0]['id']
                    b_x = obs_dict['fighter_obs_list'][y]['r_visible_list'][0]['pos_x']
                    b_y = obs_dict['fighter_obs_list'][y]['r_visible_list'][0]['pos_y']

                    d = interface.get_distance(r_x, r_y, b_x, b_y)
                    agl = interface.angle_cal(r_x, r_y, b_x, b_y)
                    # self.search = True
                    # if y == 1:
                    #     print('attacker:%d:主动雷达发现目标' % obs_dict['fighter_obs_list'][y]['id'])
                    #     print('attacker:发现列表：', obs_dict['fighter_obs_list'][y]['r_visible_list'])
                    #     print('attacker:敌我距离：', d)
                    #     print("")
                    # print('attacker:敌我距离：', d)
                    # print("")
                    # print('attacker:%d:主动雷达发现目标' % obs_dict['fighter_obs_list'][y]['id'])
                    # print('attacker:发现列表：', obs_dict['fighter_obs_list'][y]['r_visible_list'])
                    print('attacker:敌我距离：', d)
                    # print('attacker:打击列表：', obs_dict['fighter_obs_list'][y]['striking_dict_list'])

                    # print("")

                    true_action[0] = agl
                    self.pre[y] = agl
                    # 攻击条件
                    # obs_dict['fighter_obs_list'][y]['striking_list']


                    if 50 < d <= 120:
                        # true_action[3] = b_id
                        if obs_dict['fighter_obs_list'][y]['l_missile_left'] == 0:
                            true_action[3] = 0
                    if d <= 50:
                        # true_action[3] = b_id + 10
                        if obs_dict['fighter_obs_list'][y]['s_missile_left'] == 0:
                            true_action[3] = 0
            true_action[0] = 0

            if y==1 and obs_dict['fighter_obs_list'][y]['pos_x'] == 300:
                self.turn = True
            if y==0:
                true_action[0] = 0
                true_action[2] = 2
            if y==1 and self.turn == True:

                true_action[0] = 180
            if y==1:
                true_action[2] = 2


            fighter_action.append(copy.deepcopy(true_action))

        fighter_action = np.array(fighter_action)
        return detector_action, fighter_action
