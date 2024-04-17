#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from base_agent import BaseAgent
import interface
from world import config
import copy
import math
import random
import numpy as np


class Agent(BaseAgent):
    def __init__(self):
        # 初始化
        BaseAgent.__init__(self)
        self.obs_ind = 'raw'
        self.team_list = []
        self.jugement_list = []
        self.open_radar_list = []
        self.share_list = []
        self.crash_list = []
        self.last_direction = [-180, -180, 180, 180]
        self.switch = False
        self.change = False
        self.r = 180
        self.theta = 60
        self.team(2)

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        # 根据需要自行选择函数实现形式
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def team(self, team_num):
        self.team_num = team_num
        # self.per_team_member_num = int(self.fighter_num / team_num)
        self.per_team_member_num = 2
        fighter = [x for x in range(self.fighter_num)]
        for t in range(team_num):
            self.team_list.append([i + 1 for i in fighter[self.per_team_member_num * t:self.per_team_member_num * (t + 1)]])

    def is_in_jugement(self, id_a, id_b):
        a_x = self.obs_dict['fighter_obs_list'][id_a - 1]['pos_x']
        a_y = self.obs_dict['fighter_obs_list'][id_a - 1]['pos_y']
        a_gamma = self.obs_dict['fighter_obs_list'][id_a - 1]['course']
        b_x = self.obs_dict['fighter_obs_list'][id_b - 1]['pos_x']
        b_y = self.obs_dict['fighter_obs_list'][id_b - 1]['pos_y']
        b_gamma = self.obs_dict['fighter_obs_list'][id_b - 1]['course']
        d = interface.get_distance(a_x, a_y, b_x, b_y)
        if d <= self.r * math.sin(self.theta * np.pi / 360) and abs(a_gamma - b_gamma) <= self.theta:
            return True
        else:
            return False

    def detector_rule(self):
        self.jugement_list = []
        fighter_list = []
        for i in range(self.fighter_num):
            if self.obs_dict['fighter_obs_list'][i]['alive']:
                fighter_list.append(self.obs_dict['fighter_obs_list'][i]['id'])
        # 遍历存活飞机，生成全局判定列表
        L = len(fighter_list)
        for a in range(L):
            for b in range(L):
                if b != a:
                    self.jugement_list.append([self.is_in_jugement(a+1, b+1), a+1, b+1])

    def distance(self, id_a, id_b):
        m_x = self.obs_dict['fighter_obs_list'][id_a-1]['pos_x']
        m_y = self.obs_dict['fighter_obs_list'][id_a-1]['pos_y']
        e_x = self.obs_dict['fighter_obs_list'][id_b-1]['pos_x']
        e_y = self.obs_dict['fighter_obs_list'][id_b-1]['pos_y']
        return interface.get_distance(e_x, e_y, m_x, m_y)

    def angle(self, id_a, id_b):
        m_x = self.obs_dict['fighter_obs_list'][id_a - 1]['pos_x']
        m_y = self.obs_dict['fighter_obs_list'][id_a - 1]['pos_y']
        e_x = self.obs_dict['fighter_obs_list'][id_b - 1]['pos_x']
        e_y = self.obs_dict['fighter_obs_list'][id_b - 1]['pos_y']
        return interface.angle_cal(e_x, e_y, m_x, m_y)

# 当前存在的问题：同一目标多架飞机发射导弹（任务分配问题）
    def get_action(self, obs_dict, step_cnt):
        self.obs_dict = obs_dict
        # obs_dict为状态，step_cnt为当前步数 从1开始
        if step_cnt == 1:
            self.team_list = []
            self.jugement_list = []
            self.open_radar_list = []
            self.crash_list = []
            self.already_list = []
            self.last_direction = [-180, -180, 180, 180]
            self.switch = False
            self.change = False
            self.team(2)

        detector_action = []
        fighter_action = []
        # 生成全局判定列表
        self.detector_rule()
        # 遍历每一个飞机
        # 伪过程

        for i in range(self.fighter_num):
            # print('position:', obs_dict['fighter_obs_list'][i])
            true_action = np.array([self.last_direction[i], 1, 0, 0], dtype=np.int32)
            # if i != 0:
            #     print(self.is_in_jugement(i+1, 1))
            team_id = int(i/self.per_team_member_num)
            # 判断长机
            if self.team_list[team_id][0] == i+1:
                if obs_dict['fighter_obs_list'][i]['alive'] is not True:
                    self.team_list[team_id][0] = -(i+1)
            # 判断僚机
            else:
                leader_id = self.team_list[team_id][0]
                if obs_dict['fighter_obs_list'][i]['alive'] is True and leader_id < 0:
                    n = i % self.per_team_member_num
                    self.team_list[team_id][n] = leader_id
                    self.team_list[team_id][0] = i+1
            # 遍历全局判定列表，判定本机是否开启雷达
            L = len(self.jugement_list)
            self.open_radar_list = self.jugement_list
            for a in range(L):
                b = a + 1
                while b < L:
                    # print('jugement', self.jugement_list)
                    if self.jugement_list[a][1] == self.jugement_list[b][1]:
                        self.open_radar_list[a] = [False, 0, 0]
                    b = b + 1
            print('done')
            # 初始化动作序列

            # 探测雷达决策（默认开启频点为1的雷达，无需变换频点）
            for radar in self.open_radar_list:
                if i+1 == radar[1]:
                    true_action[1] = 1
            # 干扰雷达决策（阻塞干扰）
            true_action[2] = 11
            # 搜索（固定50步为时间步长）
            # if step_cnt % 50 == 0:
            #     # 同一编队航向一致
            #     value = random.randint(1, 3)
            #     random.seed(team_id * 1000 + int(step_cnt/50))
            #     if obs_dict['fighter_obs_list'][i]['pos_x'] == 0:
            #         self.switch = True
            #     if obs_dict['fighter_obs_list'][i]['pos_x'] == 1000:
            #         self.switch = False
            #         if value == 1:
            #             true_action[0] = random.randint(0, 120)
            #         if value == 2:
            #             true_action[0] = random.randint(120, 240)
            #         if value == 3:
            #             true_action[0] = random.randint(240, 360)
            #
            #     if self.switch is False:
            #         # true_action[0] = random.randint(90, 270)
            #         if value == 1:
            #             true_action[0] = random.randint(0,120)
            #         if value == 2:
            #             true_action[0] = random.randint(120,240)
            #         if value == 3:
            #             true_action[0] = random.randint(240,360)
            #     else:
            #         # true_action[0] = random.randint(0, 90) * value + random.randint(270, 359) * (1 - value)
            #         if value == 1:
            #             true_action[0] = random.randint(0,120)
            #         if value == 2:
            #             true_action[0] = random.randint(120,240)
            #         if value == 3:
            #             true_action[0] = random.randint(240,360)
            #     self.last_direction[i] = true_action[0]



                # print('true_action_0', true_action[0])

                # 不进行判决区判定

            if obs_dict['fighter_obs_list'][i]['pos_x'] > 525 and self.change is False:
                if i+1 != self.team_list[team_id][0]:
                    # print('angler:',self.angle(self.team_list[team_id][0], i+1))
                    # print('reslut:',abs(math.sin(self.angle(self.team_list[team_id][0], i+1))))
                    # print('distance:',self.distance(i+1, self.team_list[team_id][0]))
                    if self.distance(i+1, self.team_list[team_id][0])* abs(math.sin(self.angle(self.team_list[team_id][0], i+1)*math.pi/360)) > 52.5:
                        true_action[0] = self.angle(self.team_list[team_id][0], i+1)
                        # true_action[0] = 90
                    else:
                        true_action[0] = 180
                # print('jinru')
                # true_action[0] = 90
                # self.last_direction[i] = true_action[0]
                # 解注释
                else:
                    if step_cnt < 45:
                        true_action[0] = 90
                    elif 45 < step_cnt < 90:
                        true_action[0] = 270
                    else:
                        true_action[0] = self.last_direction[i]
            else:
                self.change = True
                if step_cnt % 50 == 0:
                    # 同一编队航向一致
                    value = random.randint(1, 3)
                    random.seed(team_id * 1000 + int(step_cnt/50))
                    if obs_dict['fighter_obs_list'][i]['pos_x'] == 0:
                        self.switch = True
                    if obs_dict['fighter_obs_list'][i]['pos_x'] == 1000:
                        self.switch = False
                        if value == 1:
                            true_action[0] = random.randint(0, 120)
                        if value == 2:
                            true_action[0] = random.randint(120, 240)
                        if value == 3:
                            true_action[0] = random.randint(240, 360)

                    if self.switch is False:
                        # true_action[0] = random.randint(90, 270)
                        if value == 1:
                            true_action[0] = random.randint(0,120)
                        if value == 2:
                            true_action[0] = random.randint(120,240)
                        if value == 3:
                            true_action[0] = random.randint(240,360)
                    else:
                        # true_action[0] = random.randint(0, 90) * value + random.randint(270, 359) * (1 - value)
                        if value == 1:
                            true_action[0] = random.randint(0,120)
                        if value == 2:
                            true_action[0] = random.randint(120,240)
                        if value == 3:
                            true_action[0] = random.randint(240,360)
                    self.last_direction[i] = true_action[0]

            # 如果我方无人机阵亡，其他友机赶来支援
            if obs_dict['fighter_obs_list'][i]['alive'] is not True:
                if not obs_dict['fighter_obs_list'][i]['id'] in self.already_list:
                    self.crash_list.append(obs_dict['fighter_obs_list'][i]['id'])
                    self.already_list.append(obs_dict['fighter_obs_list'][i]['id'])
                    print('阵亡列表:', self.crash_list)

            if self.crash_list:
                true_action[0] = self.angle(self.crash_list[0], i+1)
                if i+1 != self.crash_list[0] and self.distance(i+1, self.crash_list[0]) <= 10:
                    self.crash_list.remove(self.crash_list[0])
                    # print('清除！')
            # 打击策略
            # 探测到目标
            # 被动雷达
            if obs_dict['fighter_obs_list'][i]['j_recv_list']:
                print(i,'被动雷达发现目标！',obs_dict['fighter_obs_list'][i]['j_recv_list'][0]['direction'])
                true_action[0] = obs_dict['fighter_obs_list'][i]['j_recv_list'][0]['direction']
            if obs_dict['fighter_obs_list'][i]['r_visible_list']:
                print('发现目标！', obs_dict['fighter_obs_list'][i]['r_visible_list'])
                print('打击列表：', obs_dict['fighter_obs_list'][i]['striking_dict_list'])
                # 密集型编队
                # 将发现飞机id加入到发现列表中
                # if [team_id, i] not in self.share_list:
                #     self.share_list.append([team_id, i])
                # 同一编队其余飞机跟随飞行


                d_list = []
                agl_list = []
                # 剩余远程导弹
                for enemy in obs_dict['fighter_obs_list'][i]['r_visible_list']:
                    # 遍历所有目标，计算距离,放入距离列表
                    print('inner')
                    m_x = obs_dict['fighter_obs_list'][i]['pos_x']
                    m_y = obs_dict['fighter_obs_list'][i]['pos_y']
                    e_x = enemy['pos_x']
                    e_y = enemy['pos_y']
                    d = interface.get_distance(m_x, m_y, e_x, e_y)
                    angle = interface.angle_cal(m_x, m_y, e_x, e_y)
                    # e_id = enemy['id']
                    # print('id:',m_id,e_id)
                    print('distance_enemy:', d)
                    d_list.append(d)
                    agl_list.append(angle)
                    # print('d_list',d_list)

                    print('missile_left:',obs_dict['fighter_obs_list'][i]['l_missile_left'])
                    if obs_dict['fighter_obs_list'][i]['l_missile_left']:
                        # 追踪
                        true_action[0] = angle
                        self.last_direction[i] = true_action[0]
                        print('追踪中...',true_action[0])
                        if d <= 120:
                            true_action[3] = enemy['id']
                            # true_action[3] = 0

                            print('打:', true_action[3],d)
                    # 无远程导弹，有中距导弹
                    elif obs_dict['fighter_obs_list'][i]['s_missile_left']:
                        # 选择距离最小的方向追击
                        true_action[0] = agl_list[d_list.index(min(d_list))]
                        self.last_direction[i] = true_action[0]
                        if min(d_list) <= 50:
                            true_action[3] = enemy['id'] + self.fighter_num
                    if d_list[0] <= 50:
                        true_action[3] = enemy['id'] + self.fighter_num
                        # true_action[3] = 0

            # 开始n步内进行密集编队,跟随长机
            # if step_cnt < 300:
            #     if i+1 != self.team_list[team_id][0]:
            #         true_action[0] = self.angle(self.team_list[team_id][0],i+1)
            #         # print('==',i+1,self.team_list[team_id][0])
            #     else:
            #         true_action[0] = self.last_direction[i]

            # if step_cnt < 500:
            #     for teamate in self.team_list[team_id]:
            #         if teamate > 0 and teamate != self.team_list[team_id][0]:
            #             true_action[0] = self.angle(teamate, i + 1)
            #         else:
            #             true_action[0] = self.last_direction[i]
            # 编队选择
            # 紧密编队
            # 自己的探测列表为空，队友的探测列表非空，跟随队友，反之分散编队
            if self.share_list:
                # print('share_list:', self.share_list)
                for item in self.share_list:
                    # print('list',self.share_list)
                    if team_id == item[0] and i != item[1]:
                        true_action[0] = self.angle(item[1]+1, i+1)
                        # print('方向改变')

            # if not obs_dict['fighter_obs_list'][i]['r_visible_list']:
            #     for teamate in self.team_list[team_id]:
            #         print('in', teamate)
            #         if teamate > 0 and teamate - 1 != i:
            #             if obs_dict['fighter_obs_list'][teamate-1]['r_visible_list']:
            #                 true_action[0] = self.angle(teamate, i + 1)
            #                 print('gai:', true_action[0])
            # else:
            #     true_action[0] = self.last_direction[i]

            # 规避策略
            if obs_dict['fighter_obs_list'][i]['l_missile_left'] == 0 and obs_dict['fighter_obs_list'][i]['s_missile_left'] == 0:
                # 关闭雷达
                true_action[1] = 0

            fighter_action.append(copy.deepcopy(true_action))
            # print('action', true_action)

        fighter_action = np.array(fighter_action)
        return detector_action, fighter_action
