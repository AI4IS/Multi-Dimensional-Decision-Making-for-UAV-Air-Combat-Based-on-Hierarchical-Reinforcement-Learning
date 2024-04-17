#! /usr/bin/env python
# -*- coding: utf-8 -*-
from pytransform import pyarmor_runtime
pyarmor_runtime()
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: agent.py
@time: 2018/3/13 0013 10:51
@desc: rule based agent
"""

from fix_rule.agent_core import Agent as ag


class Agent:
    def __init__(self):
        """
        Init this agent
        """
        self.agent_core = ag()

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.fighter_num = fighter_num
        return self.agent_core.set_map_info(size_x, size_y, detector_num, fighter_num)

    def get_action(self, obs_dict, step_cnt):
        """
        get actions
        :param detector_obs_list:
        :param fighter_obs_list:
        :param joint_obs_dict:
        :param step_cnt:
        :return:
        """
        # for i in range(self.fighter_num):
        #      if obs_dict['fighter_obs_list'][i]['j_recv_list']:
        #          print("观测值：",obs_dict['fighter_obs_list'][i])
        #          print('动作：',self.agent_core.get_action(obs_dict, step_cnt)[1][i])
        return self.agent_core.get_action(obs_dict, step_cnt)

    def get_obs_ind(self):
        return self.agent_core.get_obs_ind()
