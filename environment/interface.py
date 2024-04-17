#! /usr/bin/env python
# -*- coding: utf-8 -*-
from pytransform import pyarmor_runtime
pyarmor_runtime()
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: interface.py
@time: 2018/4/10 0010 8:52
@desc: environment interface
"""

import importlib
from world.em_battle import BattleField
from world.replay import Replay
from world.load_map import Map
import world.position_calc as position_calc


class Environment:
    """
    Environment interface
    """
    # 环境实例初始化
    def __init__(self, map_path, side1_obs_ind, side2_obs_ind, max_step=5000, render=False, render_interval=1,
                 random_pos=False, log=False, random_seed=-1, external_render=False, side1_name='unknown',
                 side2_name='unknown'):
        """
        Environment initiation
        :param size_x: battlefield horizontal size. got from LoadMap.get_map_size
        :param size_y: battlefield vertical size. got from LoadMap.get_map_size
        :param side1_detector_list: side 1 detector configuration. got from LoadMap.get_unit_property_list
        :param side1_fighter_list: side 1 fighter configuration. got from LoadMap.get_unit_property_list
        :param side2_detector_list: side 2 detector configuration. got from LoadMap.get_unit_property_list
        :param side2_fighter_list: side 2 fighter configuration. got from LoadMap.get_unit_property_list
        :param max_step: max step，0：unlimited
        :param render: display enable control, True: enable display, False: disable display
        :param render_interval: display interval, skip how many steps to display a frame
        :param random_pos: start location initial method. False: side 1 on right, side2 on left. True: random position on top, bottom, right and left)
        :param log: log control，False：disable log，other value：the folder name of log.
        :param random_seed: random digit，-1：generate a new one，other value：use an exist random digit value
        """

        # load map
        self.map = Map(map_path)
        self.size_x, self.size_y = self.map.get_map_size()
        self.side1_detector_num, self.side1_fighter_num, self.side2_detector_num, self.side2_fighter_num = self.map.get_unit_num()
        # make env
        self.side1_detector_list, self.side1_fighter_list, self.side2_detector_list, self.side2_fighter_list = self.map.get_unit_property_list()
        self.env = BattleField(self.size_x, self.size_y, self.side1_detector_list, self.side1_fighter_list,
                               self.side2_detector_list, self.side2_fighter_list, max_step, render, render_interval,
                               random_pos, log, random_seed, external_render, side1_name=side1_name,
                               side2_name=side2_name)
        # import obs construct class
        if 'raw' == side1_obs_ind:
            self.side1_obs_path = 'raw'
        elif 'vector' == side1_obs_ind:
            self.side1_obs_path = 'vector'
        else:
            self.side1_obs_path = 'obs_construct.' + side1_obs_ind + '.construct'
            self.agent1_obs_module = importlib.import_module(self.side1_obs_path)
            self.agent1_obs = self.agent1_obs_module.ObsConstruct(self.size_x, self.size_y, self.side1_detector_num,
                                                                  self.side1_fighter_num)
        if 'raw' == side2_obs_ind:
            self.side2_obs_path = 'raw'
        elif 'vector' == side2_obs_ind:
            self.side2_obs_path = 'vector'
        else:
            self.side2_obs_path = 'obs_construct.' + side2_obs_ind + '.construct'
            self.agent2_obs_module = importlib.import_module(self.side2_obs_path)
            self.agent2_obs = self.agent2_obs_module.ObsConstruct(self.size_x, self.size_y, self.side2_detector_num,
                                                                  self.side2_fighter_num)

    # 判断对战是否结束
    def get_done(self):
        """
        Get done
        :return: done: True, False
        """
        return self.env.done

    def get_obs_vector(self):
        side1_obs_data, side2_obs_data = self.env.get_obs_vector()
        return side1_obs_data, side2_obs_data

    # 获得组合的观测信息
    def get_obs(self):
        """
        Get image-based observation
        :return: side1_obs
        :return: side2_obs
        """
        side1_obs_data = []
        side2_obs_data = []
        side1_obs_raw_dict, side2_obs_raw_dict = self.get_obs_raw()
        if 'vector' == self.side1_obs_path or 'vector' == self.side2_obs_path:
            side1_obs_data, side2_obs_data = self.get_obs_vector()
        # side2_detector_data_obs_list, side2_fighter_data_obs_list, side2_joint_data_obs_dict = self.env.get_obs_raw()
        if 'raw' == self.side1_obs_path:
            side1_obs = side1_obs_raw_dict
        elif 'vector' == self.side1_obs_path:
            side1_obs = side1_obs_data
        else:
            side1_obs = self.agent1_obs.obs_construct(side1_obs_raw_dict)

        if 'raw' == self.side2_obs_path:
            side2_obs = side2_obs_raw_dict
        elif 'vector' == self.side2_obs_path:
            side2_obs = side2_obs_data
        else:
            side2_obs = self.agent2_obs.obs_construct(side2_obs_raw_dict)

        return side1_obs, side2_obs

    def get_obs_raw(self):
        """
        Get raw data observation
        :return: side1_detector_data
        :return: side1_fighter_data
        :return: side2_detector_data
        :return: side2_fighter_data
        detector obs:{'id':id, 'alive': alive status, 'pos_x': horizontal coordinate, 'pos_y': vertical coordinate, 'course': course, 'r_iswork': radar enable status, 'r_fre_point': radar frequency point, 'r_visible_list': radar visible enemy}
        fighter obs:{'id':id, 'alive': alive status, 'pos_x': horizontal coordinate, 'pos_y': vertical coordinate, 'course': course, 'r_iswork': radar enable status, 'r_fre_point': radar frequency point, 'r_visible_list': radar visible enemy, 'j_iswork': jammer enable status, 'j_fre_point': jammer frequency point, 'j_recv_list': jammer received enemy, 'l_missile_left': long range missile left, 's_missile_left': short range missile left}
        """

        side1_obs_dict = {}
        side2_obs_dict = {}
        side1_detector_data_obs_list, side1_fighter_data_obs_list, side1_joint_data_obs_dict, \
        side2_detector_data_obs_list, side2_fighter_data_obs_list, side2_joint_data_obs_dict = self.env.get_obs_raw()
        side1_obs_dict.update({'detector_obs_list': side1_detector_data_obs_list})
        side1_obs_dict.update({'fighter_obs_list': side1_fighter_data_obs_list})
        side1_obs_dict.update({'joint_obs_dict': side1_joint_data_obs_dict})
        side2_obs_dict.update({'detector_obs_list': side2_detector_data_obs_list})
        side2_obs_dict.update({'fighter_obs_list': side2_fighter_data_obs_list})
        side2_obs_dict.update({'joint_obs_dict': side2_joint_data_obs_dict})
        return side1_obs_dict, side2_obs_dict

    # 获取个作战单元的存活状态
    def get_alive_status(self,side1_detector_obs_raw_list,side1_fighter_obs_raw_list,side2_detector_obs_raw_list,side2_fighter_obs_raw_list):
        return self.env.get_alive_status(side1_detector_obs_raw_list,side1_fighter_obs_raw_list,side2_detector_obs_raw_list,side2_fighter_obs_raw_list)

    # 获取回报
    def get_reward(self):
        """
        get reward
        :return:side1_detector：side1 detector reward，side1_fighter：side1 fighter reward，side1_round: side1 round reward, side2_detector：side2 detector reward，side2_fighter：side2 fighter reward，side2_round: side1 round reward
        """
        return self.env.get_reward()

    # 环境重置
    def reset(self):
        """
        Reset environment
        :return: none
        """
        self.env.reset()

    # 环境运行一步
    def step(self, side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action):
        """
        Run a step
        :param side1_detector_action: Numpy ndarray [detector_quantity, 2]
        :param side1_fighter_action: Numpy ndarray [fighter_quantity, 4]
        :param side2_detector_action: Numpy ndarray [detector_quantity, 2]
        :param side2_fighter_action: Numpy ndarray [fighter_quantity, 4]
        :return: True, run succeed, False, run Failed
        """
        return self.env.step(side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action)

    # 获取地图大小信息
    def get_map_size(self):
        """
        Get map size
        :return: size_x: horizontal size
        :return: size_y: vertical size
        """
        return self.map.get_map_size()

    # 获取各类别作战单元数目
    def get_unit_num(self):
        """
        Get unit number
        :return: side1_detector_num
        :return: side1_fighter_num
        :return: side2_detector_num
        :return: side2_fighter_num
        """
        return self.map.get_unit_num()

    # 获取各作战单元属性信息
    def get_unit_property_list(self):
        """
        Get unit config information
        :return: side1_detector_list, should be directly forward to Environment init interface
        :return: side1_fighter_list, should be directly forward to Environment init interface
        :return: side2_detector_list, should be directly forward to Environment init interface
        :return: side2_fighter_list, should be directly forward to Environment init interface
        """
        return self.map.get_unit_property_list()

    def set_surrender(self, side):
        '''
        surrender
        :param side: side1: 0, side2: 1
        :return:
        '''
        return self.env.set_surrender(side)


class PlayBack:
    """
    Replay
    """
    def __init__(self, log_name, external_render=False, display_delay_time=0):
        """
        Initial replay class
        :param log_name:
        :param display_delay_time:
        """
        self.rp = Replay(log_name, external_render, display_delay_time)

    def start(self):
        """
        Replay begin
        """
        self.rp.start()

# Utilities
def get_distance(a_x, a_y, b_x, b_y):
    """
    Get distance between two coordinates
    :param a_x: point a horizontal coordinate
    :param a_y: point a vertical coordinate
    :param b_x: point b horizontal coordinate
    :param b_y: point b vertical coordinate
    :return: distance value
    """
    return position_calc.get_distance(a_x, a_y, b_x, b_y)


def angle_cal(o_x, o_y, e_x, e_y):
    """
    Get a direction (angle) from a point to another point.
    :param o_x: starting point horizontal coordinate
    :param o_y: starting point vertical coordinate
    :param e_x: end point horizontal coordinate
    :param e_y: end point vertical coordinate
    :return: angle value
    """
    return position_calc.angle_cal(o_x, o_y, e_x, e_y)
