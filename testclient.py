#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: testClient.py
@time: 2020/7/1 0001 14:21
@desc:
"""

import socket
import os
import time


def connect(ip, port):
    msg = 'aaa'
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((ip, port))
            break
        except:
            time.sleep(5)
            continue
    if os.path.exists('networklog.txt'):
        os.remove('networklog.txt')
    while True:
        f = open('networklog.txt', 'a')
        s.send(msg.encode('utf-8'))
        # print('req msg send!')
        data = s.recv(8192)
        f.write(data.decode('utf-8') + '\n')
        f.close()
        # print(data.decode('utf-8'))


if __name__ == "__main__":
    connect('127.0.0.1', 9999)

