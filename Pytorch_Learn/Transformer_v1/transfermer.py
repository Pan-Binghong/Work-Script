#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   transfermer.py
@Time    :   2024/09/26 09:37:36
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''
import typing
import torch.nn as nn
import io
import sys
import os

file_path = os.path.abspath(__file__)

dir = os.path.dirname(file_path)

os.chdir(dir)

with io.StringIO() as buf, open("./transfermer.txt", "w+") as f:
    old_stdout = sys.stdout
    sys.stdout = buf
    
    help(nn.Transformer)

    sys.stdout = old_stdout

    f.write(buf.getvalue())