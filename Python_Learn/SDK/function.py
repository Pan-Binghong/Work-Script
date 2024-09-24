#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   function.py
@Time    :   2024/09/24 14:28:46
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   
'''

from typing import Union

def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    return a + b


def muti(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    return a * b

# 示例调用
result = add(5, 3.5)
print(result)  # 输出: 8.5

print(muti(1, 2.5))


if __name__ == '__main__':
    pass

