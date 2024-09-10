#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   count_dict.py
@Time    :   2024/05/11 09:18:04
@Author  :   pan binghong 
@Email   :   19909442097@163.com
@description   :   通过日志统计服务器对应的link down次数
'''

import pandas as pd
import json
import re
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

def load_csv_and_process():
    file_path = filedialog.askopenfilename(filetypes=[('CSV文件', '*.csv')])

    if file_path:
        df = pd.read_csv(file_path)
        df_warning = df[(df['Severity'] == 'Warning') & (df['Event Name'] == 'Link is Down')]
        pattern1 = re.compile(r'Source\s+([a-fA-F0-9]+)')
        mac_list = []
        for source in df_warning['Source']:
            match1 = pattern1.search(source)
            if match1:
                mac_list.append(match1.group(1))
        df_result = find_service_names_and_counts(mac_list, mac_dict)
        show_dataframe_in_gui(df_result, frame)

        pattern2 = re.compile(r'\(Computer:([^)]+)\)')  
        computer_list = []  # 用于存储所有匹配的计算机名  
        for source in df_warning['Description']:  
            match2 = pattern2.search(source)  
            if match2:  
                computer_list.append(match2.group(1))    
        df_result_computer = find_computer_names_and_counts(computer_list)
        show_dataframe_in_gui(df_result_computer, frame)

def find_computer_names_and_counts(computer_list):  
    # 创建一个字典来记录每个服务名称的linkdown次数  
    computer_counts = {}  
    
    # 遍历mac_list中的每个MAC地址  
    for com in computer_list:  
        computer_counts[com] = computer_counts.get(com, 0) + 1 
    df_result_computer = pd.DataFrame.from_dict(computer_counts, orient='index', columns=['Count of Link is Down']).reset_index()  
    df_result_computer.columns = ['Computer Name', 'Count of Link is Down'] 
    print(df_result_computer) 
    return df_result_computer

def find_service_names_and_counts(mac_list, mac_dict):  
    # 创建一个字典来记录每个服务名称的linkdown次数  
    service_counts = {}  
      
    # 遍历mac_list中的每个MAC地址  
    for mac in mac_list:  
        # 查找MAC地址在哪个服务名称的列表中  
        for service_name, macs in mac_dict.items():  
            if mac in macs:  
                # 如果找到，增加该服务名称的计数  
                service_counts[service_name] = service_counts.get(service_name, 0) + 1  
                break  # 跳出内部循环，因为每个MAC地址只应对应一个服务名称  
    df_result = pd.DataFrame.from_dict(service_counts, orient='index', columns=['Count of Link is Down']).reset_index()  
    df_result.columns = ['Service Name', 'Count of Link is Down']  
    return df_result  

def show_dataframe_in_gui(df, frame_widget):  
    # 清除之前的表格内容（如果有的话）  
    for widget in frame_widget.winfo_children():  
        widget.destroy()  
  
    # 注意这里：将df.columns转换为列表  
    column_names = list(df.columns)  
  
    # 创建一个Treeview组件来显示数据  
    tree = ttk.Treeview(frame_widget, columns=column_names, show='headings')  
  
    # 设置列标题  
    for col in column_names:  
        tree.heading(col, text=col)  
  
    # 插入数据到Treeview  
    for index, row in df.iterrows():  
        tree.insert('', 'end', values=row.tolist())  
  
    # 将Treeview添加到frame中（如果它之前没有被添加到frame中，这一步是必要的）  
    tree.pack(fill='both', expand=True)

def main():
    root = tk.Tk()
    root.geometry('400x300')
    root.title('Link Down Count')

    # 创建一个Frame来放置Treeview  
    global frame
    frame = tk.Frame(root)  
    frame.pack(fill='both', expand=True) 

    load_button = tk.Button(root, text='选择CSV文件', command=load_csv_and_process)
    load_button.pack()


    with open('mac_dict.json', 'r') as f:
        global mac_dict
        mac_dict = json.load(f)

    root.mainloop()

if __name__ == '__main__':
    main()