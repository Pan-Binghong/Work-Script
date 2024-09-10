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
        df_result_computer = find_computer_names_and_counts(df_warning['Description'])
        
        # 清除之前的表格内容（如果有的话）
        for widget in frame.winfo_children():
            widget.destroy()
        
        notebook = ttk.Notebook(frame)
        notebook.pack(fill='both', expand=True)

        # 第一页：服务名称统计
        service_frame = ttk.Frame(notebook)
        notebook.add(service_frame, text='Service Name')
        show_dataframe_in_gui(df_result, service_frame)

        # 第二页：计算机名称统计
        computer_frame = ttk.Frame(notebook)
        notebook.add(computer_frame, text='Computer Name')
        show_dataframe_in_gui(df_result_computer, computer_frame)

def find_computer_names_and_counts(descriptions):  
    # 创建一个字典来记录每个计算机名称的linkdown次数  
    computer_counts = {}  
    
    pattern2 = re.compile(r'\(Computer:([^)]+)\)')  
    for desc in descriptions:  
        match2 = pattern2.search(desc)  
        if match2:  
            computer_name = match2.group(1)
            computer_counts[computer_name] = computer_counts.get(computer_name, 0) + 1 

    df_result_computer = pd.DataFrame.from_dict(computer_counts, orient='index', columns=['Count of Link is Down']).reset_index()  
    df_result_computer.columns = ['Computer Name', 'Count of Link is Down'] 
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

