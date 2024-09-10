#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 13:46
# @Author  : Li Peng Ju
# @File    : bing_streamlit_text.py
# @Software: PyCharm

# 导入python相关库的包
import streamlit as st
import pandas as pd
import base64
import requests
import json
import re
import tempfile
import fitz  # PyMuPDF
import datetime
import os
import socket

# 设置页面标题和布局为宽屏
st.set_page_config(page_title="BMY AI Knowledge Bing", layout="wide")

# 使用markdown注入CSS样式来调整侧边栏宽度
st.markdown("""
        <style>
        /* 调整侧边栏的宽度 */
        [data-testid="stSidebar"] > div:first-child {
            width: 100%; /* 设置侧边栏宽度为100%，可以根据需要调整 */
        }
        </style>""", unsafe_allow_html=True)

# 读取外部js文件数据,并以utf-8的编码读取内容
global_js = open("text.js", encoding="utf-8").read()

# 使用streamlit的markdown 功能来嵌入 JavaScript 代码
st.markdown(f'''<script>{global_js}</script>''', unsafe_allow_html=True)

# 获取服务器ip函数
def get_local_ip():
    try:
        # 获取主机名
        hostname = socket.gethostname()
        # 将主机名转换为IP地址
        host_ip_info = socket.gethostbyname_ex(hostname)
        # 获取本地IP地址
        local_ip = host_ip_info[2][0]
        return local_ip

    # 捕捉异常信息
    except Exception as e:
        # 如果出错，返回错误信息
        return str(e)

# 自定义函数用于处理文档名称
def get_doc_urls(docs_list):
    # 初始化一个空列表来存储所有找到的URL
    all_urls = []
    # 遍历docs_list中的每个元素
    for doc in docs_list:
        # 确保元素是字符串
        if isinstance(doc, str):
            # 使用正则表达式匹配URL
            urls = re.findall(r'http[s]?://[^\s]+', doc)
            # 将找到的URL添加到列表中
            all_urls.extend(urls)

    # 返回URL地址
    return all_urls

# 将图片转换成Base64函数，以便嵌入到CSS中
def get_image_as_base64(background_image_path):
    # 使用with语句打开图片文件，确保文件在使用后会被正确关闭，'rb'模式表示以二进制形式读取文件
    with open(background_image_path, 'rb') as image_file:
        # 读取图片文件的全部内容，使用base64库将图片数据编码为Base64格式，
        # （注：.decode('utf-8')将Base64编码类型转换成utf-8编码的字符串）
        data = base64.b64encode(image_file.read()).decode('utf-8')

        # 返回转换后的Base64字符
    return data

# 初始excel表格数据（暂且未使用），保存Excel的函数，包括用户访问时间，用户访问IP,用户的问题，大模型的回答
def save_to_excel(question, answer, save_file_path='question_answer.xlsx'):
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 从Flask的request对象中获取用户的IP
    user_ip = get_local_ip()
    # 创建一个新的DataFrame来存储数据
    data = {
        "Timestamp": [timestamp],           # 获取用户访问时间
        "User_IP": [user_ip],               # 获取用户IP
        "user_Question": [question],        # 获取用户的问题
        "AI_Answer": [answer],              # 获取大模型的回答
    }
    # 将数据转成DataFrame格式的数据
    df = pd.DataFrame(data)

    try:
        # 如果文件存在，追加到现有文件；如果不存在，创建新文件
        if os.path.exists(save_file_path):
            # 读取现有的数据
            existing_data = pd.read_excel(save_file_path)
            # 合并数据
            combined_data = pd.concat([existing_data, df])
        else:
            # 文件已存在，使用ExcelWriteer以追加模式写入
            combined_data = df
            # 保存DataFrame到excel文件，不包含索引
        combined_data.to_excel(save_file_path, index=False)

        # 捕捉异常信息将错误信息给打印出来
    except Exception as e:
        print(f"保存excel时发生错误: {str(e)}")

# 保存新的Excel的函数，包括用户访问时间，用户访问IP,用户提出的问题，AI大模型的回答，用户的反馈,AI的回答时间,反馈信息的收集
def save_to_new_excel(question, answer, feedback, model_knowledge, query_time, answer_time, error_message, source,save_file_path='question_answer_feedback.xlsx'):
    # 从Flask的request对象中获取用户的IP
    user_ip = get_local_ip()
    # 创建一个新的DataFrame来存储数据
    data = {
        "query_time": [query_time],                                     # 问题查询的时间
        "User_IP": [user_ip],                                           # 用户的IP地址
        "User_Question": [question],                                    # 用户提出的问题
        "AI_Answer": [answer],                                          # AI大模型回答的结果
        "User_Feedback": [feedback if feedback else "无反馈"],           # 如果没有反馈则默认为"无反馈"
        'model_knowledge': [model_knowledge],                           # 选择的知识库
        "answer_time": [answer_time],                                   # AI回答的时间
        'error_message': [error_message],                               # 反馈回的信息收集
        "source": [source]                                              # 数据来源
    }
    # 将数据转成DataFrame格式的数据
    df = pd.DataFrame(data)

    # 检查文件是否存在，如果存在则读取并追加数据，否则直接保存
    try:
        # 检查文件是否存在
        if os.path.exists(save_file_path):
            # 读取现有的数据
            existing_data = pd.read_excel(save_file_path)
            # 合并数据
            combined_data = pd.concat([existing_data, df], ignore_index=True)
            # 将保存合并后的数据写入到excel文件中
            combined_data.to_excel(save_file_path, index=False)
        else:
            # 如果文件不存在，直接保存到数据到Excel文件
            df.to_excel(save_file_path, index=False)

    # 捕捉异常信息将错误信息打印出来
    except Exception as e:
        print(f"保存excel时发生错误: {str(e)}")

# 调用知识库接口数据函数
def get_response(user_input, selected_knowledge_base, conversation_history):
    # 知识库接口数据
    data = {
        "query": user_input,                                        # 用户输入的问题
        "knowledge_base_name": selected_knowledge_base,             # 当前选定的知识库名
        "top_k": 3,                                                 # 返回结果的数量
        "score_threshold": 0.6,                                     # 结果分数的阈值设置
        "history": conversation_history,                            # 当前对话的历史记录，用于上下文理解
        "stream": False,                                            # 是否以流式输出的结果返回，这里设置成False的意思是一次性返回所有结果
        "model_name": "Qwen-72B-Chat-Int4",                         # 当前使用的模型名称
        "temperature": 0.7,                                         # 随机性的控制参数，0.7表示相对较高的随机性
        "max_tokens": None,                                         # 最大返回的token数量限制，None表示没有限制
        "prompt_name": "default"                                    # 当前使用的提示词模板
    }

    try:
        # 调用知识库数据接口，发送post请求数据到服务器接口，json=data (注：表示将data字典数据作为JSON数据发送)
        response = requests.post('http://127.0.0.1:7861/chat/knowledge_base_chat', json=data)
        # 会在响应状态指示错误时抛出HTTPEroor异常
        response.raise_for_status()
        # 如果请求成功，返回相应对象
        return response

    # 捕捉请求过程中的异常信息，并且将错误信息打印出来
    except requests.RequestException as e:
        print(f"请求发生错误：{str(e)}")

# 处理知识库接口数据函数
def get_answer_list(response):
    # 检查相应是否存在
    if response:
        try:
            # 获取相应的文本内容
            response_text = response.text
            # 解析相应文本中的JSON数据
            answer_dict = json.loads(response_text.split("data:", 1)[1].strip())
            # 从解析后的JSON中获取"answer"字段
            answer = answer_dict.get('answer')
            # 从解析后的JSON中获取”docs“字段，如果没有则默认为空列表
            docs = answer_dict.get('docs', [])
            # 打印文档列表
            print('docs', docs)

            # 初始化用于存储页面编号的列表
            text_a = []
            # 遍历数据列表
            for text in docs:
                # 使用正则表达式查找文档中的页面编号
                matches = re.findall(r'page:\{(\d+)\}', text)
                # 将找到的页面编号添加到列表中去
                text_a = text_a + matches
                # 使用集合去除重复的页面编号
                text_a = list(set(text_a))
            # 打印页面编号列表
            print("页数列表", text_a)

            # 构建包含页面链接的HTML字符串
            links_html = ""
            # 遍历包含页面编号的列表
            for page_number in text_a:
                # 为每一个页面编号添加一个链接
                links_html += f"<a href='#page{page_number}'>第{page_number}页，</a>"
            # 构造最终的回答字符串，包含原始回答和页面链接
            final_answer = f"{answer}"
            # 构造包含原始回答和页面链接的HTML结构
            answer_list = f"{final_answer}该回答出自:{links_html} "
            # 返回最终构造的HTML字符串
            return answer_list

        # 捕捉JSON解析错误
        except json.JSONDecodeError as e:
            # 如果发生错误，返回空字符串
            return st.error(f'解析返回数据发生错误: {e}')

# 调用Bing搜索引擎对话接口数据
def query_bing_search(user_input):
    # 搜索引擎的接口数据
    bing_data = {
        "query": user_input,                                    # 用户提出的问题
        "search_engine_name": "bing",                           # 使用的搜索引擎名称
        "top_k": 3,                                             # 返回结果的数量限制
        "stream": False,                                        # 是否是流式输出结果
        "model_name": "Qwen-72B-Chat-Int4",                     # 使用的模型名称
        "temperature": 0.7,                                     # 控制结果的随机性
        "max_tokens": 0,                                        # 最大的返回的token数量，0表示无限制
        "prompt_name": "default",                               # 当前所使用的提示词模板
        "split_result": False                                   # 是否分割结果
    }
    try:
        # 向搜索引擎接口数据发起post请求，数据格式为json数据
        bing_search_response = requests.post('http://127.0.0.1:7861/chat/search_engine_chat', json=bing_data)
        # 检查请求是否成功，如果失败则抛出异常
        bing_search_response.raise_for_status()
        # 检查返回内容是否为空
        if bing_search_response.text.strip() == "":
            st.error('返回响应为空。')
            return None

        # 解析JSON数据
        response_text = bing_search_response.text
        # 如果相应内容是以'data'开头，则去掉改前缀
        if response_text.startswith('data:'):
            response_text = response_text[len('data:'):].strip()
        # 返回解析后的JSON数据
        return json.loads(response_text)

    # 捕获requests库抛出异常
    except requests.RequestException as e:
        # 使用st.error记录错误
        st.error(f'在调用bing_search接口时发生错误：{e}')
        return None

    # 捕捉JSON解析错误
    except json.JSONDecodeError as e:
        # 使用st.error记录错误
        st.error(f'解析返回数据时发生错误: {e}')

# 检查相应对象中是否有可用的答案函数
def not_satisfactory(response):
    # 检查相应对象是否为空
    if response:
        # 尝试从相应对象中获取'answer'键对应的值
        answer = response.get('answer')
        # 检查是否有找到的答案
        if answer:
            # 将答案转成字符串，这里实际就是直接赋值
            answer_list = f'{answer}'
            # 返回答案字符串
            return answer_list
        else:
            return '没有找到相关回答'
    else:
        # 如果相应对象为空，理论上这里不会执行，因为 if response已经检查了非空
        return '相应对象为空'

# 正则匹配联网出自网页的地址函数
def extract_urls(docs):
    # 初始化一个空列表，用于存储找到的网址
    urls = []
    # 遍历文档列表
    for doc in docs:
        # 使用正则匹配出联网查询出的所有网址 （注：https?: 匹配 'http' 后跟一个可选的 's'，\S+: 匹配一个或多个非空白字符）
        urls.extend(re.findall(r'(https?://\S+)', doc))
        # 返回包含所有网址的列表
    return urls

# 初始化session state变量
# 检查"user_input" 键是否存在于session_state中，如果不存在，则初始化为空字符串
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# 检查"answer" 键是否存在于session_state中，如果不存在，则初始化为空字符串
if 'answer' not in st.session_state:
    st.session_state['answer'] = ''

# 检查"feedback" 键是否存在于session_state中，如果不存在，则初始化为空字符串
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = ''

# 检查"model_knowledge" 键是否存在于session_state中，如果不存在，则初始化为空字符串
if 'model_knowledge' not in st.session_state:
    st.session_state['model_knowledge'] = ''

# 检查"answer_time" 键是否存在于session_state中，如果不存在，则初始化为空字符串
if 'answer_time' not in st.session_state:
    st.session_state['answer_time'] = ''

# 检查"s" 键是否存在于session_state中，如果不存在，则初始化为空字符串
if 'source' not in st.session_state:
    st.session_state['source'] = '知识库'

# 创建页面侧边栏，包括logo图标，下选择框
with st.sidebar:
    # 在侧边栏加载公司logo图标
    st.image(os.path.join('img', "baimoyun_logo.png"))

    # 定义一个包含不同知识库名称的列表
    knowledge_base_list = ['请选择知识库', 'BIM标准流程与内审机制库--(梳理)', 'BIM标准流程与内审机制库--(原生)', '佰模伝宣传册']
    # 使用streamlit的st.selectbox组件创建下拉选择框，让用户选择对应的知识库
    model_knowledge = st.selectbox('知识库名称列表', knowledge_base_list, index=0)
    # 创建一个字典，映射知识库的中文名称到他们各自英文名称或标识符
    knowledge_base_dict = {
        "BIM标准流程与内审机制库--(梳理)": "BIM standard process and internal audit mechanism",
        "BIM标准流程与内审机制库--(原生)": "BIM standard  and  Internal",
        "佰模伝宣传册": "Brochures",
    }
    # 根据用户的选择，从字典中获取对应的英文标识
    if model_knowledge != '请选择知识库':
        selected_knowledge_base = knowledge_base_dict[model_knowledge]
    else:
        selected_knowledge_base = None
        st.error('请选择您想查询的知识库')

# 主要布局分为两列
col1, col2 = st.columns(2)

# 存储对话历史
conversation_history = []

# 创建佰模伝AI知识库页面模块
with col2:
    # 设置页面的副标题为"佰模伝AI知识库"
    st.subheader('佰模伝AI知识库')
    # 创建一个文本输入框，用户可以在这里输入他们的问题，使用streamlit的key参数来标识这个输入框
    user_input = st.text_input('请输入你的问题', key='user_input')
    # 创建一个按钮，文本为“发送”,当用户点击这个按钮时，就会触发下面这个代码块
    send_button = st.button('发送')

    # 当用户点击发送按钮时，并且输入框存在非空空白字符时，执行以下代码。
    if send_button and user_input.strip():
        # 记录用户提问的时间
        st.session_state['query_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # 调用get_response函数，将用户的输入问题，选择知识库以及对家历史发送给后端
        response_knowledge = get_response(user_input, selected_knowledge_base, conversation_history)
        print('知识库数据是否请求成功', response_knowledge)

        # 将response存储到session_state中
        st.session_state['response'] = response_knowledge
        # 调用get_answer_list函数，从中获取答案列表
        answer_list = get_answer_list(response_knowledge)
        print('AI知识库回答内容', answer_list)

        # 检查是否是特定的回答，如果是，则调用query_bing_search函数来进行联网查询
        if '根据您所提问的问题，我没有在知识库中找到对应的答案' in str(answer_list):
            # 调用一个函数来执行Bing搜索，解析bing搜索的返回结果
            bing_search_response = query_bing_search(user_input)

            # 如果执行请求成功
            if bing_search_response:

                st.session_state['source'] = '联网查询'
                # streamlit页面上显示成功信息
                st.success('启动联网数据: 请求成功')
                # 调用相应对象中是否有可用的答案函数，对结果进行筛选或格式化
                answer_list = not_satisfactory(bing_search_response)
                st.session_state['answer'] = answer_list

                # 从搜索结果中提取出处的URLS列表
                docs = bing_search_response.get('docs', [])
                urls = extract_urls(docs)

                # 格式化URL列表，使其在Markdown中显示为链接
                url_list = '\n'.join([f"[联网查询网址 {i + 1}]({url})" for i, url in enumerate(urls)])

                # 将联网回答答案和出处网址显成绿色
                colored_answer = f'<span style="color:green;">{answer_list}</span>'

                # 使用Markdown格式显示联网查询的回答结果和出处网址,允许HTML标签
                st.markdown(f'**联网查询回答结果**: {colored_answer}\n\n**出处:**\n{url_list}', unsafe_allow_html=True)

                # 记录联网查询AI回答的时间，并格式化为字符串
                st.session_state['answer_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            else:
                # 如果搜索请求失败，反之打印报错信息
                st.error('数据请求失败')
        else:
            # 将获取到的答案保存到session_state中，以便后续使用
            st.session_state['answer'] = answer_list

            # 后台打印输出用户提出的问题，AI的回答以及答案来源的知识库
            print(f'用户输入的问题：{user_input} \n佰模伝AI知识库回答：'
                  f'{st.session_state["answer"]} \n出自哪个数据库：{model_knowledge}')

            # 使用markdown格式在streamlit页面上显示AI回答，允许HTML标签的使用
            st.markdown(f"**佰模伝AI知识库回答**: {answer_list}", unsafe_allow_html=True)

            # 记录AI知识库回答的时间，并格式化为字符串
            st.session_state['answer_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # 如果用户点击发送按钮，但输入为空，则显示st.error中的提示信息
    elif send_button:
        st.error('亲，请正确输入您的问题')

    # 新增用户反馈功能模块
    if st.session_state.get('answer'):
        # 用户反馈区   将输入框始终显示，而不是在点击按钮后显示
        feedback = st.text_input('请输入您的反馈', key='feedback')
        if st.button('提供反馈', key='feedback_button'):
            if feedback.strip():
                # 更新session state,并保存问题，答案和反馈
                save_to_new_excel(st.session_state['user_input'], st.session_state['answer'],
                                  feedback, model_knowledge,
                                  st.session_state['query_time'], st.session_state['answer_time'],
                                  st.session_state['source'],'')
                st.success('反馈已提交，谢谢！')
            else:
                st.error('请提供具体的反馈内容')

# 创建文档展示区页面模块
with col1:
    # 设置页面的副标题为“文档展示”
    st.subheader('文档展示')
    if 'response' in st.session_state and st.session_state['response'] and send_button and user_input.strip():
        try:
            # 尝试从response对象中获取文本数据
            response = st.session_state['response']
            response_text = response.text
            # 将相应文本解析为json，并获取“answer”键中对应的值
            answer_dict = json.loads(response_text.split("data:", 1)[1].strip())
            answer = answer_dict.get('answer')
            # 获取文档列表，这些文档可能包含在相应中
            docs = answer_dict.get('docs', [])  # 尝试获取的URL

            # 使用自定义函数处理文档中的URL
            doc_urls = get_doc_urls(docs)
            # 如果提取到了URL，尝试进一步处理
            if doc_urls:
                # 假设提取到的URL列表的第一个URL是我们需要的
                url_string = doc_urls[0]
                # 如果URL以多余的字符结尾，则去除掉他们
                url_string = url_string[:-1]
                # 将处理后的URL赋值给doc_urls变量
                doc_urls = url_string

            # 如果列表为空，打印提示信息
            else:
                print("列表为空，没有URL可以提取。")

            # 检查是否有URL，并且是有效的PDF链接
            if doc_urls.endswith('.pdf'):
                # 从URL下载PDF文件
                response = requests.get(doc_urls)

                # 如果下载成功（HTTP状态为200），即为请求成功
                if response.status_code == 200:
                    # 创建一个临时文件来保存PDF数据   注：delete=False表示在文件关闭后不要自动删除文件，这样我们之后就可以访问它，
                                                  # suffix = '.pdf'表示临时文件的后缀名为.pdf，及创建一个PDF文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
                        # 将HTTP相应内容写入到临时文件，response.content包含了请求得到的PDF数据
                        tmpfile.write(response.content)
                        # 关闭临时文件，注意由于使用的with语句，文件会在with语句块结束时自动关闭，
                        # 将临时文件的路劲保存到变量tmpfile_path中，这样我们可以在后续代码中引用它
                        tmpfile_path = tmpfile.name
                    try:
                        # 使用PyMuPDF读取PDF文件
                        pdf_document = fitz.open(tmpfile_path)
                    except Exception as e:
                        # 如果打开PDF失败，显示错误信息
                        st.error("无法打开PDF文件: " + str(e))
                    else:
                        # 如果PDF文件成功打开，记录PDF文档对象和总页数到session_state
                        st.session_state['pdf_document'] = pdf_document
                        # 记录pdf文档的总页数
                        st.session_state['pdf_pages'] = len(pdf_document)

                        # 定义CSS样式,用于改变图片在鼠标悬停时的显示效果，
                        css_style = """
                                       <style>
                                       .img-hover:hover {
                                         transform: scale(1.2); /* 放大到原来的120% */
                                         transition: transform 0.5s ease; /* 平滑过渡效果 */
                                       }
                                       </style>
                                       """

                        # 添加CSS样式
                        st.markdown(css_style, unsafe_allow_html=True)

                        # 创建一个数组
                        list = []
                        # 初始化一个空白字符串,用于存储所有页面的HTML标记
                        imgListStr = ''

                        # 遍历所有页面并显示所有页面
                        for page_number in range(st.session_state['pdf_pages']):
                            # 从PDF文档中获取指定页面，（page_number -1） 是因为rang函数生成的是从0开始的索引
                            page = pdf_document[page_number - 1]
                            # 使用PyMuPDF的get_pixmap()方法将页面渲染页面为图片对象
                            pix = page.get_pixmap()
                            # 将图片转换为PNG格式的字节
                            img_bytes = pix.tobytes("png")
                            # 将图片字节转换成Base64编码的字符串
                            img_base64 = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')
                            # 为每一个页面生成的HTNL图像标签，并添加到imgListStr字符串，注：设置图片宽度为560像素，并添加一个id属性，id的值为“page‘加上页面的编号
                            imgListStr += f"<div id='page{page_number + 1}' style='text-align: center;'><img class='img-hover' src='{img_base64}' width='560'></div>"

                        # 遍历之前初始化的数组并生成图像标签
                        for i in list:
                            imgListStr += f"<img src='{i}' width='560'>"

                        # 将所有页面的HTML标记包装到一个可滚动的div中，并显示，注：设置div的样式居中，高度为1000像素，且垂直方向上可滚动，允许HTML标签通过设置unsafe_allow_html参数为True
                        st.markdown(
                            f"""<div style='text-align: center;height:1000px;overflow-y: scroll'>{imgListStr}</div>""",
                            unsafe_allow_html=True)

                        # 关闭PDF文档
                        pdf_document.close()
                        # 删除临时文件
                        os.remove(tmpfile_path)
                else:
                    # 如果下载PDF失败，显示错误信息
                    st.error("未找到有效的PDF链接。")

            else:
                # 文档展示区默认加载图片的样式
                st.image(os.path.join('img', "baimoyun.png"))

        except json.JSONDecodeError as e:
            # 如果相应文本无法解析为JSON，保存错误信息并显示错误提示
            save_to_new_excel(st.session_state['user_input'], st.session_state['answer'], '', '',
                              st.session_state['query_time'], st.session_state['answer_time'], str(e))

            st.error('无法解析响应为JSON格式。错误信息：' + str(e))

    # 如果用户没有点击发送按钮或者输入框为空，则显示默认的图片
    else:
        # 文档展示区默认加载图片的样式
        st.image(os.path.join('img', "baimoyun.png"))

# 清理会话状态
st.session_state.pop('pdf_document', None)
st.session_state.pop('pdf_pages', None)