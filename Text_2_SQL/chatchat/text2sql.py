#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/21 16:13
# @Author  : Li Peng Ju
# @File    : text2sql.py
# @Software: PyCharm
import re
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts.prompt import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from chatchat.server.pydantic_v1 import Field
from chatchat.server.utils import get_tool_config

from .tools_registry import BaseToolOutput, regist_tool

READ_ONLY_PROMPT_TEMPLATE = """You are an Oracle expert. The database is currently in read-only mode. 
Given an input question, determine if the related SQL can be executed in read-only mode.
If the SQL can be executed normally, return Answer: 'SQL can be executed normally'.
If the SQL cannot be executed normally, return Answer: 'SQL cannot be executed normally'.
Use the following format:

Answer: Final answer here

Question: {query}
"""


def intercept_sql(conn, cursor, statement, parameters, context, executemany):
    write_operations = (
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "truncate",
        "rename",
    )
    if any(statement.strip().lower().startswith(op) for op in write_operations):
        raise OperationalError(
            "Database is read-only. Write operations are not allowed.",
            params=None,
            orig=None,
        )


import re


def query_database(query: str, config: dict):
    model_name = config["model_name"]
    top_k = config["top_k"]
    return_intermediate_steps = config["return_intermediate_steps"]
    sqlalchemy_connect_str = config["sqlalchemy_connect_str"]
    db = SQLDatabase.from_uri(sqlalchemy_connect_str)

    from chatchat.server.utils import get_ChatOpenAI

    llm = get_ChatOpenAI(
        model_name=model_name,
        temperature=0.1,
        streaming=True,
        local_wrap=True,
        verbose=True,
    )

    table_names = config["table_names"]
    result = None

    if len(table_names) > 0:
        db_chain = SQLDatabaseChain.from_llm(
            llm,
            db,
            verbose=True,
            top_k=top_k,
            return_intermediate_steps=return_intermediate_steps,
        )
        result = db_chain.invoke({"query": query, "table_names_to_use": table_names})

        # 检查 result['intermediate_steps'] 是否为列表，并包含字典
        if isinstance(result['intermediate_steps'], list):
            last_step = result['intermediate_steps'][-1]
            if isinstance(last_step, dict) and 'sql_cmd' in last_step:
                # 提取生成的 SQL 语句
                sql_cmd = last_step['sql_cmd']
            elif isinstance(last_step, str):
                # 如果是字符串，尝试解析并提取SQL语句
                print(f"警告：最后一步是字符串，尝试解析: {last_step}")
                sql_cmd = last_step.split("SQLQuery:")[-1].strip()
                # 移除可能出现的反引号和其他多余字符
                sql_cmd = sql_cmd.replace('```sql', '').replace('```', '').strip()
            else:
                raise TypeError(
                    f"Unexpected format in last intermediate step. Expected a dictionary with 'sql_cmd' key or a parsable string, got: {last_step}")
        else:
            raise TypeError(
                f"Unexpected format in intermediate_steps. Expected a list, got: {type(result['intermediate_steps'])}")

        # 打印原始 SQL 语句
        print("原始 SQL 语句:", sql_cmd)

        # 修正 Oracle 不支持的 FETCH NEXT 语法
        if re.search(r'FETCH (FIRST|NEXT) \d+ ROWS ONLY', sql_cmd, re.IGNORECASE):
            match = re.search(r'FETCH (FIRST|NEXT) (\d+) ROWS ONLY', sql_cmd, re.IGNORECASE)
            if match:
                limit = match.group(2)
                # 将 SQL 查询封装到一个子查询中，然后用 ROWNUM 限制行数
                sql_cmd = re.sub(r'SELECT', f'SELECT * FROM (SELECT', sql_cmd, flags=re.IGNORECASE)
                sql_cmd = re.sub(r'FETCH (FIRST|NEXT) \d+ ROWS ONLY', f') WHERE ROWNUM <= {limit}', sql_cmd,
                                 flags=re.IGNORECASE)
                print("已修正 SQL 语法:", sql_cmd)
        else:
            print("无需修正的 SQL 语法:", sql_cmd)

        try:
            # 执行调整后的 SQL 查询
            final_result = db.run(sql_cmd)
            print("执行的最终 SQL 语句:", sql_cmd)

        except Exception as e:
            print(f"执行 SQL 查询时出错: {e}")
            raise e

        context = f"""查询结果: {final_result}\n\n"""

        return context


@regist_tool(title="数据库对话")
def text2sql(
        query: str = Field(
            description="No need for SQL statements, just input the natural language that you want to chat with database"
        ),
):
    """
    This function interacts with the database using natural language queries.
    It converts the natural language input into SQL, executes it, and returns the result.
    """
    tool_config = get_tool_config("text2sql")
    return BaseToolOutput(query_database(query=query, config=tool_config))