import json
import psycopg2
from psycopg2 import OperationalError, Error
from psycopg2 import sql
class PostgresDB:
    def __init__(self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            print("数据库连接成功")
        except OperationalError as e:
            print(f"无法连接到数据库，错误: {e}")
            self.connection = None

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")

    def execute_query(self, query):
        if self.connection is None:
            print("数据库未连接")
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Error as e:
            print(f"查询失败，错误: {e}")
            return None

    def execute_commit(self, query, params=None):
        if self.connection is None:
            print("数据库未连接")
            return None
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            cursor.close()
            print("操作成功提交")
        except Error as e:
            print(f"提交失败，错误: {e}")
            return None

    def create_table(self, table_name, columns):
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("""
                  CREATE TABLE IF NOT EXISTS {table_name} (
                      {columns}
                  );
              """).format(
                table_name=sql.Identifier(table_name),
                columns=sql.SQL(', ').join(map(sql.Identifier, columns))
            ))
            self.conn.commit()

    def insert(self, table_name, data):
        columns = data.keys()
        values = [data[column] for column in columns]
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("""
                  INSERT INTO {table_name} ({columns}) VALUES ({values});
              """).format(
                table_name=sql.Identifier(table_name),
                columns=sql.SQL(', ').join(map(sql.Identifier, columns)),
                values=sql.SQL(', ').join(sql.Placeholder() * len(values))
            ), values)
            self.conn.commit()

    def delete(self, table_name, condition):
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("""
                  DELETE FROM {table_name} WHERE {condition};
              """).format(
                table_name=sql.Identifier(table_name),
                condition=sql.SQL(condition)
            ))
            self.conn.commit()

    def update(self, table_name, data, condition):
        set_clause = ', '.join([f"{column} = %s" for column in data.keys()])
        values = list(data.values())
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("""
                  UPDATE {table_name} SET {set_clause} WHERE {condition};
              """).format(
                table_name=sql.Identifier(table_name),
                set_clause=sql.SQL(set_clause),
                condition=sql.SQL(condition)
            ), values)
            self.conn.commit()

    def execute_query_with_conditions(self, table_name, columns=None, conditions=None, like_match=False):
        if self.connection is None:
            print("数据库未连接")
            return None

        # 构建查询列
        columns_part = "*" if columns is None else ", ".join(columns)

        # 构建查询条件
        where_clauses = []
        query_params = []
        if conditions:
            for column, value in conditions.items():
                if like_match and isinstance(value, str) and '%' in value:
                    where_clauses.append(f"{column} LIKE %s")
                else:
                    where_clauses.append(f"{column} = %s")
                query_params.append(value)

        where_part = ""
        if where_clauses:
            where_part = "WHERE " + " AND ".join(where_clauses)

        # 构建完整的查询语句
        query = f"SELECT {columns_part} FROM {table_name} {where_part};"

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, query_params)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Error as e:
            print(f"查询失败，错误: {e}")
            return None

    def read_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def create_summary_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS llm.summary_data (
            id SERIAL PRIMARY KEY,
            chapter TEXT,
            section TEXT,
            summary TEXT
        );
        """
        self.execute_commit(query)

    def insert_summary_data(self, data):
        query = """
        INSERT INTO llm.summary_data (chapter, section, summary)
        VALUES (%s, %s, %s);
        """
        for item in data:
            self.execute_commit(query, (item['chapter'], item['section'], item['summary']))

# 使用示例
if __name__ == "__main__":
    db = PostgresDB('192.168.1.201', 'tjjg_db', 'tjjg', 'Sinops1234~')
    db.connect()

    # 创建新表
    db.create_summary_table()

    # 读取 JSON 文件
    json_data = db.read_json_file('data pre processing/数据库2表格/data.json')

    # 插入数据到新表
    db.insert_summary_data(json_data)

    print("数据已成功导入到数据库")

    db.close_connection()
