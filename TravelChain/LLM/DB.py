import mysql.connector
from mysql import PooledMySQLConnection, MySQLConnectionAbstract

class DBConnector():
    conn: PooledMySQLConnection | MySQLConnectionAbstract
    
    def Connect(self, host:str, user:str, password:str, database:str):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        self.cursor = self.conn.cursor()
        
    # 테이블 생성
    '''
    self.cursor.execute("""
                        CREATE TABLE IF NOT EXISTS users(
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            name VARCHAR(255),
                            age INT
                        )
                        """)
    '''
    # 데이터 삽입
    """
    self.cursor.execute('''
                        INSERT INTO users (name, age) VALUES (%s, %s)
                        ''', ('Bob', 25))
    """
    
    # 커밋
    # self.conn.commit()
    
    # 데이터 조회
    # cursor.execute('SELECT * FROM users')
    # rows = cursor.fetchall()
    
    def Close(self):
        self.conn.close()
    
        
    