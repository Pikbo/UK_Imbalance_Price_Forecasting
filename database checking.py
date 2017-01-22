import sqlite3
import os
import pandas as pd

# setting base directory
base_path = os.path.dirname(os.path.abspath(__file__))

# location of SQL database 
sql_path = os.path.join(base_path,'data','ELEXON DATA.sqlite')

# function that inputs the database path 
def get_data(db_path,table):
    conn = sqlite3.connect(db_path)
    data_DF = pd.read_sql(sql='SELECT * from ' + str(table),con = conn)
    conn.close()
    print('got sql data')
    return data_DF # returns a df with all table data

B1770 = get_data(db_path = sql_path,table='B1770')  
B1780 = get_data(db_path = sql_path,table='B1780')

reports_L = [B1770, B1780]

for report in reports_L:
    report_index = report['index']
    
    for i in range(0,len(report)-1):
            
        a = pd.to_datetime(report_index.iloc[i])
        b = pd.to_datetime(report_index.iloc[i+1])
        
        if a == b:
            print(i)
            print(a)
            print('identical index')
            
    for i in range(0,len(report)-1):
            
        a = pd.to_datetime(report_index.iloc[i])
        b = pd.to_datetime(report_index.iloc[i+1])
        
        if a - b != pd.to_timedelta("-1 days +23:30:00"):
            print(i)
            print(a)        
             