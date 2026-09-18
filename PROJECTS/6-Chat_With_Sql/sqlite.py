import sqlite3

connection=sqlite3.connect("student.db")
cursor=connection.cursor()
table_info="""
CREATE TABLE IF NOT EXISTS STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

cursor.execute("Insert Into STUDENT values('Krish','Data Science','A',90)")

print("The inserted records are:")
data=cursor.execute("Select * from STUDENT")
for row in data:
    print(row)

connection.commit()
connection.close()