#!/usr/local/bin/python
import psycopg2
import csv

csv_data = open('./data/preprocessed_data.csv')

#connect to db
try:
	conn = psycopg2.connect("dbname='postgres' user='Yahui' host='localhost' password='dbpass'")
except:
	print "unable to connect"

cur = conn.cursor()
table_name = 'raw_data'

delete_table_sql = 'DROP TABLE IF EXISTS ' + table_name + ';'
header = csv_data.readline().split(",") #get attribute name form header
header_type = ["Integer"] * 2 + ["Float"] * 50
attributes_sql = '(default_id Serial, ' + ", ".join([" ".join(list(a)) for a in zip(header[1:], header_type)]) + ', PRIMARY KEY(id, timestamp));'
attributes_list = 'default_id, ' + ", ".join(header[1:])
create_table_sql = 'CREATE TABLE ' + table_name + attributes_sql

try:
	cur.execute(delete_table_sql) #reset
	cur.execute(create_table_sql) #create table
	cur.copy_from(csv_data, table_name, sep=',', null="") #read data
	cur.execute("select count(*) from " + table_name + ";") #results
except Exception as e:
	raise e
else:
	print "Table " + table_name + " created"
	print "Data Imported"
	print "Total Records: ", cur.fetchone()[0]

cur.close()
conn.commit()
conn.close()
