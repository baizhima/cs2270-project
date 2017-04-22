import psycopg2
import re
import pandas as pd
import time

#TODO: reorganize cur.execute

training_filepath = './data/preprocessed_training_data.csv'
testing_filepath = './data/preprocessed_testing_data.csv'
raw_data_table = 'raw_data'
predict_table = 'predict_data'

# BASIC OPERATION
def connect():
	try:
		conn = psycopg2.connect("dbname='postgres' user='Yahui' host='localhost' password='dbpass'")
	except:
		print "unable to connect"

	cur = conn.cursor()
	return conn, cur


def close(conn, cur):
	cur.close()
	conn.close()

def commit(conn):
	conn.commit()

# CREATE DATABASE
def import_training_data(conn, cur):

	csv_data = open(training_filepath)

	delete_raw_table_sql = 'DROP TABLE IF EXISTS %s;' % raw_data_table
	delete_predict_table_sql = 'DROP TABLE IF EXISTS %s;' % predict_table
	header = csv_data.readline().split(",")[:52] # get attribute name form header
	header_type = ["Integer"] * 2 + ["Float"] * 50
	attributes_sql = '(%s , PRIMARY KEY(id, timestamp));' % ", ".join([" ".join(list(a)) for a in zip(header, header_type)])
	attributes_list = ", ".join(header)
	create_table_sql = 'CREATE TABLE %s%s' % (raw_data_table, attributes_sql)
	preduct_attributes_sql = '(id Integer, timestamp Integer, predicted_value Float, PRIMARY KEY(id, timestamp));'
	create_predict_table_sql = 'CREATE TABLE %s%s' % (predict_table, preduct_attributes_sql)

	import_data(conn, cur, create_table_sql, delete_raw_table_sql, delete_predict_table_sql, create_predict_table_sql, csv_data)
	


def import_data(conn, cur, create_table_sql, delete_raw_table_sql, delete_predict_table_sql, create_predict_table_sql, csv_data):

	try:
		cur.execute(delete_raw_table_sql) #reset
		cur.execute(delete_predict_table_sql) #reset
		cur.execute(create_table_sql) #create table
		cur.execute(create_predict_table_sql) #create predicted table
		cur.copy_from(csv_data, raw_data_table, sep=',', null="") #read data
		cur.execute("select count(*) from %s;" % raw_data_table) #results
	except Exception as e:
		raise e
	else:
		print "Table %s created" % raw_data_table
		print "Table %s created" % predict_table
		print "Data Imported"
		print "Total Records: ", cur.fetchone()[0]
	finally:
		conn.commit()


# CQL SECTION

# Given cql, extract id and forecast timestamp
# CQL: select ? from ? where id = ? forecast ?
# case sensitive
def transform_cql(cql):
	m = re.search('id\s*=\s*(.+?)\s*forecast\s*\n*(.+?)\s*\n*;', cql)
	if m:
		return m.group(1, 2)
	else:
		return None

def execute_cql(conn, cur, cql):
	tran_attr = transform_cql(cql)
	if tran_attr:
		id = tran_attr[0]
		forecast_ts = tran_attr[1]
		# Get current highest timestamp and then forecast
		recent_time_sql = 'SELECT max(timestamp) from %s WHERE id = %s;' % (raw_data_table, id)
		try:
			cur.execute(recent_time_sql)
		except Exception as e:
			raise e
		finally:
			conn.commit()
		# execute cql
		try:
			ts = int(cur.fetchone()[0]) + int(forecast_ts)
		except Exception as e:
			print "None existed ID", e
			return

		forecast_sql = 'SELECT predicted_value FROM %s WHERE id=%s AND timestamp=%d;' % (predict_table, id, ts)
		
		try:
			cur.execute(forecast_sql)
		except Exception as e:
			raise e
		finally:
			conn.commit()

		if cur.rowcount is 0:
			print "Need to train model"
		else:
			return cur.fetchone()[0]
    # in case of normal query
	else:
		try:
			cur.execute(cql)
		except Exception as e:
			print e
		else:
			return cur.fetchone()[0]
		finally:
			conn.commit()


# Read table into df
# TODO: ORDER BY
def read_raw_to_dataframe(conn):
	df = pd.read_sql_query('SELECT * FROM %s;' % raw_data_table, conn)
	return df


# PREDICTION AND QUERY FEEDING

def insert_raw(conn, cur, record):
	insert_raw_sql = "INSERT INTO %s VALUES(%s);" % (raw_data_table, record)		
	try:
		cur.execute(insert_raw_sql)
	except Exception as e:
		raise e
	finally:
		conn.commit()

#ts: str
def update_raw(conn, cur, id, ts, y_value):
	update_raw_sql = "UPDATE %s SET y=%s WHERE id=%s AND timestamp=%s" % (raw_data_table, y_value, id, ts)
	try:
		cur.execute(update_raw_sql)
	except Exception as e:
		raise e
	finally:
		conn.commit()

# ts: int
def update_predict(conn, cur, id, ts, value):
	# update predicted_value in predict_table		
	# forecast 5 ts for each record
	for i in range(1, 6):
		# should change values to res from model
		insert_predict_sql = "INSERT INTO %s VALUES(%s, %d, %s)" % (predict_table, id, ts+i, value)
		update_predict_sql = "UPDATE %s SET predicted_value=%s WHERE id = %s AND timestamp = %d" % (predict_table, value, id, ts+i)
		try:
			cur.execute(insert_predict_sql)
		except:
			conn.rollback()
			time.sleep(0.0001)
			cur.execute(update_predict_sql)
		finally:
			conn.commit()



def udpate_predict_training(conn, cur):
	highest_ts_sql = 'SELECT DISTINCT id, max(timestamp) OVER (PARTITION BY id) AS curr_ts FROM %s;' % raw_data_table
	try:
		cur.execute(highest_ts_sql)
	except Exception as e:
		raise e
	# print highest_ts_sql
	# print cur.rowcount
	for res in cur.fetchall():
		# TODO: get value from model
		update_predict(conn, cur, res[0], int(res[1]), '1')
