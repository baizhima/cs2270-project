import psycopg2
import re
import pandas as pd
import time
import random
import predict_helper as predict

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


	
# PREDICTION AND QUERY FEEDING

# Given id, return the df containing both curr_ts and prev_ts
# TODO: ORDER BY
def get_lag_data(conn, id):
	get_highest_sql = 'SELECT * FROM %s WHERE id=%s ORDER BY timestamp DESC LIMIT 1;' % (raw_data_table, id)
	df_curr = pd.read_sql_query(get_highest_sql, conn)
	get_prev_sql = 'SELECT * FROM %s WHERE id=%s AND timestamp=%d' % (raw_data_table, id, int(df_curr['timestamp'])-1)
	df_prev = pd.read_sql_query(get_prev_sql, conn).drop(['id', 'timestamp'], axis=1)
	
	if df_prev.empty:
		return pd.DataFrame()

	# concat
	lag_column_names = [c + '_lag1' for c in df_prev.columns]
	df_prev.columns = lag_column_names
	return pd.concat([df_curr, df_prev], axis=1)

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

def update_predict(conn, cur, id):
	# Get df
	df_lag_row = get_lag_data(conn, id)
	if df_lag_row.empty:
		return
	
	value = iter([random.uniform(-1.0, 1.0), predict.load_predict(df_lag_row)])
	ts = int(df_lag_row['timestamp'])
	# update predicted_value in predict_table		
	for i in [1, 10]:
		v = value.next()
		insert_predict_sql = "INSERT INTO %s VALUES(%s, %d, %f)" % (predict_table, id, ts+i, v)
		update_predict_sql = "UPDATE %s SET predicted_value=%f WHERE id = %s AND timestamp = %d" % (predict_table, v, id, ts+i)
		try:
			cur.execute(insert_predict_sql)
		except:
			conn.rollback()
			time.sleep(0.0001)
			cur.execute(update_predict_sql)
		finally:
			conn.commit()


def udpate_predict_training(conn, cur):
	select_id_sql = 'SELECT DISTINCT id FROM %s;' % raw_data_table
	try:
		cur.execute(select_id_sql)
	except Exception as e:
		raise e
	
	for res in cur.fetchall():
		update_predict(conn, cur, res[0])
		
