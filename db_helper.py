import psycopg2
import re

filepath = './data/preprocessed_data.csv'
raw_data_table = 'raw_data'
predict_table = 'predict_data'

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

# Given cql, extract id and forecast timestamp
# CQL: Select ? From ? Where id = ? Forecast ?
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
		recent_time_sql = 'SELECT max(timestamp) from ' + raw_data_table + ' WHERE id = ' + id + ';'
		try:
			cur.execute(recent_time_sql)
		except Exception as e:
			raise e
		# execute cql
		try:
			ts = int(cur.fetchone()[0]) + int(forecast_ts)
		except Exception as e:
			raise "None existed ID"

		forecast_sql = 'SELECT predicted_value FROM ' + predict_table + ' WHERE id = ' + id + ' AND timestamp = ' + str(ts) + ';'
		
		try:
			cur.execute(forecast_sql)
		except Exception as e:
			raise e

		if cur.rowcount is 0:
			print "Need to train model"
		else:
			return cur.fetchone()[0]

	else:
		try:
			cur.execute(cql)
		except Exception as e:
			raise e
		else:
			print cur.fetchone()[0]

# def visualize(conn, cur):

