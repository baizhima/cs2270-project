#!/usr/local/bin/python
import db_helper as db
import time
import random

def main():
	'''
	1. Feed db with testing data every 2 second
	2. Update db predict_data by invoking trained models
	3. Update the y value of db raw_data with new fed data
	'''
	conn, cur = db.connect()

	with open(db.testing_filepath , 'r') as f:
		testing_data = f.readlines()

	i = 0
	data_size = len(testing_data)
	while i <= data_size:
		line = raw_input('Please choose feeding size: > ')
		feeding_size = int(line)
		feed(conn, cur, testing_data[i:min(data_size, i + feeding_size)])
		i += feeding_size
		
	
	db.close(conn, cur)

def feed(conn, cur, records):
	# feed testing data
	for record in records:
		record_list = record.split(',')
		id = record_list[0]
		# Insert new data into raw_data_table
		# Fill record with 'NAN'
		record = ','.join([v if len(v)>0 else '\'NAN\'' for v in record_list])
		db.insert_raw(conn, cur, record)
		# update predict_table
		db.update_predict(conn, cur, id)
		# update y in raw_data_table using new record
		# db.update_raw(conn, cur, record_list[0], record_list[1], record_list[-1])


if __name__ == '__main__':
  	main()