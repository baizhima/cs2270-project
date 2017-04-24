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
	testing_data = open(db.testing_filepath)

	# TODO: update predict table after import all training data
	
	# feed testing data
	for record in testing_data:
		record_list = record.split(',')
		#Insert new data into raw_data_table
		# fill record with 'NAN'
		record = ','.join([v if len(v)>0 else '\'NAN\'' for v in record_list])
		db.insert_raw(conn, cur, record)
		# update predict_table
		db.update_predict(conn, cur, record_list[0], int(record_list[1]), random.uniform(-1.0, 1.0))
		# update y in raw_data_table using new record
		db.update_raw(conn, cur, record_list[0], record_list[1], record_list[-1])

	
	db.close(conn, cur)




if __name__ == '__main__':
  	main()