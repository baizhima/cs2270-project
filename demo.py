#!/usr/local/bin/python
import sys
import db_helper as db

# Run with training data and test data cut by timestamp
# Accept forecast query and return predicted value
def main():
	conn, cur = db.connect()
	'''
	UNCOMMENT to import training data
	'''
	# db.import_training_data(conn, cur)
	# db.udpate_predict_training(conn, cur)

	while (True):
		line = raw_input('> ')
		if line == 'exit':
			db.close(conn, cur)
			sys.exit()
		print db.execute_cql(conn, cur, line)


if __name__ == '__main__':
  	main()