import db_helper as db

def create_db(cur, conn):
	table_name = 'raw_data'
	predict_table_name = 'predict_data'

	delete_table_sql = 'DROP TABLE IF EXISTS ' + table_name + ';'
	header = csv_data.readline().split(",") #get attribute name form header
	header_type = ["Integer"] * 2 + ["Float"] * 50
	attributes_sql = '(' + ", ".join([" ".join(list(a)) for a in zip(header, header_type)]) + ', PRIMARY KEY(id, timestamp));'
	attributes_list = ", ".join(header)
	create_table_sql = 'CREATE TABLE ' + table_name + attributes_sql
	preduct_attributes_sql = '(id Integer, timestamp Integer, predicted_value Float, PRIMARY KEY(id, timestamp));'
	create_predict_table_sql = 'CREATE TABLE ' + predict_table_name + preduct_attributes_sql


	csv_data = open('./data/preprocessed_data.csv')
	#connect to db
	# conn, cur = get_conn()

	create_table(cur, create_table_sql, delete_table_sql, create_predict_table_sql, csv_data, table_name)
	conn.commit()
	# close_conn(conn, cur)


def create_table(cur, create_table_sql, delete_table_sql, create_predict_table_sql, csv_data, table_name):

	try:
		cur.execute(delete_table_sql) #reset
		cur.execute(create_table_sql) #create table
		cur.execute(create_predict_table_sql) #create predicted table
		cur.copy_from(csv_data, table_name, sep=',', null="") #read data
		cur.execute("select count(*) from " + table_name + ";") #results
	except Exception as e:
		raise e
	else:
		print "Table " + table_name + " created"
		print "Table " + predict_table_name + " created"
		print "Data Imported"
		print "Total Records: ", cur.fetchone()[0]


