import db_helper as db

def create_db(conn, cur):
	csv_data = open(db.filepath)

	delete_raw_table_sql = 'DROP TABLE IF EXISTS ' + db.raw_data_table + ';'
	delete_predict_table_sql = 'DROP TABLE IF EXISTS ' + db.predict_table + ';'
	header = csv_data.readline().split(",") #get attribute name form header
	header_type = ["Integer"] * 2 + ["Float"] * 51
	attributes_sql = '(' + ", ".join([" ".join(list(a)) for a in zip(header, header_type)]) + ', PRIMARY KEY(id, timestamp));'
	attributes_list = ", ".join(header)
	create_table_sql = 'CREATE TABLE ' + db.raw_data_table + attributes_sql
	preduct_attributes_sql = '(id Integer, timestamp Integer, predicted_value Float, PRIMARY KEY(id, timestamp));'
	create_predict_table_sql = 'CREATE TABLE ' + db.predict_table + preduct_attributes_sql


	create_table(cur, create_table_sql, delete_raw_table_sql, delete_predict_table_sql, create_predict_table_sql, csv_data)
	conn.commit()


def create_table(cur, create_table_sql, delete_raw_table_sql, delete_predict_table_sql, create_predict_table_sql, csv_data):

	try:
		cur.execute(delete_raw_table_sql) #reset
		cur.execute(delete_predict_table_sql) #reset
		cur.execute(create_table_sql) #create table
		cur.execute(create_predict_table_sql) #create predicted table
		cur.copy_from(csv_data, db.raw_data_table, sep=',', null="") #read data
		cur.execute("select count(*) from " + db.raw_data_table + ";") #results
	except Exception as e:
		raise e
	else:
		print "Table " + db.raw_data_table + " created"
		print "Table " + db.predict_table + " created"
		print "Data Imported"
		print "Total Records: ", cur.fetchone()[0]


