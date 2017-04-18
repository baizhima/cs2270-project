import psycopg2

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


def execute_cql(conn, cur, cql):


def visualize(conn, cur):

