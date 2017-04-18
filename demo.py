#!/usr/local/bin/python
import sys
import db_helper as db
import create_postgre as postgre	


conn, cur = db.connect()
#create db
# postgre.create_db(conn, cur)

cql = sys.stdin.readline()
db.execute_cql(conn, cur, cql)

db.close(conn, cur)