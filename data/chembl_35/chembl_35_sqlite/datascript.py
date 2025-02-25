import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("chembl_35.db")

# Query example: View available tables
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query, conn)
print(tables)

# Query specific data (e.g., compound properties)
data = pd.read_sql("SELECT * FROM compound_structures LIMIT 5;", conn)
print(data)
conn.close()

