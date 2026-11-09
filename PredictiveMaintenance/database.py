import sqlite3


conn = sqlite3.connect("predictions.db")

cursor = conn.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_quality_type TEXT,
    
    process_temp REAL,
    rpm INTEGER,
    torque REAL,
    tool_wear INTEGER,
    cause TEXT,
    prediction TEXT,
    confidence REAL
)
""")

conn.commit()
conn.close()

print("Database and history table created successfully!")