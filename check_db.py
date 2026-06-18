import sqlite3
import os
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

db_path = r"d:\Ragul\oprel-SDK\oprel\server\data\groups.db"
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    # try another path
    db_path = r"d:\Ragul\oprel-SDK\oprel\server\groups.db"
    
if not os.path.exists(db_path):
    # Try getting the path from oprel.server.db
    import sys
    sys.path.append(r"d:\Ragul\oprel-SDK")
    from oprel.server import db
    db_path = db.DB_PATH

print(f"Using DB: {db_path}")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("--- RECENT MESSAGES ---")
cursor.execute("SELECT * FROM group_messages ORDER BY created_at DESC LIMIT 10")
for row in cursor.fetchall():
    print(dict(row))

print("\n--- RECENT ROUNDS ---")
cursor.execute("SELECT * FROM group_rounds ORDER BY created_at DESC LIMIT 5")
for row in cursor.fetchall():
    print(dict(row))

print("\n--- GROUP MEMBERS ---")
cursor.execute("SELECT id, group_id, display_name FROM group_members WHERE group_id = 'grp_745262d8dfaf'")
for row in cursor.fetchall():
    print(dict(row))
