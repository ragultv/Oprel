from oprel.server.db import get_db
import json

conn = get_db()
c = conn.cursor()
c.execute("SELECT result_json FROM ocr_jobs ORDER BY created_at DESC LIMIT 1")
row = c.fetchone()
if row:
    items = json.loads(row[0])
    print(f"Total items: {len(items)}")
    for i, item in enumerate(items):
        cat = item.get("category", "N/A")
        text = item.get("text", "")[:40].replace("\n", " ")
        norm = item.get("bbox_norm", {})
        print(f"[{i:02d}] ({cat:12s}) {text:40s} -> L:{norm.get('left'):.3f} T:{norm.get('top'):.3f} W:{norm.get('width'):.3f} H:{norm.get('height'):.3f}")
