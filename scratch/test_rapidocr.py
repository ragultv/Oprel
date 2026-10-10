from rapidocr_onnxruntime import RapidOCR
from oprel.server.db import get_db
import io, base64

conn = get_db()
c = conn.cursor()
c.execute('SELECT image_data FROM ocr_jobs ORDER BY created_at DESC LIMIT 1')
raw = c.fetchone()[0].split(',', 1)[1]
img_bytes = base64.b64decode(raw)

engine = RapidOCR()
result, elapse = engine(img_bytes)

print(f"Total lines detected: {len(result) if result else 0}")
for i, r in enumerate(result or []):
    bbox, text, score = r
    # Normalize bbox
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    safe_text = text.encode("ascii", "replace").decode("ascii")
    print(f"[{i:02d}] x=[{x_min:5.1f}, {x_max:5.1f}] y=[{y_min:5.1f}, {y_max:5.1f}] score={score:.2f} | {safe_text}")
