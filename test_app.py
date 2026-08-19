import os
import threading
import time
import requests
import cv2
import numpy as np
from pathlib import Path

# Create dummy image and video
Path("test_assets").mkdir(exist_ok=True)
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite("test_assets/test.jpg", img)

out = cv2.VideoWriter("test_assets/test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 100))
for _ in range(10): out.write(img)
out.release()

BASE_URL = "http://localhost:5000"

def test_flow(port):
    base_url = f"http://localhost:{port}"
    print("1. Testing /health")
    r = requests.get(f"{base_url}/health")
    assert r.json()["status"] == "ok"
    
    print("2. Testing /models/status")
    r = requests.get(f"{base_url}/models/status")
    assert r.json()["ready"] is True
    
    print("3. Testing image inference")
    with open("test_assets/test.jpg", "rb") as f:
        r = requests.post(f"{base_url}/infer/image", files={"file": f})
    assert "result_url" in r.json()
    assert r.json()["n_anomalies"] == 0
    
    print("4. Testing video inference and polling")
    with open("test_assets/test.mp4", "rb") as f:
        r = requests.post(f"{base_url}/infer/video", files={"file": f})
    
    job_id = r.json()["job_id"]
    status = r.json()["status"]
    assert status == "processing"
    
    print(f"   Polling job {job_id}...")
    while True:
        r = requests.get(f"{base_url}/status/{job_id}")
        data = r.json()
        if data["status"] == "done":
            assert "result_url" in data
            print("   Video inference done!")
            break
        elif data["status"] == "error":
            print("   Video inference error:", data["error"])
            break
        time.sleep(1)
        
    print("All tests passed.")

if __name__ == "__main__":
    from app.app import app
    PORT = int(os.environ.get("PORT", 5002))
    server = threading.Thread(target=lambda: app.run(port=PORT, use_reloader=False))
    server.daemon = True
    server.start()
    time.sleep(2) # wait for server
    test_flow(PORT)
