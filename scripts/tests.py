# Test this standalone first
import requests, json

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:3b",
    "prompt": "Return ONLY this JSON, nothing else: [{\"instruction\": \"What is sickle cell disease?\", \"response\": \"A genetic blood disorder.\"}]",
    "stream": False,
    "options": {"temperature": 0.3}
})

print("Status:", response.status_code)
print("Raw response:")
print(json.dumps(response.json(), indent=2))