import json
import requests

# Get endpoint
with open('./endpoint_uri.txt') as f:
    scoring_uri = f.read()

# Get data
input_json = json.dumps(json.load(open('./test_payload.json')))

# Request prediction
response = requests.post(
    scoring_uri,
    input_json,
    headers={'Content-Type': 'application/json'}
)
prediction = json.loads(response.content)
print(prediction)
