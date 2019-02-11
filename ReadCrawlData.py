from urllib2 import urlopen
# from urllib.request import urlopen
import json

# Get the dataset
url = 'http://localhost:8085/trustchain/recent'
response = urlopen(url)

# Convert bytes to string type and string type to dict
string = response.read().decode('utf-8')
json_obj = json.loads(string)

blocks = json_obj['blocks']

print(json_obj['blocks'])  # prints the string with 'source_name' key

for b in blocks:
    print(b['hash'])
