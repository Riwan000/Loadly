import requests
import re
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

r = requests.get('https://grid-india.in/assets/index-BYJJycyS.js', headers=headers, verify=False, timeout=15)
js = r.text
print('JS bundle size:', len(js))

# Search for API URLs
matches = re.findall(r'[\'"`](https?://[^\'"`]+|/[^\'"`]+)[\'"`]', js)
for m in set(matches):
    if any(k in m.lower() for k in ['psp', 'report', 'api', 'upload', 'document']):
        print('Endpoint:', m)
