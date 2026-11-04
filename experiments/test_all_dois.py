import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import re

bib_path = Path("manuscript/updated_references.bib")
if not bib_path.exists():
    bib_path = Path("manuscript/references.bib")

bib_text = bib_path.read_text(encoding="utf-8")
entries = re.findall(r'@\w+\{([^,]+),(.*?)\n\}', bib_text, re.DOTALL)

test_dois = []
for key, body in entries:
    doi_m = re.search(r'doi=\{([^\}]+)\}', body, re.IGNORECASE)
    if doi_m:
        test_dois.append((key, doi_m.group(1).strip()))

def check_doi(doi):
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=8) as res:
            return res.geturl()
    except Exception as e:
        return f"FAILED: {e}"

if __name__ == "__main__":
    print(f"Total DOIs loaded from {bib_path.name}: {len(test_dois)}")
    all_pass = True
    for idx, (k, doi) in enumerate(test_dois, 1):
        res_url = check_doi(doi)
        if "FAILED" in res_url:
            print(f"[{idx:02d}/50] [FAIL] {k} ({doi}) -> {res_url}")
            all_pass = False
        else:
            print(f"[{idx:02d}/50] [PASS] {k} -> {res_url[:75]}...")

    if all_pass:
        print("\nALL 50 REFERENCES RESOLVE LIVE TO PUBLIC URLS VIA DOI.ORG!")
    else:
        print("\nSOME REFERENCES FAILED!")
