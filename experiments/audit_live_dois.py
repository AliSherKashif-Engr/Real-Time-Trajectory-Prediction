import re
import urllib.request
import urllib.parse
import json
import time
import difflib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def test_url(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.geturl()
    except Exception as e:
        return None

def verify_entry(key, title, authors, year):
    clean_title = re.sub(r'[{\}]', '', title).strip()
    
    # 1. Search OpenAlex
    url = "https://api.openalex.org/works?search=" + urllib.parse.quote(clean_title) + "&per-page=3"
    req = urllib.request.Request(url, headers={'User-Agent': 'BibAuditor/1.0 (mailto:academic-audit@superior.edu.pk)'})
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            results = data.get('results', [])
            for it in results:
                found_title = it.get('title') or ""
                sim = difflib.SequenceMatcher(None, clean_title.lower(), found_title.lower()).ratio()
                doi = it.get('doi')
                landing_url = it.get('primary_location', {}).get('landing_page_url') or doi
                
                # Check if high similarity
                if sim >= 0.70 or clean_title.lower() in found_title.lower() or found_title.lower() in clean_title.lower():
                    # Test if DOI/URL resolves live
                    live_url = test_url(doi) if doi else test_url(landing_url)
                    if live_url:
                        return {
                            "status": "LIVE_VERIFIED",
                            "key": key,
                            "original_title": title,
                            "verified_title": found_title,
                            "doi": doi,
                            "live_url": live_url,
                            "venue": (it.get('primary_location', {}).get('source') or {}).get('display_name', ''),
                            "year": it.get('publication_year'),
                            "citations": it.get('cited_by_count', 0),
                            "similarity": round(sim, 3)
                        }
    except Exception as e:
        pass
        
    # 2. Search Crossref
    url = "https://api.crossref.org/works?query.title=" + urllib.parse.quote(clean_title) + "&rows=3"
    req = urllib.request.Request(url, headers={'User-Agent': 'BibAuditor/1.0 (mailto:academic-audit@superior.edu.pk)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = data.get('message', {}).get('items', [])
            for it in items:
                found_title = it.get('title', [''])[0]
                sim = difflib.SequenceMatcher(None, clean_title.lower(), found_title.lower()).ratio()
                doi = it.get('DOI')
                if (sim >= 0.70 or clean_title.lower() in found_title.lower()) and doi:
                    doi_url = f"https://doi.org/{doi}"
                    live_url = test_url(doi_url)
                    if live_url:
                        return {
                            "status": "LIVE_VERIFIED",
                            "key": key,
                            "original_title": title,
                            "verified_title": found_title,
                            "doi": doi_url,
                            "live_url": live_url,
                            "venue": it.get('container-title', [''])[0],
                            "year": it.get('issued', {}).get('date-parts', [[0]])[0][0],
                            "citations": it.get('is-referenced-by-count', 0),
                            "similarity": round(sim, 3)
                        }
    except Exception as e:
        pass
        
    return {
        "status": "FAILED_UNVERIFIED",
        "key": key,
        "original_title": title,
        "original_authors": authors,
        "original_year": year
    }

def main():
    bib_path = Path("manuscript/updated_references.bib")
    if not bib_path.exists():
        bib_path = Path("manuscript/references.bib")
    bib_text = bib_path.read_text(encoding="utf-8")
    entries = re.findall(r'@(\w+)\{([^,]+),(.*?)\n\}', bib_text, re.DOTALL)
    
    verified = []
    unverified = []
    
    for idx, (entry_type, key, body) in enumerate(entries, 1):
        title_m = re.search(r'title=\{([^\}]+)\}', body, re.IGNORECASE)
        author_m = re.search(r'author=\{([^\}]+)\}', body, re.IGNORECASE)
        year_m = re.search(r'year=\{([^\}]+)\}', body, re.IGNORECASE)
        
        title = title_m.group(1).replace('{', '').replace('}', '').strip() if title_m else ""
        authors = author_m.group(1).strip() if author_m else ""
        year = year_m.group(1).strip() if year_m else ""
        
        print(f"[{idx:02d}/50] Checking {key}...", flush=True)
        res = verify_entry(key, title, authors, year)
        if res["status"] == "LIVE_VERIFIED":
            print(f"   -> [PASS] \"{res['verified_title'][:60]}...\" ({res['year']}) DOI: {res.get('doi')}", flush=True)
            verified.append(res)
        else:
            print(f"   -> [FAIL] \"{title}\" NOT FOUND LIVE", flush=True)
            unverified.append(res)
        time.sleep(0.2)
        
    Path("experiments/verified_references_audit.json").write_text(
        json.dumps({"verified": verified, "unverified": unverified}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\n" + "="*80, flush=True)
    print(f"TOTAL VERIFIED LIVE ON PUBLISHER WEBSITES: {len(verified)} / {len(entries)}", flush=True)
    print(f"TOTAL FAILED / HALLUCINATED / REMOVED:      {len(unverified)} / {len(entries)}", flush=True)
    if unverified:
        print("FAILED KEYS:")
        for u in unverified:
            print(f"  * {u['key']}: {u['original_title']}")
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
