import re
import urllib.request
import urllib.parse
import json
import time
import difflib
from pathlib import Path

def normalize(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return ' '.join(s.split())

def check_crossref(title, author=""):
    query = f"{title} {author}".strip()
    url = "https://api.crossref.org/works?query=" + urllib.parse.quote(query) + "&rows=3"
    req = urllib.request.Request(url, headers={'User-Agent': 'BibAuditor/1.0 (mailto:scholar@example.edu)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = data.get('message', {}).get('items', [])
            for it in items:
                found_title = it.get('title', [''])[0]
                norm_orig = normalize(title)
                norm_found = normalize(found_title)
                sim = difflib.SequenceMatcher(None, norm_orig, norm_found).ratio()
                doi = it.get('DOI', '')
                url_link = f"https://doi.org/{doi}" if doi else it.get('URL', '')
                if sim > 0.65 or norm_orig in norm_found or norm_found in norm_orig:
                    return {
                        "status": "LIVE_VERIFIED",
                        "found_title": found_title,
                        "doi": doi,
                        "url": url_link,
                        "similarity": round(sim, 3),
                        "source": "Crossref"
                    }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}
    return {"status": "NOT_FOUND"}

def check_semantic_scholar(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote(title) + "&limit=3&fields=title,url,citationCount,year"
    req = urllib.request.Request(url, headers={'User-Agent': 'BibAuditor/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            papers = data.get('data', [])
            for p in papers:
                found_title = p.get('title', '')
                norm_orig = normalize(title)
                norm_found = normalize(found_title)
                sim = difflib.SequenceMatcher(None, norm_orig, norm_found).ratio()
                if sim > 0.65 or norm_orig in norm_found or norm_found in norm_orig:
                    return {
                        "status": "LIVE_VERIFIED",
                        "found_title": found_title,
                        "url": p.get('url', ''),
                        "similarity": round(sim, 3),
                        "citations": p.get('citationCount', 0),
                        "source": "SemanticScholar"
                    }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}
    return {"status": "NOT_FOUND"}

def main():
    bib_path = Path("manuscript/references.bib")
    bib_text = bib_path.read_text(encoding="utf-8")
    
    entries = re.findall(r'@(\w+)\{([^,]+),(.*?)\n\}', bib_text, re.DOTALL)
    print(f"Total entries in {bib_path.name}: {len(entries)}\n", flush=True)
    
    results = {}
    verified_count = 0
    failed_count = 0
    
    for idx, (entry_type, key, body) in enumerate(entries, 1):
        title_m = re.search(r'title=\{([^\}]+)\}', body, re.IGNORECASE)
        author_m = re.search(r'author=\{([^\}]+)\}', body, re.IGNORECASE)
        year_m = re.search(r'year=\{([^\}]+)\}', body, re.IGNORECASE)
        
        title = title_m.group(1).replace('{', '').replace('}', '').strip() if title_m else ""
        authors = author_m.group(1).strip() if author_m else ""
        first_author = authors.split(' and ')[0].split(',')[0].strip() if authors else ""
        year = year_m.group(1).strip() if year_m else ""
        
        print(f"[{idx}/{len(entries)}] Checking '{key}'...", flush=True)
        res = check_crossref(title, first_author)
        if res.get("status") != "LIVE_VERIFIED":
            # fallback
            time.sleep(0.5)
            res2 = check_semantic_scholar(title)
            if res2.get("status") == "LIVE_VERIFIED":
                res = res2
                
        if res.get("status") == "LIVE_VERIFIED":
            print(f"   -> VERIFIED ({res['source']}): '{res['found_title']}' (sim={res['similarity']}) Link: {res['url']}", flush=True)
            verified_count += 1
        else:
            print(f"   -> *** NOT FOUND / UNVERIFIED ***: Title='{title}' Author='{first_author}'", flush=True)
            failed_count += 1
            
        results[key] = {
            "title": title,
            "first_author": first_author,
            "year": year,
            "verification": res
        }
        time.sleep(0.3)
        
    out_file = Path("experiments/reference_audit_report.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n==========================================", flush=True)
    print(f"Audit Complete! Verified: {verified_count}/{len(entries)}, Failed: {failed_count}/{len(entries)}", flush=True)
    print(f"Report saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
