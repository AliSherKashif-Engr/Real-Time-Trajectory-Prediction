import re
import urllib.request
import urllib.parse
import json
import time
import difflib
import sys
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

def normalize(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return ' '.join(s.split())

def query_openalex(title, author=""):
    clean_title = re.sub(r'[{\}]', '', title)
    query = clean_title
    url = "https://api.openalex.org/works?search=" + urllib.parse.quote(query) + "&per-page=5"
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicReferenceAuditor/1.0 (mailto:academic-audit@superior.edu.pk)'})
    
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            results = data.get('results', [])
            norm_target = normalize(clean_title)
            
            for item in results:
                found_title = item.get('title') or ""
                norm_found = normalize(found_title)
                sim = difflib.SequenceMatcher(None, norm_target, norm_found).ratio()
                
                # Check authors if available
                authorships = item.get('authorships', [])
                found_authors = [a.get('author', {}).get('display_name', '') for a in authorships]
                
                # Check year and venue
                pub_year = item.get('publication_year')
                primary_loc = item.get('primary_location') or {}
                source = primary_loc.get('source') or {}
                venue = source.get('display_name') or ""
                doi = item.get('doi') or ""
                landing_url = primary_loc.get('landing_page_url') or doi or item.get('id')
                
                # Good match if title similarity > 0.68 or strong substring overlap
                if sim >= 0.68 or (len(norm_target) > 15 and norm_target in norm_found) or (len(norm_found) > 15 and norm_found in norm_target):
                    return {
                        "verified": True,
                        "found_title": found_title,
                        "similarity": round(sim, 3),
                        "doi": doi,
                        "url": landing_url,
                        "venue": venue,
                        "year": pub_year,
                        "authors": found_authors[:3],
                        "cited_by_count": item.get('cited_by_count', 0),
                        "source": "OpenAlex"
                    }
    except Exception as e:
        return {"verified": False, "error": str(e)}
        
    return {"verified": False, "reason": "No match above similarity threshold in OpenAlex"}

def main():
    bib_path = Path("manuscript/references.bib")
    bib_text = bib_path.read_text(encoding="utf-8")
    
    raw_entries = re.findall(r'@(\w+)\{([^,]+),(.*?)\n\}', bib_text, re.DOTALL)
    print(f"Auditing {len(raw_entries)} bibliography entries from {bib_path.name}...\n")
    
    audit_summary = []
    
    for idx, (entry_type, key, body) in enumerate(raw_entries, 1):
        title_m = re.search(r'title=\{([^\}]+)\}', body, re.IGNORECASE)
        author_m = re.search(r'author=\{([^\}]+)\}', body, re.IGNORECASE)
        year_m = re.search(r'year=\{([^\}]+)\}', body, re.IGNORECASE)
        journal_m = re.search(r'(journal|booktitle)=\{([^\}]+)\}', body, re.IGNORECASE)
        
        title = title_m.group(1).replace('{', '').replace('}', '').strip() if title_m else ""
        authors = author_m.group(1).strip() if author_m else ""
        year = year_m.group(1).strip() if year_m else ""
        venue = journal_m.group(2).strip() if journal_m else ""
        
        res = query_openalex(title, authors)
        
        status_label = "LIVE VERIFIED" if res.get("verified") else "NOT LIVE / UNVERIFIED"
        print(f"[{idx:02d}/50] [{status_label}] {key}")
        print(f"        Bib:   {title[:75]}... ({year})")
        if res.get("verified"):
            print(f"        Found: {res['found_title'][:75]}... ({res['year']})")
            print(f"        Venue: {res['venue']} | Citations: {res['cited_by_count']}")
            print(f"        Link:  {res['url']}")
        else:
            print(f"        Alert: {res.get('reason', res.get('error', 'Unknown'))}")
        print("-" * 80)
        
        audit_summary.append({
            "index": idx,
            "key": key,
            "bib_title": title,
            "bib_authors": authors,
            "bib_year": year,
            "bib_venue": venue,
            "verification": res
        })
        time.sleep(0.2)
        
    out_path = Path("experiments/openalex_audit_report.json")
    out_path.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    verified = [x for x in audit_summary if x['verification'].get('verified')]
    unverified = [x for x in audit_summary if not x['verification'].get('verified')]
    
    print("\n" + "="*80)
    print(f"FINAL AUDIT RESULT:")
    print(f"Total Entries:      {len(raw_entries)}")
    print(f"Live Verified:      {len(verified)} ({len(verified)/len(raw_entries)*100:.1f}%)")
    print(f"Unverified / Flags: {len(unverified)}")
    if unverified:
        print("\nUnverified Keys:")
        for u in unverified:
            print(f"  - [{u['key']}]: \"{u['bib_title']}\"")
    print("="*80)

if __name__ == "__main__":
    main()
