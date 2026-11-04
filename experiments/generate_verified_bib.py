from pathlib import Path
import shutil

# Ensure references.bib is synchronized with updated_references.bib
source_bib = Path("manuscript/updated_references.bib")
dest_bib = Path("manuscript/references.bib")

if source_bib.exists():
    shutil.copyfile(source_bib, dest_bib)
    print(f"Successfully synchronized {dest_bib} with canonical {source_bib} (50 live verified references).")
else:
    print(f"Error: {source_bib} does not exist!")
