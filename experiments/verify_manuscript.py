import re
import shutil
from pathlib import Path
from pypdf import PdfReader

def verify():
    workspace = Path("d:/PhD in Data Science/Research Going/working on papers/Paper_01_DeepLearning")
    manuscript_dir = workspace / "manuscript"
    tex_path = manuscript_dir / "IEEE_TITS_Paper_01_Manuscript.tex"
    bib_path = manuscript_dir / "updated_references.bib"
    ref_bib_path = manuscript_dir / "references.bib"
    pdf_path = manuscript_dir / "IEEE_TITS_Paper_01_Manuscript.pdf"
    root_pdf_path = workspace / "PAPER_01_COMPLETE_MANUSCRIPT_DRAFT.pdf"

    print("=== 1. Syncing Root PDF ===")
    shutil.copyfile(pdf_path, root_pdf_path)
    print(f"Copied {pdf_path.name} to {root_pdf_path.name} ({root_pdf_path.stat().st_size / 1024:.1f} KB)")

    print("\n=== 2. PDF Page Count Check ===")
    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    print(f"Total PDF pages: {num_pages} (Target for IEEE journal: 10-12 pages)")

    print("\n=== 3. Abstract Word Count Check ===")
    tex_content = tex_path.read_text(encoding="utf-8")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_content, re.DOTALL)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        # strip LaTeX commands
        clean_abstract = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})?", "", abstract_text)
        words = clean_abstract.split()
        word_count = len(words)
        print(f"Abstract word count: {word_count} words (Requirement: 150-250 words)")
        assert 150 <= word_count <= 250, f"Abstract word count {word_count} out of bounds!"
    else:
        print("ERROR: Abstract not found!")

    print("\n=== 4. Markers Check ([CITATION NEEDED], [RESULT NEEDED]) ===")
    cit_needed = re.findall(r"\[CITATION NEEDED\]", tex_content, re.IGNORECASE)
    res_needed = re.findall(r"\[RESULT NEEDED[^\]]*\]", tex_content, re.IGNORECASE)
    print(f"[CITATION NEEDED] markers: {len(cit_needed)}")
    print(f"[RESULT NEEDED] markers: {len(res_needed)}")

    print("\n=== 5. Figures and Tables In-Text References Check ===")
    for fig_num in range(1, 6):
        pattern = rf"(Fig\.\s*{fig_num}|Figure\s*{fig_num}|fig:{fig_num}|fig{fig_num})"
        matches = re.findall(pattern, tex_content)
        print(f"Figure {fig_num} referenced in text: {len(matches)} times")

    table_labels = [
        ("Table I (Related Work)", r"(Table\s*I\b|TABLE\s*I\b|tab:related_work_comparison)"),
        ("Table II (Trajectory)", r"(Table\s*II\b|TABLE\s*II\b|tab:trajectory_results)"),
        ("Table III (Anomaly)", r"(Table\s*III\b|TABLE\s*III\b|tab:anomaly_results)"),
        ("Table IV (Ablation)", r"(Table\s*IV\b|TABLE\s*IV\b|tab:ablation_results)"),
        ("Table V (Throughput)", r"(Table\s*V\b|TABLE\s*V\b|tab:throughput)"),
    ]
    for name, pattern in table_labels:
        matches = re.findall(pattern, tex_content)
        print(f"{name} referenced in text: {len(matches)} times")

    print("\n=== 6. Bibliography and Citation Resolution Check ===")
    # Extract keys in updated_references.bib
    bib_content = bib_path.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))
    print(f"Total unique references in updated_references.bib: {len(bib_keys)}")

    # Extract keys in references.bib
    ref_bib_content = ref_bib_path.read_text(encoding="utf-8")
    ref_bib_keys = set(re.findall(r"@\w+\{([^,]+),", ref_bib_content))
    print(f"Total unique references in references.bib: {len(ref_bib_keys)}")
    assert bib_keys == ref_bib_keys, "Discrepancy between updated_references.bib and references.bib!"

    # Extract cited keys in .tex
    cited_keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex_content):
        keys = [k.strip() for k in match.group(1).split(",")]
        cited_keys.update(keys)
    print(f"Total unique cited keys in .tex: {len(cited_keys)}")

    missing_in_bib = cited_keys - bib_keys
    if missing_in_bib:
        print(f"WARNING: Cited keys missing in .bib: {missing_in_bib}")
        assert False, f"Missing cited keys: {missing_in_bib}"
    else:
        print("All cited keys exist in updated_references.bib! (0 missing keys)")

    unused_in_bib = bib_keys - cited_keys
    print(f"Keys in bib but not cited in tex: {len(unused_in_bib)}")
    if unused_in_bib:
        print(f"Unused keys: {unused_in_bib}")

    print("\n=== 7. PDF Text Search for Unresolved Markers ('??' or '[?]') ===")
    full_pdf_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        full_pdf_text += text
        if "??" in text:
            print(f"Warning: '??' detected on page {idx+1}")
        if "[?]" in text:
            print(f"Warning: '[?]' detected on page {idx+1}")
    print("Verification script completed successfully!")

if __name__ == "__main__":
    verify()
