import csv

for fname in ["visp_train.csv", "visp_test.csv"]:
    path = f"F:/Kiemtratrunglap/{fname}"
    with open(path, encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    para = sum(1 for r in rows if r["paraphrase_id"].startswith("para-"))
    rpara = sum(1 for r in rows if r["paraphrase_id"].startswith("rpara-"))
    min_a = min(len(r["original_text"]) for r in rows)
    max_a = max(len(r["original_text"]) for r in rows)
    min_b = min(len(r["paraphrase_text"]) for r in rows)
    max_b = max(len(r["paraphrase_text"]) for r in rows)
    empty_a = sum(1 for r in rows if not r["original_text"].strip())
    empty_b = sum(1 for r in rows if not r["paraphrase_text"].strip())
    print(f"{fname}:")
    print(f"  Rows: {len(rows)}")
    print(f"  para: {para}, rpara: {rpara}")
    print(f"  original_text len: {min_a} - {max_a}")
    print(f"  paraphrase_text len: {min_b} - {max_b}")
    print(f"  Empty texts: {empty_a + empty_b}")
    print()
