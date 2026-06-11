import csv

# Đọc paraphrase_id từ train và test
train_ids = set()
with open("F:/Kiemtratrunglap/visp_train.csv", encoding="utf-8", errors="ignore") as f:
    for row in csv.DictReader(f):
        pid = row["paraphrase_id"]
        base = pid.split("-")[1] if "-" in pid else pid
        train_ids.add(base)

test_ids = set()
with open("F:/Kiemtratrunglap/visp_test.csv", encoding="utf-8", errors="ignore") as f:
    for row in csv.DictReader(f):
        pid = row["paraphrase_id"]
        base = pid.split("-")[1] if "-" in pid else pid
        test_ids.add(base)

overlap = train_ids & test_ids
print(f"Train unique paraphrase IDs: {len(train_ids)}")
print(f"Test unique paraphrase IDs: {len(test_ids)}")
print(f"Overlap (data leakage!): {len(overlap)}")

if overlap:
    print(f"  Sample overlapping IDs: {list(overlap)[:5]}")
else:
    print("  ✅ Không có overlap — train/test sạch")
