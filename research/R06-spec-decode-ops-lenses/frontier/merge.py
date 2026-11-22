"""Merge frontier/*.tsv into ledger.tsv with L ids; write frontier/id-map.tsv. Rerunnable: appends extra rows from frontier/z-*.tsv last."""
import csv, glob, os, sys
root = os.path.dirname(os.path.abspath(__file__))
rec = os.path.dirname(root)
cols = "id title year venue type url fetched quote proves does_not_prove bearing".split()
rows, idmap, seen = [], [], {}
for path in sorted(glob.glob(os.path.join(root, "*.tsv"))):
    if path.endswith("id-map.tsv"):
        continue
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not r.get("id"):
                continue
            url = (r.get("url") or "").strip().rstrip("/")
            key = (url, (r.get("quote") or "").strip())
            if key in seen:
                idmap.append((r["id"], seen[key])); continue
            lid = f"L{len(rows)+1}"
            seen[key] = lid
            idmap.append((r["id"], lid))
            rows.append([lid] + [(r.get(c) or "").replace("\t", " ").replace("\n", " ").strip() for c in cols[1:]])
with open(os.path.join(rec, "ledger.tsv"), "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows: f.write("\t".join(r) + "\n")
with open(os.path.join(root, "id-map.tsv"), "w") as f:
    f.write("part_id\tledger_id\n")
    for a, b in idmap: f.write(f"{a}\t{b}\n")
print(len(rows), "rows;", sum(1 for r in rows if r[6] == "unverified"), "unverified")
