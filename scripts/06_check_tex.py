"""Static checks on paper/main.tex: environment balance, cite keys, \\input and figure files, stray control chars."""
import os, re, sys

P = os.path.join("paper", "main.tex")
s = open(P, encoding="utf-8").read()
ok = True
for env in ["table", "table*", "figure", "figure*", "itemize", "enumerate", "document", "abstract", "IEEEkeywords"]:
    b, e = s.count("\\begin{" + env + "}"), s.count("\\end{" + env + "}")
    if b != e: ok = False; print(f"env {env}: begin={b} end={e} MISMATCH")
keys = set()
for c in re.findall(r"\\cite\{([^}]*)\}", s):
    keys |= {k.strip() for k in c.split(",")}
bib = set(re.findall(r"@\w+\{([^,]+),", open(os.path.join("paper", "refs.bib"), encoding="utf-8").read()))
if keys - bib: ok = False; print("missing bib keys:", keys - bib)
print(f"{len(keys)} cite keys, {len(bib)} bib entries, unused: {sorted(bib - keys)}")
for f in re.findall(r"\\input\{([^}]*)\}", s):
    if not os.path.exists(os.path.join("paper", f + ".tex")): ok = False; print("missing input:", f)
for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", s):
    if not os.path.exists(os.path.join("paper", "figs", f)): print("figure not yet present:", f)
if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", s): ok = False; print("control characters found")
if re.search(r"\\pm\{\}", s): ok = False; print("old \\pm{} macro still used")
labels = set(re.findall(r"\\label\{([^}]*)\}", s)); refs = set(re.findall(r"\\ref\{([^}]*)\}", s))
if refs - labels: ok = False; print("undefined refs:", refs - labels)
print("todo markers:", len(re.findall(r"\\todo\{", s)))
print("OK" if ok else "PROBLEMS FOUND"); sys.exit(0 if ok else 1)
