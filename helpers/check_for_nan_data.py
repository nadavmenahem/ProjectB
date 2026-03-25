import numpy as np, glob, os, re

DELETE_NAN = True
WRITE_TRIPPED_LINES = False  # if False, we won't touch OUTPUT_FILE
# PATH = "C:/Users/nadavmen/Work/ProjectB/Code/datasets/TT40_OT20_NS003_M40_N-2/ieee39"
PATH = "/mnt/c/Users/nadav/work/ProjectB/datasets/TT40_OT20_NS003_M40_N-2/ieee14"

OUTPUT_FILE = os.path.join(PATH, "nan_cases_outaged_lines.txt")

def extract_case_number(filename):
    m = re.search(r"case_(\d+)\.npz$", os.path.basename(filename))
    return int(m.group(1)) if m else -1

contain_nan = False
wrote_any = False

files = sorted(
    glob.glob(os.path.join(PATH, "cases", "case_*.npz")),
    key=extract_case_number
)

print("Search path:", os.path.join(PATH, "cases", "case_*.npz"))
print("Found files:", len(files))

# Prepare a lazy file handle only if we intend to write
fout = None
try:
    if WRITE_TRIPPED_LINES:
        os.makedirs(PATH, exist_ok=True)
        # Append mode so previous content is preserved across runs
        fout = open(OUTPUT_FILE, "a", encoding="utf-8")

    for f in files:
        fname = os.path.basename(f)
        print(fname)

        # compute flags inside with; do NOT keep arrays after exiting
        has_nan = False
        outaged_lines = None

        with np.load(f) as data:  # no mmap -> safe to delete later on Windows
            if "x" not in data:
                print(f"{fname} missing 'x' array — skipping")
                continue
            if "y" not in data:
                print(f"{fname} missing 'y' (outage indices) — skipping")
                continue

            x = data["x"]
            y = data["y"]

            if np.issubdtype(x.dtype, np.floating):
                has_nan = np.isnan(x).any()

            if has_nan:
                contain_nan = True
                print(f"{fname} contains NaNs")
                if WRITE_TRIPPED_LINES:
                    outaged_lines = np.flatnonzero(y).tolist()

        # file is closed here; safe to delete
        if has_nan and DELETE_NAN:
            print(f"Deleting file: {fname}")
            try:
                os.remove(f)
            except PermissionError as e:
                print(f"Could not delete {fname} (locked): {e}")

        if has_nan and WRITE_TRIPPED_LINES and outaged_lines is not None:
            fout.write(f"{fname}\t{outaged_lines}\n")
            wrote_any = True

finally:
    if fout is not None:
        fout.close()

if not contain_nan:
    print("Does not contain NaN")
else:
    if WRITE_TRIPPED_LINES and wrote_any:
        print(f"Wrote outaged-line indices for NaN cases to: {OUTPUT_FILE}")
    elif WRITE_TRIPPED_LINES and not wrote_any:
        print("Some files had NaNs but no outages were recorded.")
    else:
        print("NaN cases found; outages not written (WRITE_TRIPPED_LINES=False).")
