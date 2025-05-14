import numpy as np, glob, os, itertools as it

contain_nan = False

for f in sorted(glob.glob("C:/Users/nadav/OneDrive - Technion/technion/ProjectB/Code/outage_dataset_noisy/ieee14/cases/case_*.npz")):
    x = np.load(f)["x"]
    if np.isnan(x).any():
        print(os.path.basename(f), "contains NaNs")
        contain_nan = True

if (not contain_nan):
    print("Does not contain NaN")
