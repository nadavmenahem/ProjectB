import numpy as np, glob, os, itertools as it


DELETE_NAN = True
PATH = "datasets\outage_dataset_with_loads\ieee14"


contain_nan = False

for f in sorted(glob.glob(PATH + "/cases/case_*.npz")):
    x = np.load(f)["x"]
    if np.isnan(x).any():
        print(os.path.basename(f), "contains NaNs")
        contain_nan = True

        if (DELETE_NAN):
            print("Deleting file")
            os.remove(f)

if (not contain_nan):
    print("Does not contain NaN")
