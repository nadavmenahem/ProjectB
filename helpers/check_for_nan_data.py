import numpy as np, glob, os, itertools as it


DELETE_NAN = False
# PATH = "datasets/outage_dataset_with_loads_N2/ieee14" 
PATH = "datasets/N-2_random_outage_time/ieee14" 


contain_nan = False

for f in sorted(glob.glob(PATH + "/cases/case_*.npz")):
    x = np.load(f)["x"]
    # print("no nan")
    if np.isnan(x).any():
        print(os.path.basename(f), "contains NaNs")
        contain_nan = True

        if (DELETE_NAN):
            print("Deleting file")
            os.remove(f)
    

if (not contain_nan):
    print("Does not contain NaN")
