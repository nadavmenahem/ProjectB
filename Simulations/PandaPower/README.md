```bash
pip install pandapower
pip install numba
```

# PandaPower
## net.res_bus.va_degree.values
After running a power flow (pp.runpp(net)), pandapower stores the results in the res_* DataFrames
- net.res_bus: a table of results for all buses.
- va_degree: the voltage angle (phasor angle) at each bus, in degrees.
- .values: given you the raw NumPy array version of the va_degree column.

# .npz
.npz is a NumPy zip file format - it's a compressed container for mulitple NumPy arrays.
