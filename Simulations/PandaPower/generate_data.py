import pandapower as pp
import pandapower.networks as pn
import pandapower.timeseries as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower.plotting as pp_plot

net_9 = pn.case9()  # IEEE 39-bus system


# Define time parameters
total_time = 200  # seconds
sampling_rate = 8  # Hz
timesteps = total_time * sampling_rate  # 1600 time points

# Create time series container
data = pd.DataFrame(index=range(timesteps), columns=range(len(net_9.bus)))

line_outage_idx = 3

# Run power flow for each time step
for t in range(timesteps):
    if t == 100 * sampling_rate:  # Apply line outage at 100s
        net_9.line.at[line_outage_idx, "in_service"] = False  # Outage
        print(f"Line {line_outage_idx} outaged at t={t/sampling_rate} s")

    pp.runpp(net_9)  # Run power flow
    data.iloc[t] = net_9.res_bus.va_degree.values  # Store bus angles


# Select a bus to plot (e.g., bus index 5)
bus_index = 3

# Create a time vector (in seconds)
time_vector = np.arange(0, total_time, 1 / sampling_rate)

# Extract phasor angle of the selected bus
phasor_angles = data[bus_index].values

# Plot the phasor angle over time
plt.figure(figsize=(10, 5))
plt.plot(time_vector, phasor_angles, label=f"Bus {bus_index}", color='b')
plt.axvline(x=100, color='r', linestyle='--', label="Outage at 100s")  # Mark outage time
plt.xlabel("Time (s)")
plt.ylabel("Phasor Angle (degrees)")
plt.title(f"Phasor Angle of Bus {bus_index} Over Time")
plt.legend()
plt.grid()
plt.show()


# Plot using simple plot
pp_plot.simple_plot(net_9, show_plot=True)