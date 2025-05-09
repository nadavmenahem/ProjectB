import matplotlib.pyplot as plt
import numpy as np
import pandapower.plotting as pp_plot


CONFIG_FILE = "data_config.yaml"


def plot_data(data, bus_index, total_time, sampling_rate, outage_time):
    """
    Plot the phasor angle of a specific bus over time.
    """
    # Create a time vector (in seconds)
    time_vector = np.arange(0, total_time, 1 / sampling_rate)

    # Extract phasor angle of the selected bus
    phasor_angles = data[bus_index].values

    # Plot the phasor angle over time
    plt.figure(figsize=(10, 5))
    plt.plot(time_vector, phasor_angles, label=f"Bus {bus_index}", color='b')
    plt.axvline(x=outage_time, color='r', linestyle='--', label=f"Outage at {outage_time}s")  # Mark outage time
    plt.xlabel("Time (s)")
    plt.ylabel("Phasor Angle (degrees)")
    plt.title(f"Phasor Angle of Bus {bus_index} Over Time")
    plt.legend()
    plt.grid()
    plt.show(block = False)  # Show the plot without blocking the script
    plt.pause(1)  # Optional: Give the GUI time to draw
    plt.close()     # Optional: Close automatically


def plot_network(net):
    """
    Plot the power network using pandapower's simple plot function.
    """
    pp_plot.simple_plot(net, show_plot=False)  # Don't show it yet
    plt.show(block=False)  # Show the plot
    plt.pause(1)
    plt.close()
    # plt.draw()
    # plt.pause(0.001)