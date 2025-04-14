import pandapower as pp
import pandapower.networks as pn
# import pandapower.timeseries as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower.plotting as pp_plot
import os
import json

TOTAL_TIME = 200  # seconds
SAMPLING_RATE = 8  # Hz
NUM_CASES = 20
SAVE_PATH = "outage_dataset"  # Path to save the dataset
SIMULATION = False  # Set to True to run the simulation


def save_graph(net, filename):
    edges = extract_edges_from_net(net)  # Extract edges from the network
    
    print(f"edges: {edges} of length {len(edges)}")

    # Convert the edges to a NumPy array of shape (2, num_edges)
    edges = np.array(edges).T  # Transpose to make it (2, num_edges)
    
    # Save to npy file in the same dataset folder
    np.save(os.path.join(SAVE_PATH, f"{filename}.npy"), edges)
    print(f"Saved edge index to {os.path.join(SAVE_PATH, f'{filename}.npy')}")


def extract_edges_from_net(net):
    edges = []

    # Add standard lines
    for _, row in net.line.iterrows():
        from_bus = row['from_bus']
        to_bus = row['to_bus']
        edges.append((from_bus, to_bus))

    # Add two-winding transformers
    for _, row in net.trafo.iterrows():
        from_bus = row['hv_bus']
        to_bus = row['lv_bus']
        edges.append((from_bus, to_bus))

    # Optionally add 3-winding transformers if any exist
    if len(net.trafo3w):
        for _, row in net.trafo3w.iterrows():
            # Connect all pairs of the 3 buses (hv, mv, lv)
            buses = [row['hv_bus'], row['mv_bus'], row['lv_bus']]
            edges.extend([(b1, b2) for i, b1 in enumerate(buses) for b2 in buses[i+1:]])

    return edges


def Simulation(net, outage_cases=None):
    """
    Simulate power flow for a given network and save the results.
    """

    num_lines = len(net.line)

    print("Simulation started")
    for i, outage in enumerate(outage_cases):
        print(f"Simulation {i+1} started | Outage lines: {outage}")

        # Reset network to all lines in service before each run
        net.line["in_service"] = True

        # Label vector: 1 where the line is outaged
        y = np.zeros(num_lines, dtype=np.int32)
        for line in outage:
            y[line] = 1

        data = simulate_power_flow(net, outage_lines=outage, total_time=TOTAL_TIME, sampling_rate=SAMPLING_RATE)
        X = data.values.astype(np.float32)  # shape: [1600, num_buses]

        filename = os.path.join(SAVE_PATH, f"case_{i:03d}.npz")
        np.savez(filename, x=X, y=y)
        print(f"Saved simulation to {filename}")

        # plot_data(data, bus_index=bus_index, total_time=total_time, sampling_rate=sampling_rate)
        # plot_network(net)  # Plot the network

    print("Simulation completed")


def simulate_power_flow(net, outage_lines, total_time=200, sampling_rate=8):
    """
    Simulate power flow over a specified time period with a line outage.
    """
    print("start simulation...")
    timesteps = total_time * sampling_rate  # Total number of time steps
    data = pd.DataFrame(index=range(timesteps), columns=range(len(net.bus)))  # DataFrame to store results

    for t in range(timesteps):
        if t == 100 * sampling_rate:  # Apply line outage at 100s
            for line in outage_lines:
                net.line.at[line, "in_service"] = False
                print(f"Line {line} outaged at t={t/sampling_rate} s")

        pp.runpp(net)  # Run power flow
        data.iloc[t] = net.res_bus.va_degree.values  # Store bus angles

    print("simulation done")

    return data


def generate_outage_cases(net, num_cases=20):
    """
    Generate the outage cases with either 1 or 2 lines failing or no outage at all.
    return a list of lists, where each inner list contains the indices of the lines that are outaged.
    """
    num_lines = len(net.line)

    # Reserve 1 case for no outage
    remaining_cases = num_cases - 1

    # 75% for 1-line outages, 25% for 2-line outages
    num_1line = int(np.floor(0.75 * remaining_cases))
    num_2line = remaining_cases - num_1line  # Ensures total adds to (num_cases - 1)

    outage_cases = []
    outage_cases.append([])  # No outage case

    # Generate 1-line outages
    for _ in range(num_1line):
        line = np.random.choice(num_lines)
        outage_cases.append([line])

    # Generate 2-line outages
    for _ in range(num_2line):
        pair = np.random.choice(num_lines, size=2, replace=False)
        outage_cases.append(list(pair))

    # cleaning the cases a bit... some are regular ints and some are np.int32
    outage_cases = [[int(line) for line in case] for case in outage_cases]  # Convert to int

    return outage_cases


def plot_data(data, bus_index, total_time, sampling_rate):
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
    plt.axvline(x=100, color='r', linestyle='--', label="Outage at 100s")  # Mark outage time
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


def main():
    # net_9 = pn.case9()  # IEEE 9-bus system

    net_39 = pn.case39()  # IEEE 39-bus system
    # plot_network(net_39)  # Plot the network

    os.makedirs(SAVE_PATH, exist_ok=True)

    save_graph(net_39, "Graph")  # Save the network graph to a JSON file

    outage_cases = generate_outage_cases(net_39, num_cases=NUM_CASES)  # Generate outage cases
    print(f"outage cases: {outage_cases}")  # Print the generated outage cases
    
    
    if SIMULATION:
        Simulation(net=net_39, outage_cases=outage_cases)  # Run the simulation

    
if __name__=="__main__":
    main()