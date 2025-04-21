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
OUTAGE_TIME = 100  # seconds
DATA_ROOT = "outage_dataset"  # Path to save the dataset

SIMULATION = True  # Set to True to run the simulation
PLOTTING = False  # Set to True to plot the data


network_options = {
    "9": pn.case9,
    "14": pn.case14,
    "30": pn.case30,
    "39": pn.case39,
    "118": pn.case118,
    "24" : pn.case24_ieee_rts, 
}


def save_graph(net, path):
    filename = "graph.npy"

    edges = extract_edges_from_net(net)
    print(f"edges: {edges} of length {len(edges)}")

    edges = np.array(edges).T  # shape: [2, num_edges]
    np.save(os.path.join(path, filename), edges)
    print(f"Saved edge index to {os.path.join(path, filename)}")


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


# # temporal function. for now 
# def zero_pad(Y):
#     """
#     Pad the Y vector with zeros to match the length of the edges extracted from the network.
#     """
#     return np.pad(Y, (0, len(extract_edges_from_net) - len(Y)), mode='constant', constant_values=0)


def get_num_lines(net):
    """
    Get the number of lines in the network.
    """
    return len(net.line) + len(net.trafo) + len(net.trafo3w)  # Total number of lines and transformers


def Simulation(net, outage_cases, save_path):
    print("Simulation started")

    # num_lines = len(net.line)
    num_lines = get_num_lines(net)

    for i, outage in enumerate(outage_cases):
        print(f"Simulation {i+1} started | Outage lines: {outage}")

        # Reset network to all lines in service before each run
        net.line["in_service"] = True

        # Label vector: 1 where the line is outaged
        Y = np.zeros(num_lines, dtype=np.int32)
        for line in outage:
            Y[line] = 1

        data = simulate_power_flow(net, outage_lines=outage, total_time=TOTAL_TIME, sampling_rate=SAMPLING_RATE)
        X = data.values.astype(np.float32)  # shape: [1600, num_buses]

        cases_folder = os.path.join(save_path, "cases")
        os.makedirs(cases_folder, exist_ok=True)

        filename = os.path.join(cases_folder, f"case_{i:03d}.npz")
        np.savez(filename, x=X, y=Y)
        print(f"Saved simulation to {filename}")

        if PLOTTING:
            plot_data(data, bus_index=3, total_time=TOTAL_TIME, sampling_rate=SAMPLING_RATE) # change...
            plot_network(net)  # Plot the network

    print("Simulation completed")


def simulate_power_flow(net, outage_lines, total_time=200, sampling_rate=8):
    timesteps = total_time * sampling_rate  # Total number of time steps
    data = pd.DataFrame(index=range(timesteps), columns=range(len(net.bus)))  # DataFrame to store results

    for t in range(timesteps):
        if t == OUTAGE_TIME * sampling_rate:  # Apply line outage at OUTAGE_TIME sec
            for line in outage_lines:
                net.line.at[line, "in_service"] = False
                print(f"Line {line} outaged at t={t/sampling_rate} s")

        pp.runpp(net)  # Run power flow
        data.iloc[t] = net.res_bus.va_degree.values  # Store bus angles

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


def save_metadata(net, path):
    meta = {
        "num_buses": len(net.bus),
        "num_lines": len(net.line),
        "sampling_rate": SAMPLING_RATE,
        "total_time": TOTAL_TIME,
        "generator": "pandapower",
        "otuage_time": OUTAGE_TIME
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)


def prepare_dataset(net, name):
    save_path = os.path.join(DATA_ROOT, name)
    os.makedirs(os.path.join(save_path, "cases"), exist_ok=True)

    save_graph(net, save_path)
    save_metadata(net, save_path)

    outage_cases = generate_outage_cases(net, num_cases=NUM_CASES)
    print(f"Generated outage cases for {name}: {outage_cases}")

    if SIMULATION:
        Simulation(net, outage_cases, save_path)



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
    plt.axvline(x=OUTAGE_TIME, color='r', linestyle='--', label=f"Outage at {OUTAGE_TIME}s")  # Mark outage time
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
    print("Select a power network to generate the dataset:")
    print("Options:", ", ".join(f"IEEE {k}-bus" for k in network_options))

    network_choice = input("Enter number of buses (e.g., '39' for IEEE 39): ").strip()

    if network_choice in network_options:
        net = network_options[network_choice]()  # Lazy load
        prepare_dataset(net, f"ieee{network_choice}")
    else:
        print(f"Unknown choice '{network_choice}'. Please choose from: {', '.join(network_options.keys())}")

    
if __name__=="__main__":
    main()