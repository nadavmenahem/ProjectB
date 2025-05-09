#===================IMPORTS==================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


#==================FUNCTIONS==================
# not exaactly the same as in SIMULATIONS
def plot_data(angles, bus_index, total_time=200, sampling_rate=8, outage_time=100, fixed_ylim=(-20, 20)): 
    """
    Plot the phasor angle of a specific bus over time, with fixed y-axis scale for all buses.
    """
    time_vector = np.arange(0, total_time, 1 / sampling_rate)
    angles = angles[:, bus_index]
    # print(f"angles shape: {angles.shape}") = (1600,)

    plt.figure(figsize=(10, 5))
    plt.plot(time_vector, angles, label=f"Bus {bus_index}", color='b')
    plt.axvline(x=outage_time, color='r', linestyle='--', label=f"Outage at {outage_time}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Phasor Angle (degrees)")
    plt.title(f"Phasor Angle of Bus {bus_index} Over Time")
    plt.legend()
    plt.grid()

    if fixed_ylim is not None:
        plt.ylim(fixed_ylim)  # <<< fix y-axis no matter what

    plt.show()

    # plt.show(block = False)  # Show the plot without blocking the script
    # plt.pause(1)  # Optional: Give the GUI time to draw
    # plt.close()     # Optional: Close automatically


def plot_all_buses(case_data, total_time=200, sampling_rate=8, outage_time=100):

    if case_data.ndim == 3:
        case_data = case_data.squeeze(1)  # remove feature dimension if needed

    min = case_data.min()
    max = case_data.max()

    margin10 = 0.1 * (max - min)  # 10% margin
    y_lim = (min - margin10, max + margin10)

    # plot_data(X[case_number], bus_index=3, sampling_rate=sampling_rate, total_time=total_time,
    #            outage_time=outage_time, fixed_ylim=y_lim)

    print(f"case_data shape: {case_data.shape} and time: {case_data.shape[1]}")

    for i in range(case_data.shape[1]):
        plot_data(case_data, bus_index=i, sampling_rate=sampling_rate, 
                    total_time=total_time, outage_time=outage_time, fixed_ylim=y_lim)


def plot_graph(G, signal=None, numbering=False, faulty_lines=None):
    # Get layout for consistent node positions
    pos = nx.spring_layout(G, seed=42)

    # Prepare edge styling
    edges = list(G.edges())
    
    # If edge_mask is provided but smaller than the number of edges, pad it with zeros
    if faulty_lines is not None:
        if len(faulty_lines) < len(edges):
            faulty_lines = np.pad(faulty_lines, (0, len(edges) - len(faulty_lines)), 'constant', constant_values=0)
    
    if faulty_lines is not None:
        default_edges = [e for i, e in enumerate(edges) if faulty_lines[i] == 0]
        bold_edges = [e for i, e in enumerate(edges) if faulty_lines[i] == 1]
    else:
        default_edges = edges
        bold_edges = []

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=default_edges, edge_color='gray', style='dotted', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=bold_edges, edge_color='blue', width=2.5)

    # Draw edges
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=100)

    if numbering:
        # Draw node labels (optional, if you want to see node indices too)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        # Add edge indices as labels
        edge_labels = {edge: str(i) for i, edge in enumerate(edges)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # If a signal is provided, plot it as vertical bars on top of nodes
    if signal is not None:
        for i, (x, y) in pos.items():
            plt.plot([x, x], [y, y + signal[i]], color='blue', linewidth=2)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_test_case_probs(probs, true_labels, case_idx):
    num_lines = len(probs)
    x = np.arange(num_lines)

    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot predicted probabilities
    ax.bar(x - width/2, probs, width, label='Predicted Probability', color='blue', alpha=0.7)

    # Plot true labels
    ax.bar(x + width/2, true_labels, width, label='True Label (Normalized)', color='orange', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_lines)])
    ax.set_xlabel("Line Index")
    ax.set_ylabel("Value")
    ax.set_title(f"Test Case {case_idx}: Prediction vs Ground Truth")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()