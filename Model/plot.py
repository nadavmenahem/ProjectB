#===================IMPORTS==================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


#==================FUNCTIONS==================
# not exaactly the same as in SIMULATIONS
def plot_data(angles, bus_index, total_time=200, sampling_rate=8): 
    """
    Plot the phasor angle of a specific bus over time.
    """
    # Create a time vector (in seconds)
    time_vector = np.arange(0, total_time, 1 / sampling_rate)

    angles = angles[:,bus_index]
    print(f"angles shape: {angles.shape}")

    # Plot the phasor angle over time
    plt.figure(figsize=(10, 5))
    plt.plot(time_vector, angles, label=f"Bus {bus_index}", color='b')
    plt.axvline(x=100, color='r', linestyle='--', label="Outage at 100s")  # Mark outage time
    plt.xlabel("Time (s)")
    plt.ylabel("Phasor Angle (degrees)")
    plt.title(f"Phasor Angle of Bus {bus_index} Over Time")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.show(block = False)  # Show the plot without blocking the script
    # plt.pause(1)  # Optional: Give the GUI time to draw
    # plt.close()     # Optional: Close automatically


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
