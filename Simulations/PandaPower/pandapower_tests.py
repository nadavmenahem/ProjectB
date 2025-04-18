import pandapower.networks as nw
import pandapower as pp
import numpy as np
import time
import pandapower.plotting as pplt


def save_graph(net, path):
    # filename = "graph.npy"

    edges = extract_edges_from_net(net)
    print(f"edges: {edges} of length {len(edges)}")

    edges = np.array(edges).T  # shape: [2, num_edges]
    # np.save(os.path.join(path, filename), edges)
    # print(f"Saved edge index to {os.path.join(path, filename)}")

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


def main():
    # net = nw.case24_ieee_rts()
    net = nw.case118()

    print(f"num_buses: {len(net.bus)}, num_lines: {len(net.line)}")
    print(f"lines: {net.line}")

    save_graph(net, ".")


if __name__=="__main__":
    main()