#===================IMPORTS==================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.fx import symbolic_trace
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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
    if outage_time is not None:
        plt.axvline(x=outage_time, color='r', linestyle='--', label=f"Outage at {outage_time}s")
    plt.xlabel("Step")
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


def plot_graph(G, signal=None, numbering=False, faulty_lines=None, bar_alpha=0.6, label_shift=0):
    """
    expects list of edge indices for faulty_lines
    """

    # 1) compute layout & edges
    pos   = nx.spring_layout(G, seed=42)
    edges = list(G.edges())
    nE    = len(edges)

    # 2) build a 0/1 mask array of length n_edges
    if faulty_lines is not None:
        mask = np.zeros(nE, dtype=int)
        for i in faulty_lines:
            if 0 <= i < nE:
                mask[i] = 1
        default_edges = [e for i, e in enumerate(edges) if mask[i] == 0]
        bold_edges    = [e for i, e in enumerate(edges) if mask[i] == 1]
    else:
        default_edges, bold_edges = edges, []

    # 3) draw edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=default_edges,
        edge_color='gray',
        style='dotted',
        alpha=0.5
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=bold_edges,
        edge_color='red',
        width=2.5
    )

    # 4) if you want edge‐indices printed
    if numbering:
        edge_labels = {edge: str(i) for i, edge in enumerate(edges)}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=6
        )

    # 5) if there’s a signal, plot bars **first** so nodes & labels go on top
    if signal is not None:
        # compute a little y‐offset to shift labels above the bar‐tops
        max_sig = np.max(signal)
        y_shift = (max_sig if max_sig>0 else 1) * label_shift

        for i, (x, y) in pos.items():
            plt.plot([x, x], [y, y + signal[i]],
                     linewidth=2,
                     alpha=bar_alpha,
                     color='green')

    # 6) draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='yellow',
        node_size=150)

    # 7) draw node‐labels (bus indices), shifted up if we have a signal
    if numbering:
        if signal is not None:
            # shift all labels up by y_shift
            label_pos = {i: (x, y + y_shift) for i, (x, y) in pos.items()}
        else:
            label_pos = pos
        nx.draw_networkx_labels(
            G, label_pos,
            font_size=8,
            font_color='black')

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_test_case_probs(probs, true_labels, case_idx, topk_indices=None, conformal_set=None):
    num_lines = len(probs)
    x = np.arange(num_lines)
    width = 0.35

    k = len(topk_indices) if topk_indices is not None else 1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    ax.bar(x - width/2, probs, width, label='Predicted Probability', color='blue', alpha=0.7)
    ax.bar(x + width/2, true_labels, width, label='True Label (Normalized)', color='orange', alpha=0.7)

    # Plot vertical lines for conformal set
    if conformal_set is not None:
        for idx in conformal_set:
            ax.axvline(
                x=idx,
                color='purple',         # high‐contrast
                linestyle='-',          # solid line
                linewidth=2.5,          # extra bold
                alpha=0.9,              # nearly opaque
                label='Conformal Set' if idx == conformal_set[0] else None
            )

    # Correct prediction?
    true_indices = np.where(true_labels > 0.1)[0]
    pred_indices = topk_indices if topk_indices is not None else np.argsort(probs)[::-1][:1]
    is_correct = set(true_indices).issubset(set(pred_indices))
    result_str = "Correct prediction" if is_correct else "Incorrect prediction"
    ax.set_title(f"Test Case {case_idx}: Prediction vs Ground Truth ({result_str})")

    # Basic x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_lines)])

    # Bold and color top-k indices
    if topk_indices is not None:
        fig.canvas.draw()  # ← Needed to populate tick labels
        tick_labels = ax.get_xticklabels()
        for i in topk_indices:
            tick_labels[i].set_color('red')
            tick_labels[i].set_fontweight('bold')

    ax.set_xlabel(f"Line Index (Top-{k} in red)")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)
    ax.grid(True)

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.show()


def plot_test_case_clean(
        probs,
        true_labels,
        case_idx,
        conformal_set=None,  # iterable of indices in CP set
        top_n=7,
        true_threshold=0.1,
):
    """
    Clean single-panel plot:
      - Horizontal bars for top-N probabilities (sorted desc).
      - Bars that belong to the conformal set are highlighted.
      - True label(s) within top-N are starred.
    """
    p = np.asarray(probs, float)
    y_true = np.asarray(true_labels, float)
    L = p.size

    S = set(conformal_set) if conformal_set is not None else set()
    true_idx = np.flatnonzero(y_true > true_threshold).tolist()

    # sort and keep only top-N
    order = np.argsort(p)[::-1]
    show = order[:min(top_n, L)]
    ranks = np.arange(1, show.size + 1)
    y = np.arange(show.size)[::-1]  # highest rank at top

    # colors: in-CP purple, otherwise blue
    in_cp = np.array([idx in S for idx in show])
    colors = np.where(in_cp, "#7b3294", "#4c78a8")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.barh(y, p[show], color=colors, alpha=0.9)

    # annotate probabilities
    for i, val in enumerate(p[show]):
        ax.text(val + 0.005, y[i], f"{val:.3f}", va="center", fontsize=9)

    # mark true labels (only if within top-N)
    for t in true_idx:
        if t in show:
            i = np.where(show == t)[0][0]
            ax.scatter(p[t], y[i], marker='*', s=160, facecolor='gold',
                       edgecolor='black', zorder=3)

    # y ticks: show rank + original index
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{r}  (idx {idx})" for r, idx in zip(ranks, show)], fontsize=10)

    # axes cosmetics
    ax.set_xlim(0, min(1.0, p[show].max() * 1.15 + 0.02))
    ax.set_xlabel("Predicted probability", fontsize=11)

    # coverage + set size summary
    covered = set(true_idx).issubset(S) if S else False
    ax.set_title(
        f"Case {case_idx} — CP cover: {'YES' if covered else 'NO'} (|S|={len(S)})",
        fontsize=13
    )

    # legend (only what matters)
    handles = [
        Patch(facecolor="#7b3294", label="In conformal set"),
        Patch(facecolor="#4c78a8", label="Not in set"),
        Line2D([0], [0], marker='*', ms=12, mfc='gold', mec='black',
               linestyle='None', label="True label"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    # footer note if CP set extends beyond top-N
    if S:
        shown_in_set = in_cp.sum()
        if shown_in_set < len(S):
            ax.text(
                0.99, 0.02,
                f"+{len(S) - shown_in_set} CP members not shown (outside top-{top_n})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="dimgray"
            )

    plt.tight_layout()
    plt.show()


def visualize_model(model, example_input):
    model.eval()
    gm = symbolic_trace(model)                 # FX GraphModule
    # Run once so shapes propagate if your model uses dynamic branches
    _ = model(example_input)

    G = nx.DiGraph()
    for node in gm.graph.nodes:
        G.add_node(node.name, label=f"{node.op}:{node.target}")

    # Edges: from input nodes to this node
    for node in gm.graph.nodes:
        for src in node.all_input_nodes:
            G.add_edge(src.name, node.name)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=0)          # layout free; can try kamada_kawai_layout
    nx.draw(G, pos, with_labels=False, node_size=800, alpha=0.9)
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.axis("off"); plt.tight_layout(); plt.show()