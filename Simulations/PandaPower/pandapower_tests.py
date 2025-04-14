import pandapower.networks as nw
import pandapower as pp
import numpy as np
import time
import pandapower.plotting as pplt


def main():
    net = nw.case9()

    from_buses = net.line["from_bus"].values
    to_buses = net.line["to_bus"].values

    edges = list(zip(from_buses, to_buses))  # List of (source, target)
    print(edges)

    # Plot using simple plot
    # pplt.simple_plot(net_9, show_plot=True)  
    # ax = pplt.simple_plot(net_9, show_plot=False)
    # clc = pplt.create_line_collection(net_9, net_9.line.index, color="red", linewidth=2)
    # pplt.draw_collections([clc], ax=ax)
    # pplt.show_plot()

    # ax = pplt.simple_plot(net_39, show_plot=False)
    # clc = pplt.create_line_collection(net_39, net_39.line.index, color="red", linewidth=2)
    # pplt.draw_collections([clc], ax=ax)
    # pplt.show_plot()


    # net_118 = nw.case118()
    # print("hello")
    # pp.runpp(net_39)  # Re-run power flow
    # angles = net_39.res_bus.va_degree.values  # Collect voltage angles
    # print(angles)
    # print(angles.size)


if __name__=="__main__":
    main()