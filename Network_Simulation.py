# -*- coding: utf-8 -*-
# Created with Python 3.6
"""
This code generates a co-evolving network.
An exchange of a quantity takes place on the network, while the weights of the links change randomly.
"""

import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
import numpy as np
from numpy import matlib as ml
import networkx as nx
import names
import copy


# Declare network parameters
NODES = 19
LINK_TH = 0.25
LINK_W_BASE = 0.25
LINK_W_DEV = 0.05
NODE_Q_INIT = 2

# Drawing parameters
NODE_SIZE_MIN = 0.4
NODE_FONT_SIZE = 20*NODE_SIZE_MIN
NODE_SIZE_SCALING = 3000
LINK_SIZE_SCALING = 10

# Simulation parameters
ITERATIONS = 1000
LINK_MOD_BASE = 1
LINK_MOD_DEV = 0.15
NODE_Q_MIN = 0.01
NODE_Q_MAX = 3
LINK_W_MIN = 0.01
LINK_W_MAX = 1

# Optional node dynamics
funcs = []
func1 = lambda x: np.tanh(x)
func2 = lambda x: -1*np.tanh(x)
for n in range(NODES):
    funcs.append(func1) if np.random.uniform() > 0.5 else funcs.append(func2)


def initialize_network():

    # Get names & quantities of nodes
    name_list, quantities = [], []
    for i in range(NODES):
        name = names.get_first_name()
        while name in name_list:
            name = names.get_first_name()
        name_list.append(name)
        quantities.append(np.random.uniform(0, NODE_Q_INIT))
    quantities_dict = dict(zip(name_list, quantities))

    # Get adjacency matrix
    links = np.random.uniform(LINK_W_BASE-LINK_W_DEV, LINK_W_BASE+LINK_W_DEV, size=(NODES, NODES))
    for i in range(NODES):
        for j in range(NODES):
            if i == j or np.random.uniform() > LINK_TH:
                links[i,j] = 0

    # Build network from adjacency matrix
    net = nx.from_numpy_matrix(links, create_using=nx.DiGraph())

    # Assign names
    mapping = dict(zip(net, name_list))
    net = nx.relabel_nodes(net, mapping)
    # Assign quantities
    nx.set_node_attributes(net, quantities_dict, 'Quantities')

    # Get layouting parameters
    layout  = nx.shell_layout(net)
    return net, name_list, layout

def update_visualization(net, layout, name_list, quantities):

    colors = ['black', '#003071', 'white']
    arrow = ArrowStyle('simple', head_length=25, head_width=20, tail_width=.75)

    nodesizes = copy.deepcopy(quantities)
    for n in range(len(quantities)):
        nodesizes[n] = NODE_SIZE_MIN if nodesizes[n] < NODE_SIZE_MIN else nodesizes[n]
        nodesizes[n] = nodesizes[n]*NODE_SIZE_SCALING
    edges = net.edges()
    linkwidths = [net[u][v]['weight']*LINK_SIZE_SCALING for u,v in edges]

    # Update network visualization
    nx.draw(net, pos=layout, node_size=nodesizes, node_color=colors[0],
            edges=edges, width=linkwidths, edge_color=colors[1],
            arrowstyle=arrow, arrowsize=.5, connectionstyle='arc3,rad=0.2',
            with_labels='True', font_color=colors[2], font_size=NODE_FONT_SIZE)

    txt = 'Total Quantity ' + str(np.round(np.sum(quantities),2))
    plt.text(-0.95, 1, txt, horizontalalignment='center', verticalalignment='center', fontsize=20)
    pylab.draw()
    plt.pause(.001)
    plt.clf()

def network_dynamics(quantities, links):

    # Quantities
    # Amount of quantity transferred to each node
    transfers = np.dot(quantities, links)
    # Amount of quantity lost at each node
    quantities_rep = ml.repmat(quantities, len(quantities), 1)
    losses = np.diagonal(np.dot(links, quantities_rep))
    # Merge losses & transfers
    #quantities = [quantities[n] - losses[n] + funcs[n](transfers[n]) for n in range(len(name_list))]
    quantities = [quantities[n] - losses[n] + transfers[n] for n in range(len(name_list))]
    # Links
    links = np.multiply(links, np.random.uniform(LINK_MOD_BASE-LINK_MOD_DEV, LINK_MOD_BASE+LINK_MOD_DEV,
                                                 size=(links.shape)))
    return quantities, links

def flatten(links):

    # Flatten links matrix
    links_flat = []
    for lin in range(len(links)):
        for col in range(len(links)):
            if links[lin, col] != 0:
                links_flat.append(links[lin, col])
    return links_flat

def limit_params(quantities, links_flat):

    # Constraining both the quantity and the link weights
    for params, boundaries in zip([quantities, links_flat], [[NODE_Q_MIN, NODE_Q_MAX], [LINK_W_MIN, LINK_W_MAX]]):
        for p in range(len(params)):
            if params[p] > boundaries[0] and params[p] < boundaries[1]:
                params[p] = params[p]
            elif params[p] < boundaries[0]:
                params[p] = boundaries[0]
            elif params[p] > boundaries[1]:
                params[p] = boundaries[1]
    return quantities, links_flat


if __name__ == "__main__":
    pylab.ion()
    fig = plt.figure(0, figsize=(16,8))
    fig.canvas.set_window_title('Propagation of a Quantity on a Dynamic Network')
    net, name_list, layout = initialize_network()

    # Repeatedly run dynamics and update visualization
    for i in range(ITERATIONS):

        # Get quantities
        quantities_dict = nx.get_node_attributes(net, 'Quantities')
        quantities = [quantities_dict[name] for name in name_list]
        update_visualization(net, layout, name_list, quantities)

        # Get links
        links = np.array(nx.convert_matrix.to_numpy_matrix(net, nodelist=name_list))
        # Simulate dynamics
        quantities, links = network_dynamics(quantities, links)
        # Convert matrix to list
        links_flat = flatten(links)
        # Constrain both quantities and links to a certain level
        quantities, links_flat = limit_params(quantities, links_flat)

        # Reassign attributes to nodes
        quantities_dict = dict(zip(name_list, quantities))
        nx.set_node_attributes(net, quantities_dict, 'Quantities')
        # Reassign weights to links
        for edge_triple, link in zip(net.edges(data=True), links_flat):
            edge_triple[-1]['weight'] = link

        # If user stops visualization, terminate the script
        if not plt.fignum_exists(0):
            break
