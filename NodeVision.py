import networkx as nx
import matplotlib.pyplot as plt
import math
from GraphPositioning import GraphPositioning as gpos
from random import choice, random


class NodeVision(object):
    """
    This object is used for laying out the nodes of a graph according to \
    the local vision of a specific node (root node)
    """

    def __init__(self, graph=None, root_node=0):
        """
        :param graph: Peer network
        :param root_node: Node id in the network, by default 0
        """
        if graph is None:
            self.graph = nx.DiGraph()
            self.graph.add_node(root_node)
        else:
            self.graph = graph

        self.root_node = root_node
        self.bfs_tree = {}
        self.pos = self.lay_down_nodes()
        self.component = None

    def set_root_node(self, rootnode):
        self.root_node = rootnode
        self.pos = self.lay_down_nodes()

    @property
    def n_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def node_positions(self):
        return dict(self.graph.nodes(data='pos'))

    def reposition_nodes(self):
        self.pos = self.lay_down_nodes()

    def add_transactions(self, transactions):
        for tr in transactions:
            # Gw.graph.add_edge(tr['downloader'],
            #                  tr['uploader'], weight=tr['amount'])
            self.add_edge_to_vision(tr['downloader'],
                                    tr['uploader'], tr['amount'])

    def make_random_transactions(self, tr_count):
        trs = []
        for i in range(tr_count):
            neigh = choice(list(self.graph.nodes().keys()))
            if neigh == self.root_node:
                continue
            if random() > 0.8:
                trs.append({'downloader': self.root_node,
                            'uploader': neigh,
                            'amount': random() * 100})
            else:
                trs.append({'downloader': neigh,
                            'uploader': self.root_node,
                            'amount': random() * 100})
        self.add_transactions(trs)

    def add_edge_to_vision(self, n1, n2, w):
        if n1 in self.graph and n2 in self.graph.successors(n1):
            self.graph[n1][n2]['weight'] *= 0.8
            self.graph[n1][n2]['weight'] += (0.2 * w)
            print('Existing edge !!!')
        else:
            print('Non-Existing edge !!!')
            self.graph.add_edge(n1, n2, weight=w)

    def diminish_weights(self, remove=True):
        to_be_removed = []
        n_rem, n_not_rem = 0, 0
        for n1, n2 in self.graph.edges():
            self.graph[n1][n2]['weight'] *= 0.9
            if self.graph[n1][n2]['weight'] < 0.5:
                n_rem += 1
                to_be_removed.append((n1, n2))
            else:
                n_not_rem += 1
        if remove:
            print('Removed: {}, Not removed: {}'.format(n_rem, n_not_rem))
            self.graph.remove_edges_from(to_be_removed)

    def normalize_edge_weights(self, minwidth=0.5, maxwidth=2):
        weights = [w for (n1, n2, w) in self.graph.edges(data='weight')]
        maxw = max(weights)
        minw = min(weights)

        width_diff = (maxwidth - minwidth)
        weight_diff = (maxw - minw)

        for n1, n2 in self.graph.edges():
            w = self.graph[n1][n2]['weight']
            self.graph[n1][n2]['weight'] = minwidth + (width_diff
                                                       * ((w - minw)
                                                          / weight_diff))

    def lay_down_nodes(self):
        H = self.graph.to_undirected()
        # Remove disconnected nodes from the graph
        component_nodes = nx.node_connected_component(H, self.root_node)
        for node in list(H.nodes()):
            if node not in component_nodes:
                H.remove_node(node)
        bfs_tree = nx.bfs_tree(H, self.root_node)
        self.bfs_tree[self.root_node] = bfs_tree
        pos = gpos.hierarchy_pos(bfs_tree, self.root_node,
                                 width=2 * math.pi, xcenter=0)
        new_pos = {u: (r * math.cos(theta), r * math.sin(theta))
                   for u, (theta, r) in pos.items()}

        nx.set_node_attributes(self.graph, new_pos, 'pos')
        return new_pos

    def show_undirected_bfs_tree(self):

        nx.draw(self.bfs_tree[self.root_node],
                pos=self.node_positions,
                node_size=50)
        plt.show()

    def update_component(self):
        H = self.graph.to_undirected()
        self.component = nx.DiGraph(self.graph)

        component_nodes = nx.node_connected_component(H, self.root_node)
        for node in self.graph:
            if node not in component_nodes:
                self.component.remove_node(node)

    def show_directed_neighborhood(self):
        # nx.draw_networkx_nodes(self.graph, pos=self.pos, node_size=50)
        # nx.draw_networkx_nodes(self.graph, pos=self.pos,
        #                        nodelist=[self.rootnode],
        #                        node_color='blue', node_size=100)
        # nx.draw_networkx_edges(self.graph, pos=self.pos, edge_color='gray',
        #                        alpha=0.5, style='dashed')

        nx.draw_networkx_nodes(self.component, pos=self.node_positions,
                               node_size=50)
        nx.draw_networkx_nodes(self.component, pos=self.node_positions,
                               nodelist=[self.root_node],
                               node_color='blue', node_size=100)
        nx.draw_networkx_edges(self.component, pos=self.node_positions,
                               edge_color='gray', alpha=0.5, style='dashed')
        plt.show()
