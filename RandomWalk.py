import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.lines as mtline
from matplotlib.animation import FuncAnimation
import networkx as nx
import math
import random


class RandomWalk(object):

    def __init__(self, sample_graph):
        self.gr = sample_graph.graph
        self.local_vision = sample_graph
        self.n_nodes = sample_graph.n_nodes
        self.n_edges = len(self.gr.edges())
        self.nodes = np.zeros(self.n_nodes,
                              dtype=[('position', float, 2),
                                     ('size',     float, 1),
                                     ('rank',     float, 1),
                                     ('growth',   float, 1),
                                     ('color',    float, 4),
                                     ('facecolor', float, 4)])

        self.edges = np.zeros(self.n_edges,
                              dtype=[('width',     float, 1),
                                     ('edgestyle', np.unicode, 16),
                                     ('alpha',     float, 1),
                                     ('color',     float, 4)])

        self.pos = sample_graph.pos

        self.lines = []
        self.scat = None
        self.walk_params = self.default_walk_params()
        self.edge_indices = {}

    def default_walk_params(self):
        return {'growthrate': 30,
                'n_walk': 300,
                'n_step': 300,
                'reset_prob': 0.1}

    def set_walk_params(self, params):
        if 'growthrate' in params:
            self.walk_params['growthrate'] = params['growthrate']
        if 'n_walk' in params:
            self.walk_params['n_walk'] = params['n_walk']
        if 'n_step' in params:
            self.walk_params['n_step'] = params['n_step']
        if 'reset_prob' in params:
            self.walk_params['reset_prob'] = params['reset_prob']

    def show_walk(self, savevid=False):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 1), ax.set_xticks([])
        ax.set_ylim(0, 1), ax.set_yticks([])

        self.nodes['position'] = self.normalize_positions()
        self.nodes['size'] = np.random.uniform(100, 100, self.n_nodes)
        self.nodes['growth'] = np.random.uniform(10, 20, self.n_nodes)
        self.nodes['facecolor'] = np.random.uniform(0, 1, (self.n_nodes, 4))

        # Draw edges
        x1s, x2s, y1s, y2s, lws = [], [], [], [], []

        for edge in self.gr.edges:
            x1s.append(self.nodes['position'][edge[0], 0])
            y1s.append(self.nodes['position'][edge[0], 1])
            x2s.append(self.nodes['position'][edge[1], 0])
            y2s.append(self.nodes['position'][edge[1], 1])
            lws.append(self.gr[edge[0]][edge[1]]['weight'])

        self.lines = ax.plot([x1s, x2s], [y1s, y2s], color='gray',
                             alpha=0.4, linestyle='--', lw=0.5)

        for i in range(len(self.lines)):
            self.lines[i].set_xdata([x1s[i], x2s[i]])
            self.lines[i].set_ydata([y1s[i], y2s[i]])
            self.lines[i].set_linewidth(lws[i]/80)
            self.lines[i].set_ls('dashed')

        # Draw nodes
        self.scat = ax.scatter(self.nodes['position'][:, 0],
                               self.nodes['position'][:, 1],
                               s=self.nodes['size'], lw=0.5,
                               edgecolors=self.nodes['color'],
                               facecolors='red')

        self.current_node = self.local_vision.rootnode
        self.growthrate = self.walk_params['growthrate']
        self.n_walk = self.walk_params['n_walk']
        self.n_step = self.walk_params['n_step']
        self.reset_prob = self.walk_params['reset_prob']
        self.visiteds = []

        self.animation = FuncAnimation(fig, self.update,
                                       interval=10, init_func=self.init)

        if savevid:
            # self.animation.save('anim.gif', dpi=60, writer='imagemagick')
            self.animation.save('anim.mp4')
        else:
            plt.show()

    def make_step(self, nodeid, visiteds):
        # print(nodeid)
        self.n_step -= 1
        if ((self.gr.out_degree(nodeid) == 0) or (np.random.uniform(0, 1) <
                                                  self.reset_prob)):
            return self.local_vision.rootnode, []

        total_weight = 0
        for neigh in self.gr.successors(nodeid):
            total_weight += self.gr[nodeid][neigh]['weight']
            print('{} - {} '.format(neigh, self.gr[nodeid][neigh]['weight'])),

        prob = np.random.uniform(0, total_weight)
        print(prob)

        total_weight = 0
        for neigh in self.gr.neighbors(nodeid):
            total_weight += self.gr[nodeid][neigh]['weight']
            if prob < total_weight:
                if neigh in visiteds:
                    return self.local_vision.rootnode, []
                visiteds.append(neigh)
                return neigh, visiteds

    def init(self):
        self.nodes['facecolor'][:, 0] = 1
        self.nodes['facecolor'][:, 1] = 0
        self.nodes['facecolor'][:, 2] = 0
        self.nodes['facecolor'][:, 3] = 0.3

        self.nodes['facecolor'][self.local_vision.rootnode] = (0.1, 0.8,
                                                               0.1, 0.7)

        x1s, x2s, y1s, y2s, lws, lcs, lss = [], [], [], [], [], [], []

        curr_index = 0
        for edge in self.gr.edges:
            x1s.append(self.nodes['position'][edge[0], 0])
            y1s.append(self.nodes['position'][edge[0], 1])
            x2s.append(self.nodes['position'][edge[1], 0])
            y2s.append(self.nodes['position'][edge[1], 1])

            lcs.append('gray')
            lws.append(self.gr[edge[0]][edge[1]]['weight'])
            lss.append('--')

            self.edge_indices[(edge[0], edge[1])] = curr_index
            self.edge_indices[(edge[1], edge[0])] = curr_index
            curr_index += 1

        for i in range(len(self.lines)):
            self.lines[i].set_xdata([x1s[i], x2s[i]])
            self.lines[i].set_ydata([y1s[i], y2s[i]])
            self.lines[i].set_lw(lws[i]/80)
            self.lines[i].set_color(lcs[i])
            self.lines[i].set_ls(lss[i])

    def update(self, frame_number):
        self.current_index = frame_number % self.n_nodes

        print('Walk', self.n_walk)
        if self.n_walk < 1:
            self.animation.event_source.stop()
            return

        self.next_node, self.visiteds = self.make_step(self.current_node,
                                                       self.visiteds)

        if self.next_node == 0:
            self.n_walk -= 1
            self.n_step = 300

        self.nodes['facecolor'][self.current_node] = (0, 1, 0, 1)
        self.nodes['facecolor'][self.local_vision.rootnode] = (0, 1, 1, 1)
        self.nodes['size'][self.current_node] += self.growthrate

        # self.scat.set_edgecolors(self.nodes['color'])
        self.scat.set_facecolors(self.nodes['facecolor'])
        self.scat.set_sizes(self.nodes['size'])
        # self.scat.set_offsets(self.nodes['position'])

        lws, lcs, lss = [], [], []

        if len(self.visiteds) == 0:
            # Reset colors

            lws, lcs, lss = [], [], []
            for edge in self.gr.edges:
                lcs.append('gray')
                lws.append(self.gr[edge[0]][edge[1]]['weight'])
                lss.append('--')

            for i in range(len(self.lines)):
                self.lines[i].set_lw(lws[i]/80)
                self.lines[i].set_color(lcs[i])
                self.lines[i].set_ls(lss[i])
        else:
            ind = self.edge_indices[(self.current_node, self.next_node)]
            self.lines[ind].set_color('b')
            self.lines[ind].set_lw(4)
            self.lines[ind].set_ls('--')

        self.current_node = self.next_node

    def normalize_positions(self):
        poslist = list(self.pos.values())
        minx = min(poslist, key=lambda t: t[0])[0]
        miny = min(poslist, key=lambda t: t[1])[1]
        maxx = max(poslist, key=lambda t: t[0])[0]
        maxy = max(poslist, key=lambda t: t[1])[1]

        newposlist = []

        for pos in poslist:
            nposx = ((pos[0] - minx) / (maxx - minx)) * 0.9 + 0.05
            nposy = ((pos[1] - miny) / (maxy - miny)) * 0.9 + 0.05
            newposlist.append((nposx, nposy))

        return newposlist


class NodeVision(object):
    """
    This object is used for laying out the nodes of a graph according to \
    the local vision of a specific node (root node)

    TODO: Put create_directed_graph out of the class.
          Receive graph as an input.
    """

    def __init__(self, n_nodes, rootnode=0):
        self.n_nodes = n_nodes
        self.graph = self.create_directed_graph(n_nodes)
        self.rootnode = rootnode
        self.bfstree = {}
        self.pos = self.lay_down_nodes()

    def set_root_node(self, rootnode):
        self.rootnode = rootnode
        self.pos = self.lay_down_nodes()

    def create_directed_graph(self, n_nodes, min_w=10, max_w=100):
        G = nx.random_k_out_graph(n_nodes, 3, 0.9)
        Gw = nx.DiGraph()
        for edge in G.edges():
            Gw.add_edge(edge[0], edge[1],
                        weight=np.random.uniform(min_w, max_w))
        return Gw

    def lay_down_nodes(self):
        H = self.graph.to_undirected()
        bfstree = nx.bfs_tree(H, self.rootnode)
        self.bfstree[self.rootnode] = bfstree
        # bfs = list(nx.bfs_tree(H, rootnode).edges())
        pos = GraphPositioning.hierarchy_pos(bfstree, self.rootnode,
                                             width=2*math.pi, xcenter=0)
        new_pos = {u: (r*math.cos(theta), r*math.sin(theta))
                   for u, (theta, r) in pos.items()}
        return new_pos

    def show_undirected_bfs_tree(self):
        nx.draw(self.bfstree[self.rootnode], pos=self.pos, node_size=50)
        plt.show()

    def show_directed_neighborhood(self):
        nx.draw_networkx_nodes(self.graph, pos=self.pos, node_size=50)
        nx.draw_networkx_nodes(self.graph, pos=self.pos,
                               nodelist=[self.rootnode],
                               node_color='blue', node_size=100)
        nx.draw_networkx_edges(self.graph, pos=self.pos, edge_color='gray',
                               alpha=0.5, style='dashed')
        plt.show()


class GraphPositioning(object):
    """
    This class is for the calculation of the positions of the nodes of
    a given tree and from the perspective of a given central node
    """

    @staticmethod
    def hierarchy_pos(G, root=None, width=1., vert_gap=0.2,
                      vert_loc=0, xcenter=0.5):
        """
        Taken from: https://bit.ly/2tetWxf

        If the graph is a tree this will return the positions to plot this in a
        hierarchical layout.

        G: the graph (must be a tree)

        root: the root node of current branch
        - if the tree is directed and this is not given,
          the root will be found and used
        - if the tree is directed and this is given, then the positions
          will be just for the descendants of this node.
        - if the tree is undirected and not given, then a random
          choice will be used.

        width: horizontal space allocated for this branch - avoids overlap
          with other branches

        vert_gap: gap between levels of hierarchy

        vert_loc: vertical location of root

        xcenter: horizontal location of root
        """

        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a'
                            ' graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                # allows back compatibility with nx version 1.11
                root = next(iter(nx.topological_sort(G)))
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0,
                           xcenter=0.5, pos=None, parent=None):
            """
            see hierarchy_pos docstring for most arguments
            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            """

            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                dx = width/len(children)
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                         vert_loc=vert_loc-vert_gap,
                                         xcenter=nextx, pos=pos, parent=root)
            return pos

        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
