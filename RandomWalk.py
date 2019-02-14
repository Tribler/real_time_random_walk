import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.lines as mtline
from matplotlib.animation import FuncAnimation
import networkx as nx
import math
import random


class RandomWalk(object):

    def __init__(self, sample_graph):
        self.local_vision = sample_graph

        self.attr = {}
        self.attr['size'] = {}
        self.attr['rank'] = {}
        self.attr['color'] = {}

        self.attr['size'] = {nodeid: 100 for nodeid in self.gr.nodes()}
        self.attr['color'] = {nodeid: (1, 0, 0, 0.3)
                              for nodeid in self.gr.nodes()}

        self.pos = sample_graph.node_positions
        self.normal_pos = self.normalize_positions_dict()

        self.lines = []
        self.scat = None
        self.ax = None
        self.walk_params = self.default_walk_params()
        self.move_params = self.default_move_params()
        self.edge_indices = {}
        self.oldpos = None
        self.fig = None

    @property
    def gr(self):
        return self.local_vision.graph

    @property
    def component(self):
        return self.local_vision.component

    @property
    def node_sizes(self):
        return [self.attr['size'][nodeid]
                for nodeid in self.component.nodes()]

    @property
    def node_colors(self):
        return [self.attr['color'][nodeid]
                for nodeid in self.component.nodes()]

    def update_local_vision(self, sample_graph, animate=False):
        self.local_vision = sample_graph
        if not animate:
            self.pos = sample_graph.node_positions
            self.normal_pos = self.normalize_positions_dict()
            for nodeid in self.gr.nodes():
                if nodeid not in self.attr['size']:
                    self.attr['size'][nodeid] = 100
                    self.attr['color'][nodeid] = (1.0, 0.0, 0.0, 0.3)
        else:
            self.oldpos = dict(self.normal_pos)
            self.pos = sample_graph.node_positions
            self.normal_pos = self.normalize_positions_dict()
            for nodeid in self.gr.nodes():
                if nodeid not in self.attr['size']:
                    self.attr['size'][nodeid] = 100
                    self.attr['color'][nodeid] = (1.0, 0.0, 0.0, 0.3)
            self.move_anim()

    def move_anim(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        self.ax.set_xlim(0, 1), self.ax.set_xticks([])
        self.ax.set_ylim(0, 1), self.ax.set_yticks([])

        self.animation = FuncAnimation(self.fig, self.move_anim_update,
                                       interval=10,
                                       init_func=self.move_anim_init)
        plt.show()

    def move_anim_init(self):
        frame_number = 0
        actual_pos = {}

        for nodeid in self.gr.nodes():
            if nodeid not in self.pos.keys() or self.pos[nodeid] is None:
                continue
            if nodeid not in self.oldpos.keys():
                actual_pos[nodeid] = (0.9
                                      + (frame_number
                                         * (self.normal_pos[nodeid][0] - 0.9)
                                         / self.move_params['time_to_finish']),
                                      0.9
                                      + (frame_number
                                         * (self.normal_pos[nodeid][1] - 0.9)
                                         / self.move_params['time_to_finish']))
            else:
                actual_pos[nodeid] = (self.oldpos[nodeid][0]
                                      + (frame_number
                                         * (self.normal_pos[nodeid][0]
                                            - self.oldpos[nodeid][0])
                                         / self.move_params['time_to_finish']),
                                      self.oldpos[nodeid][1]
                                      + (frame_number
                                         * (self.normal_pos[nodeid][1]
                                            - self.oldpos[nodeid][1])
                                         / self.move_params['time_to_finish']))

        # Draw edges
        x1s, x2s, y1s, y2s, lws = [], [], [], [], []

        # print('Setting edge positions')
        print(len(self.local_vision.component.edges))
        for edge in self.local_vision.component.edges:
            x1s.append(actual_pos[edge[0]][0])
            y1s.append(actual_pos[edge[0]][1])
            x2s.append(actual_pos[edge[1]][0])
            y2s.append(actual_pos[edge[1]][1])
            lws.append(self.gr[edge[0]][edge[1]]['weight'])

        # print('Drawing edges')
        self.lines = self.ax.plot([x1s, x2s], [y1s, y2s], color='gray',
                                  alpha=0.4, linestyle='--', lw=0.5)

        # print('Setting edge attributes')
        for i in range(len(self.lines)):
            self.lines[i].set_xdata([x1s[i], x2s[i]])
            self.lines[i].set_ydata([y1s[i], y2s[i]])
            self.lines[i].set_linewidth(lws[i])
            self.lines[i].set_ls('dashed')

        # Draw nodes
        x_pos = [actual_pos[nodeid][0]
                 for nodeid in self.component.nodes()]
        y_pos = [actual_pos[nodeid][1]
                 for nodeid in self.component.nodes()]
        sizes = [self.attr['size'][nodeid]
                 for nodeid in self.component.nodes()]

        self.scat = self.ax.scatter(x_pos, y_pos,
                                    s=sizes, lw=0.5,
                                    edgecolors=self.node_colors,
                                    facecolors=self.node_colors)

    def move_anim_update(self, frame_number):
        actual_pos = {}

        if frame_number > self.move_params['time_to_finish']:
            return

        for nodeid in self.gr.nodes():
            if (((nodeid not in self.normal_pos.keys())
                 or (self.pos[nodeid] is None))):
                continue

            if nodeid not in self.oldpos.keys():
                actual_pos[nodeid] = (0.9
                                      + (frame_number
                                         * (self.normal_pos[nodeid][0] - 0.9)
                                         / self.move_params['time_to_finish']),
                                      0.9
                                      + (frame_number
                                         * (self.normal_pos[nodeid][1] - 0.9)
                                         / self.move_params['time_to_finish']))
            else:
                actual_pos[nodeid] = (self.oldpos[nodeid][0]
                                      + (frame_number
                                         * (self.normal_pos[nodeid][0]
                                            - self.oldpos[nodeid][0])
                                         / self.move_params['time_to_finish']),
                                      self.oldpos[nodeid][1]
                                      + (frame_number
                                         * (self.normal_pos[nodeid][1]
                                            - self.oldpos[nodeid][1])
                                         / self.move_params['time_to_finish']))

        # Draw edges
        x1s, x2s, y1s, y2s, lws = [], [], [], [], []

        # print('Setting edge positions')
        for edge in self.local_vision.component.edges:
            x1s.append(actual_pos[edge[0]][0])
            y1s.append(actual_pos[edge[0]][1])
            x2s.append(actual_pos[edge[1]][0])
            y2s.append(actual_pos[edge[1]][1])
            lws.append(self.gr[edge[0]][edge[1]]['weight'])

        # print('Drawing edges')

        # print('Setting edge attributes')
        for i in range(len(self.lines)):
            self.lines[i].set_xdata([x1s[i], x2s[i]])
            self.lines[i].set_ydata([y1s[i], y2s[i]])
            self.lines[i].set_linewidth(lws[i])
            self.lines[i].set_ls('dashed')

        # Draw nodes
        # print('Drawing nodes')
        posits = [actual_pos[nodeid]
                  for nodeid in self.component.nodes()]

        self.scat.set_offsets(posits)

    def default_walk_params(self):
        return {'growthrate': 30.0,
                'n_walk': 300,
                'n_step': 300,
                'reset_prob': 0.1}

    def default_move_params(self):
        return {'time_to_finish': 30}

    def set_walk_params(self, params):
        if 'growthrate' in params:
            self.walk_params['growthrate'] = params['growthrate']
        if 'n_walk' in params:
            self.walk_params['n_walk'] = params['n_walk']
        if 'n_step' in params:
            self.walk_params['n_step'] = params['n_step']
        if 'reset_prob' in params:
            self.walk_params['reset_prob'] = params['reset_prob']

    def set_move_params(self, params):
        if 'time_to_finish' in params:
            self.move_params['time_to_finish'] = params['time_to_finish']

    def remove_old_nodes(self, remove_prob=0.1):
        for node in self.component:
            if self.attr['size'][node] == 50:
                if random.random() < remove_prob:
                    self.local_vision.graph.remove_node(node)
                    print("Node {} is removed from the graph".format(node))

    def apply_function_to_attr(self, attrname, f):
        for nodeid in self.attr[attrname].keys():
            self.attr[attrname][nodeid] = f(self.attr[attrname][nodeid])

    def show_walk(self, savevid=False):
        self.fig = plt.figure(figsize=(10, 10))
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 1), ax.set_xticks([])
        ax.set_ylim(0, 1), ax.set_yticks([])

        self.normal_pos = self.normalize_positions_dict()

        # Draw edges
        x1s, x2s, y1s, y2s, lws = [], [], [], [], []

        # print('Setting edge positions')
        for edge in self.component.edges:
            x1s.append(self.normal_pos[edge[0]][0])
            y1s.append(self.normal_pos[edge[0]][1])
            x2s.append(self.normal_pos[edge[1]][0])
            y2s.append(self.normal_pos[edge[1]][1])
            lws.append(self.gr[edge[0]][edge[1]]['weight'])

        # print('Drawing edges')
        self.lines = ax.plot([x1s, x2s], [y1s, y2s], color='gray',
                             alpha=0.4, linestyle='--', lw=0.5)

        # print('Setting edge attributes')
        for i in range(len(self.lines)):
            self.lines[i].set_xdata([x1s[i], x2s[i]])
            self.lines[i].set_ydata([y1s[i], y2s[i]])
            self.lines[i].set_linewidth(lws[i])
            self.lines[i].set_ls('dashed')

        # Draw nodes
        # print('Drawing nodes')
        x_pos = [self.normal_pos[nodeid][0]
                 for nodeid in self.component.nodes()]
        y_pos = [self.normal_pos[nodeid][1]
                 for nodeid in self.component.nodes()]
        sizes = [self.attr['size'][nodeid]
                 for nodeid in self.component.nodes()]

        self.scat = ax.scatter(x_pos, y_pos,
                               s=sizes, lw=0.5,
                               edgecolors=self.node_colors,
                               facecolors=self.node_colors)

        self.current_node = self.local_vision.rootnode
        self.growthrate = self.walk_params['growthrate']
        self.n_walk = self.walk_params['n_walk']
        self.n_step = self.walk_params['n_step']
        self.reset_prob = self.walk_params['reset_prob']
        self.visiteds = []
        self.edge_indices = {}

        self.animation = FuncAnimation(self.fig, self.animation_update,
                                       interval=10,
                                       init_func=self.animation_init)

        if savevid:
            # self.animation.save('anim.gif', dpi=60, writer='imagemagick')
            self.animation.save('anim.mp4')
        else:
            plt.show()

    def make_step(self, nodeid, visiteds):
        self.n_step -= 1
        if (((self.component.out_degree(nodeid) == 0)
             or (np.random.uniform(0, 1) < self.reset_prob))):

            return self.local_vision.rootnode, []

        total_weight = 0
        for neigh in self.component.successors(nodeid):
            total_weight += self.gr[nodeid][neigh]['weight']
            # print('{} - {} '.format(neigh, self.gr[nodeid][neigh]['weight']))

        prob = np.random.uniform(0, total_weight)

        total_weight = 0
        for neigh in self.component.neighbors(nodeid):
            total_weight += self.gr[nodeid][neigh]['weight']
            if prob < total_weight:
                if neigh in visiteds:
                    return self.local_vision.rootnode, []
                visiteds.append(neigh)
                return neigh, visiteds

    def animation_init(self):

        self.attr['color'][self.local_vision.rootnode] = (0.1, 0.8, 0.1, 0.7)

        x1s, x2s, y1s, y2s, lws, lcs, lss = [], [], [], [], [], [], []

        curr_index = 0
        for edge in self.component.edges:
            x1s.append(self.normal_pos[edge[0]][0])
            y1s.append(self.normal_pos[edge[0]][1])
            x2s.append(self.normal_pos[edge[1]][0])
            y2s.append(self.normal_pos[edge[1]][1])

            lcs.append('gray')
            lws.append(self.gr[edge[0]][edge[1]]['weight'])
            lss.append('--')

            self.edge_indices[(edge[0], edge[1])] = curr_index
            # self.edge_indices[(edge[1], edge[0])] = curr_index
            curr_index += 1

        for i in range(len(self.lines)):
            self.lines[i].set_xdata([x1s[i], x2s[i]])
            self.lines[i].set_ydata([y1s[i], y2s[i]])
            self.lines[i].set_lw(lws[i])
            self.lines[i].set_color(lcs[i])
            self.lines[i].set_ls(lss[i])

    def animation_update(self, frame_number):
        # self.current_index = frame_number % self.n_nodes

        print('Walk', self.n_walk)
        if self.n_walk < 1:
            self.animation.event_source.stop()
            return

        self.next_node, self.visiteds = self.make_step(self.current_node,
                                                       self.visiteds)

        if self.next_node == self.local_vision.rootnode:
            self.n_walk -= 1
            self.n_step = 300

        # Update node sizes
        if self.current_node != self.local_vision.rootnode:
            self.attr['size'][self.current_node] += self.growthrate
        self.apply_function_to_attr('size',
                                    f=lambda x:
                                    max(50, x-(self.growthrate
                                               * 0.01
                                               * random.random()
                                               - 0.002)))
        print('minsize', min(list(self.attr['size'].values())))
        self.scat.set_sizes(self.node_sizes)

        # Update node colors
        self.attr['color'][self.next_node] = (0, 1, 0, 1)
        self.attr['color'][self.local_vision.rootnode] = (0, 1, 1, 1)
        for node in self.local_vision.graph.nodes():
            if self.attr['size'][node] == 50:
                self.attr['color'][node] = (1, 1, 0, 1)
                
        self.scat.set_facecolor(self.node_colors)

        # self.scat.set_edgecolors(self.nodes['color'])
        # self.scat.set_offsets(self.nodes['position'])

        lws, lcs, lss = [], [], []
        if len(self.visiteds) == 0:
            # Reset colors
            print('Starting new walk')
            lws, lcs, lss = [], [], []
            for edge in self.component.edges:
                lcs.append('gray')
                lws.append(self.gr[edge[0]][edge[1]]['weight'])
                lss.append('--')

            # print('Setting new lines'),
            for i in range(len(self.lines)):
                self.lines[i].set_lw(lws[i])
                self.lines[i].set_color(lcs[i])
                self.lines[i].set_ls(lss[i])
            # print('... Finished')
        else:
            # print('Arranging new edge'),
            ind = self.edge_indices[(self.current_node, self.next_node)]
            self.lines[ind].set_color('b')
            self.lines[ind].set_lw(4)
            self.lines[ind].set_ls('--')
            # print('... Finished')

        self.current_node = self.next_node

    def normalize_positions(self):
        poslist = [v for v in self.pos.values() if v is not None]
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

    def normalize_positions_dict(self):
        poslist = [v for v in self.pos.values() if v is not None]
        minx = min(poslist, key=lambda t: t[0])[0]
        miny = min(poslist, key=lambda t: t[1])[1]
        maxx = max(poslist, key=lambda t: t[0])[0]
        maxy = max(poslist, key=lambda t: t[1])[1]

        newposlist = {}

        for node, pos in self.pos.items():
            if pos is not None:
                nposx = ((pos[0] - minx) / (maxx - minx)) * 0.9 + 0.05
                nposy = ((pos[1] - miny) / (maxy - miny)) * 0.9 + 0.05
                newposlist[node] = (nposx, nposy)

        return newposlist


class NodeVision(object):
    """
    This object is used for laying out the nodes of a graph according to \
    the local vision of a specific node (root node)

    TODO: Put create_directed_graph out of the class.
          Receive graph as an input.
    """

    def __init__(self, gr=None, n_nodes=0, rootnode=0):
        if gr is None:
            self._n_nodes = n_nodes
            self.graph = self.create_directed_graph(n_nodes)
        else:
            self.graph = gr
            self._n_nodes = gr.number_of_nodes()
        self.rootnode = rootnode
        self.bfstree = {}
        self.pos = self.lay_down_nodes()
        self.component = None

    def set_root_node(self, rootnode):
        self.rootnode = rootnode
        self.pos = self.lay_down_nodes()

    @property
    def n_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def node_positions(self):
        return dict(self.graph.nodes(data='pos'))

    def create_directed_graph(self, n_nodes, min_w=10, max_w=100):
        G = nx.random_k_out_graph(n_nodes, 3, 0.9)
        Gw = nx.DiGraph()
        for edge in G.edges():
            Gw.add_edge(edge[0], edge[1],
                        weight=np.random.uniform(min_w, max_w))
        return Gw

    def reposition_nodes(self):
        self.pos = self.lay_down_nodes()

    def add_transactions(self, transactions):
        for tr in transactions:
            # Gw.graph.add_edge(tr['downloader'],
            #                  tr['uploader'], weight=tr['amount'])
            self.add_edge_to_vision(tr['downloader'],
                                    tr['uploader'], tr['amount'])

    def add_edge_to_vision(self, n1, n2, w):
        if n1 in self.graph and n2 in self.graph.successors(n1):
            self.graph[n1][n2]['weight'] += w
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
        component_nodes = nx.node_connected_component(H, self.rootnode)
        for node in list(H.nodes()):
            if node not in component_nodes:
                H.remove_node(node)

        bfstree = nx.bfs_tree(H, self.rootnode)
        self.bfstree[self.rootnode] = bfstree
        # bfs = list(nx.bfs_tree(H, rootnode).edges())
        pos = GraphPositioning.hierarchy_pos(bfstree, self.rootnode,
                                             width=2*math.pi, xcenter=0)
        new_pos = {u: (r*math.cos(theta), r*math.sin(theta))
                   for u, (theta, r) in pos.items()}
        # for u, (x, y) in new_pos.items():
        #     self.graph[u]['pos'] = (x, y)
        nx.set_node_attributes(self.graph, new_pos, 'pos')
        return new_pos

    def show_undirected_bfs_tree(self):
        # nx.draw(self.bfstree[self.rootnode], pos=self.pos, node_size=50)
        nx.draw(self.bfstree[self.rootnode],
                pos=self.node_positions,
                node_size=50)
        plt.show()

    def update_component(self):
        H = self.graph.to_undirected()
        self.component = nx.DiGraph(self.graph)

        component_nodes = nx.node_connected_component(H, self.rootnode)
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
                               nodelist=[self.rootnode],
                               node_color='blue', node_size=100)
        nx.draw_networkx_edges(self.component, pos=self.node_positions,
                               edge_color='gray', alpha=0.5, style='dashed')
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
