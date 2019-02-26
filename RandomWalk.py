import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from TransactionDiscovery import TransactionDiscovery

anim_colors = {'init_color': (0, 0, 1, 0.3),
               'visited_color:': (0, 1, 0, 0.3),
               'explored_color': (1, 0, 1, 0.3),
               'forgotten_color': (0.3, 0.3, 0.3, 0.5)}

par_remove_size = 30
par_remove_prob = 0.5
par_init_size = 100


class RandomWalk(object):

    def __init__(self, sample_graph, discoverer):
        """
        :param discoverer: TransactionDiscorvery
        """
        self.local_vision = sample_graph

        self.attr = {'size': {nodeid: par_init_size
                              for nodeid in self.gr.nodes()},
                     'rank': {},
                     'color': {nodeid: anim_colors['init_color']
                               for nodeid in self.gr.nodes()}}

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
        self.discoverer = discoverer
        self.make_fake_transactions = False

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

    def animation_controller(self, status):
        if status == 'walkfinished':
            # Remove old nodes with some probability
            self.remove_old_nodes(par_remove_prob)

            # Discover new transactions
            trs = self.discoverer.read_transactions(100)
            self.local_vision.add_transactions(trs)
            if self.make_fake_transactions:
                self.local_vision.make_random_transactions(5)
            self.local_vision.normalize_edge_weights()
            self.local_vision.reposition_nodes()
            self.local_vision.update_component()
            self.update_local_vision(self.local_vision)
            self.n_walk = self.walk_params['n_walk']
            self.n_step = self.walk_params['n_step']

            # Restart animation manually
            print('Start move animation')
            self.animation._init_func = self.move_anim_init
            self.animation._func = self.move_anim_update
            self.frame_number = 0
            self.move_anim_init()

    def update_local_vision(self, sample_graph, animate=False):
        self.local_vision = sample_graph
        if not animate:
            self.oldpos = dict(self.normal_pos)
            self.pos = sample_graph.node_positions
            self.normal_pos = self.normalize_positions_dict()
            for nodeid in self.gr.nodes():
                if nodeid not in self.attr['size']:
                    self.attr['size'][nodeid] = par_init_size
                    self.attr['color'][nodeid] = anim_colors['explored_color']
        else:
            self.oldpos = dict(self.normal_pos)
            self.pos = sample_graph.node_positions
            self.normal_pos = self.normalize_positions_dict()
            for nodeid in self.gr.nodes():
                if nodeid not in self.attr['size']:
                    self.attr['size'][nodeid] = par_init_size
                    self.attr['color'][nodeid] = anim_colors['explored_color']
            self.move_anim()

    def move_anim(self):
        self.prepare_canvas()

        self.animation = FuncAnimation(self.fig, self.move_anim_update,
                                       interval=10,
                                       init_func=self.move_anim_init)
        plt.show()

    def move_anim_init(self):
        print('Executing move_anim_init')
        self.ax.cla()
        self.ax.set_xlim(0, 1), self.ax.set_xticks([])
        self.ax.set_ylim(0, 1), self.ax.set_yticks([])
        frame_number = 0
        self.frame_number = 0
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

    def move_anim_update(self, _frame_number):
        print('Executing move_anim_update')
        actual_pos = {}

        if self.frame_number > self.move_params['time_to_finish']:
            self.frame_number = 0
            self.animation._init_func = self.walk_anim_init
            self.animation._func = self.walk_anim_update
            self.walk_anim_init()            
            return

        for nodeid in self.gr.nodes():
            if (((nodeid not in self.normal_pos.keys())
                 or (self.pos[nodeid] is None))):
                continue

            if nodeid not in self.oldpos.keys():
                actual_pos[nodeid] = (0.9
                                      + (self.frame_number
                                         * (self.normal_pos[nodeid][0] - 0.9)
                                         / self.move_params['time_to_finish']),
                                      0.9
                                      + (self.frame_number
                                         * (self.normal_pos[nodeid][1] - 0.9)
                                         / self.move_params['time_to_finish']))
            else:
                actual_pos[nodeid] = (self.oldpos[nodeid][0]
                                      + (self.frame_number
                                         * (self.normal_pos[nodeid][0]
                                            - self.oldpos[nodeid][0])
                                         / self.move_params['time_to_finish']),
                                      self.oldpos[nodeid][1]
                                      + (self.frame_number
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
        self.frame_number += 1

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
            if self.attr['size'][node] == par_remove_size:
                if random.random() < remove_prob:
                    self.local_vision.graph.remove_node(node)
                    print("Node {} is removed from the graph".format(node))

    def apply_function_to_attr(self, attrname, f):
        for nodeid in self.attr[attrname].keys():
            self.attr[attrname][nodeid] = f(self.attr[attrname][nodeid])

    def test(self, event):
        self.n_step = 0
        self.n_walk = 0
        print('removedremovedremoved')

    def prepare_canvas(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_axes([0.025, 0.1, 1, 1], frameon=False)
        self.ax.set_xlim(0, 1), self.ax.set_xticks([])
        self.ax.set_ylim(0, 1), self.ax.set_yticks([])
        # self.ax2 = self.fig.add_axes([0.6, 0.05, 0.09, 0.05])
        # self.ax3 = self.fig.add_axes([0.72, 0.05, 0.09, 0.05])
        
        # bprev = Button(self.ax2, 'Interrupt Walk')
        # tprev = Button(self.ax3, 'T')
        # bprev.on_clicked(self.test)
        # tprev.on_clicked(self.test)

    def show_walk(self, savevid=False):
        self.prepare_canvas()
        # bprev = Button(self.ax2, 'Interrupt Walk')
        # bprev.on_clicked(self.test)
        # tprev = Button(self.ax3, 'T')
        # tprev.on_clicked(self.test)

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
        self.lines = self.ax.plot([x1s, x2s], [y1s, y2s], color='gray',
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

        self.scat = self.ax.scatter(x_pos, y_pos,
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

        self.animation = FuncAnimation(self.fig, self.walk_anim_update,
                                       interval=100,
                                       init_func=self.walk_anim_init)

        if savevid:
            self.animation.save('anim.gif', dpi=60, writer='imagemagick')
            # self.animation.save('anim.mp4')
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

    def walk_anim_init(self):

        self.attr['color'][self.local_vision.rootnode] = (0.3, 0.8, 0.3, 0.7)

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

    def walk_anim_update(self, frame_number):
        # self.current_index = frame_number % self.n_nodes

        print('Walk', self.n_walk)
        if self.n_walk < 1:
            # self.animation.event_source.stop()
            self.animation_controller('walkfinished')
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
                                    max(par_remove_size, x-(self.growthrate
                                                            * 0.02
                                                            * random.random()
                                                            - 0.002)))
        print('minsize', min(list(self.attr['size'].values())))
        self.scat.set_sizes(self.node_sizes)

        # Update node colors
        self.attr['color'][self.next_node] = (0, 1, 0, 1)
        self.attr['color'][self.local_vision.rootnode] = (0, 1, 1, 1)
        for node in self.local_vision.graph.nodes():
            if self.attr['size'][node] == par_remove_size:
                self.attr['color'][node] = anim_colors['forgotten_color']

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

    def normalize_positions_dict(self, width=0.80, margin=0.05):
        poslist = [v for v in self.pos.values() if v is not None]
        minx = min(poslist, key=lambda t: t[0])[0]
        miny = min(poslist, key=lambda t: t[1])[1]
        maxx = max(poslist, key=lambda t: t[0])[0]
        maxy = max(poslist, key=lambda t: t[1])[1]

        newposlist = {}

        for node, pos in self.pos.items():
            if pos is not None:
                nposx = ((pos[0] - minx) / (maxx - minx)) * width + margin
                nposy = ((pos[1] - miny) / (maxy - miny)) * width + margin
                newposlist[node] = (nposx, nposy)

        return newposlist
