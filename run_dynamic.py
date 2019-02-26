from RandomWalk import RandomWalk
from NodeVision import NodeVision
import FakeNetwork as fk
from networkx import nx

nodecount = 300
fg = fk.FakeGraph(nodecount)

gr = nx.DiGraph()
gr.add_node(0)
Gw = NodeVision(gr=gr)

for i in range(1, nodecount, 30):
    Gw.graph.add_edge(0, i, weight=1.0)

for i in range(12, nodecount, 100):
    Gw.graph.add_edge(i, 0, weight=1.0)


# Initialization

trs = fg.generate_transactions(500)
Gw.add_transactions(trs)
trs = fg.generate_local_transactions(Gw.rootnode, 5)
Gw.add_transactions(trs)

Gw.normalize_edge_weights()

Gw.reposition_nodes()
Gw.show_undirected_bfs_tree()
Gw.update_component()
Gw.show_directed_neighborhood()

rw = RandomWalk(Gw, fake=True)
rw.set_walk_params({'n_walk': 10, 'reset_prob': 0.1, 'n_step': 300})
rw.set_move_params({'time_to_finish': 10})

rw.show_walk()


# def step(rw):
#     # Gw.diminish_weights()
#     trs = fg.generate_transactions(500)
#     Gw.add_transactions(trs)
#     trs = fg.generate_local_transactions(Gw.rootnode, 5, 0.8, True)
#     Gw.add_transactions(trs)

#     Gw.normalize_edge_weights()
#     Gw.reposition_nodes()
#     Gw.update_component()

#     rw.update_local_vision(Gw, animate=True)

#     rw.show_walk()

#     rw.remove_old_nodes(0.9)


# for i in range(1):
#     step(rw)

# Gw = NodeVision(n_nodes=400)
# Gw.show_undirected_bfs_tree()
# Gw.show_directed_neighborhood()

# Gw.show_undirected_bfs_tree()
# Gw.show_directed_neighborhood()

# rw = RandomWalk(Gw)
# rw.set_walk_params({'n_walk': 100, 'reset_prob': 0.1, 'n_step': 300})

# rw.show_walk()
