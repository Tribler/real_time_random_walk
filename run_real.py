from RandomWalk import RandomWalk
from NodeVision import NodeVision
from networkx import nx
from TransactionDiscovery import TransactionDiscovery
from random import random

gr = nx.DiGraph()
gr.add_node(0)
Gw = NodeVision(gr=gr)

disc = TransactionDiscovery()
transactions = disc.read_transactions(fake=False, tr_count=500)

for tr in transactions:
    Gw.graph.add_edge(tr['downloader'],
                      tr['uploader'],
                      weight=tr['amount'])
    if random() < 0.25 and tr['downloader'] != Gw.rootnode:
        Gw.graph.add_edge(Gw.rootnode, tr['downloader'], weight=tr['amount'])
        # Gw.graph.add_edge(tr['downloader'], Gw.rootnode, weight=tr['amount'])

Gw.set_root_node(transactions[0]['downloader'])
# Initialization

Gw.normalize_edge_weights()

Gw.reposition_nodes()
Gw.show_undirected_bfs_tree()
Gw.update_component()
Gw.show_directed_neighborhood()

rw = RandomWalk(Gw)
rw.set_walk_params({'n_walk': 50, 'reset_prob': 0.1, 'n_step': 300})
rw.set_move_params({'time_to_finish': 10})

rw.make_fake_transactions = True

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
