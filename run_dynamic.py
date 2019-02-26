from FakeNetwork import GraphTransactionGenerator
from NodeVision import NodeVision
from RandomWalk import RandomWalk
from TransactionDiscovery import GeneratedTransactionDiscovery


def run_on_generated_data():
    # Transaction generator
    nodecount = 300
    fg = GraphTransactionGenerator(nodecount)

    # Init graph with root 0
    Gw = NodeVision()
    Gw.add_transactions(fg.generate_transactions(500))
    Gw.add_transactions(fg.generate_local_transactions(Gw.root_node, 5))
    Gw.normalize_edge_weights()
    Gw.reposition_nodes()
    Gw.show_undirected_bfs_tree()
    Gw.update_component()
    Gw.show_directed_neighborhood()

    # Use default fake transaction discovery
    discoverer = GeneratedTransactionDiscovery()

    rw = RandomWalk(Gw, discoverer)
    rw.set_walk_params({'n_walk': 10, 'reset_prob': 0.1, 'n_step': 300})
    rw.set_move_params({'time_to_finish': 10})

    rw.show_walk()


if __name__ == '__main__':
    run_on_generated_data()

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
