from random import random

from NodeVision import NodeVision
from RandomWalk import RandomWalk
from TransactionDiscovery import CrawlerTransactionDiscovery, SQLiteTransactionDiscovery

DB_NAME = ""


def run_on_real_crawler(local_db=False):
    # Use default crawler transaction discovery
    if local_db:
        disc = SQLiteTransactionDiscovery(DB_NAME)
    else:
        disc = CrawlerTransactionDiscovery()

    transactions = disc.read_transactions(tr_count=100)

    # Add crawled transactions to the graph
    gw = NodeVision()
    for tr in transactions:
        gw.graph.add_edge(tr['downloader'],
                          tr['uploader'],
                          weight=tr['amount'])
        if random() < 0.25 and tr['downloader'] != gw.root_node:
            gw.graph.add_edge(gw.root_node, tr['downloader'], weight=tr['amount'])

    gw.set_root_node(transactions[0]['downloader'])
    # Initialization

    gw.normalize_edge_weights()

    gw.reposition_nodes()
    gw.show_undirected_bfs_tree()
    gw.update_component()
    gw.show_directed_neighborhood()

    rw = RandomWalk(gw, disc)
    rw.set_walk_params({'n_walk': 50, 'reset_prob': 0.1, 'n_step': 300})
    rw.set_move_params({'time_to_finish': 10})

    rw.make_fake_transactions = True

    rw.show_walk()


if __name__ == '__main__':
    run_on_real_crawler()

# def step(rw):
#     # Gw.diminish_weights()
#     trs = fg.generate_transactions(500)
#     Gw.add_transactions(trs)
#     trs = fg.generate_local_transactions(Gw.root_node, 5, 0.8, True)
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
