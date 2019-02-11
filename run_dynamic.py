from RandomWalk import NodeVision, RandomWalk
import FakeNetwork as fk
from networkx import nx

nodecount = 150
fg = fk.FakeGraph(nodecount)

gr = nx.DiGraph()
gr.add_node(0)
Gw = NodeVision(gr=gr)


Gw.graph.add_edge(0, 5, weight=100)
Gw.graph.add_edge(0, 10, weight=100)
Gw.graph.add_edge(0, 45, weight=100)

Gw.graph.add_edge(50, 0, weight=100)
Gw.graph.add_edge(51, 0, weight=100)
Gw.graph.add_edge(52, 0, weight=100)


def step():
    trs = fg.generate_transactions(10)

    for tr in trs:
        # Gw.graph.add_edge(tr['downloader'],
        #                  tr['uploader'], weight=tr['amount'])
        Gw.add_edge_to_vision(tr['downloader'],
                              tr['uploader'], tr['amount'])

    Gw.reposition_nodes()
    Gw.show_undirected_bfs_tree()
    Gw.show_directed_neighborhood()


step()

# Gw = NodeVision(n_nodes=400)
# Gw.show_undirected_bfs_tree()
# Gw.show_directed_neighborhood()

print(Gw.graph.number_of_nodes())
print(Gw.graph.number_of_edges())

Gw.show_undirected_bfs_tree()
Gw.show_directed_neighborhood()

rw = RandomWalk(Gw)
rw.set_walk_params({'n_walk': 100, 'reset_prob': 0.1, 'n_step': 300})

rw.show_walk()
