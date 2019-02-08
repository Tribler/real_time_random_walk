from RandomWalk import NodeVision, RandomWalk

Gw = NodeVision(400)
Gw.show_undirected_bfs_tree()
Gw.show_directed_neighborhood()

rw = RandomWalk(Gw)
rw.set_walk_params({'n_walk': 300, 'reset_prob': 0.1, 'n_step': 300})
rw.show_walk()