import FakeNetwork as fk
import networkx as nx
from NodeVision import NodeVision


class TransactionDiscovery(object):

    def __init__(self, fake=False, nodecount=300):
        if fake:
            self.generate_fake_network(nodecount)

    def generate_fake_network(self, nodecount):
        self.nodecount = nodecount
        self.fg = fk.FakeGraph(self.nodecount)

        gr = nx.DiGraph()
        gr.add_node(0)
        self.Gw = NodeVision(gr=gr)

        for i in range(1, nodecount, 30):
            self.Gw.graph.add_edge(0, i, weight=1.0)

        for i in range(12, nodecount, 100):
            self.Gw.graph.add_edge(i, 0, weight=1.0)

    def read_transactions(self, fake=True, tr_count=50):
        if fake:
            glob_trs = self.fg.generate_transactions(tr_count)
            self.Gw.add_transactions(glob_trs)
            loc_trs = self.fg.generate_local_transactions(self.Gw.rootnode, 5)
            self.Gw.add_transactions(loc_trs)
            return glob_trs + loc_trs
