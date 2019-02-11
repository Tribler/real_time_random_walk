import networkx as nx
from random import choice, randint


class FakeGraph(object):
    '''
    This class is for generating network transactions in substitution
    for the tribler crawler. It first generates a random network.
    Then, when the it is needed, it generates a random traffic between
    randomly selected nodes. The number of transactions can be given as
    an input.
    '''

    def __init__(self, nodecount):
        self.nodecount = nodecount
        self.gr = nx.watts_strogatz_graph(nodecount, 5, 0.5)

    def generate_transactions(self, tr_count=10):
        transactions = []
        for t in range(tr_count):
            node1 = randint(0, self.nodecount-1)
            # print(self.gr.adj[node1].keys())
            node2 = choice(self.gr.adj[node1].keys())
            transactions.append({'uploader': node1,
                                 'downloader': node2,
                                 'amount': randint(50, 100)})
        return transactions


fg = FakeGraph(100)
print(fg.generate_transactions(5))
