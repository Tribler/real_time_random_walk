import networkx as nx
from random import choice, randint, random


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
                                 'amount': 50 + random() * 50})
        return transactions

    def generate_local_transactions(self, local_node, tr_count=1,
                                    upload_bias_prob=0.5, new_nodes=False):
        transactions = []
        for t in range(tr_count):
            node1 = local_node
            # print(self.gr.adj[node1].keys())
            node2 = choice(list(self.gr.nodes()))
            if not new_nodes:
                node2 = choice(self.gr.adj[node1].keys())

            if random() < upload_bias_prob:
                transactions.append({'uploader': node1,
                                     'downloader': node2,
                                     'amount': 50 + random() * 50})
            else:
                transactions.append({'uploader': node2,
                                     'downloader': node1,
                                     'amount': 50 + random() * 50})
        return transactions


fg = FakeGraph(100)
print(fg.generate_transactions(5))
