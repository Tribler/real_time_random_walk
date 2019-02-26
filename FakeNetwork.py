import networkx as nx
from random import choice, randint, random


class GraphTransactionGenerator(object):
    """
    This class is for generating network transactions in substitution
    for the tribler crawler. It first generates a random network.
    Then, when the it is needed, it generates a random traffic between
    randomly selected nodes. The number of transactions can be given as
    an input.
    """

    def __init__(self, node_count):
        """
        Generate Watts-Strogatz network
        :param node_count: Number of nodes in the graph
        """
        self.node_count = node_count
        self.gr = nx.watts_strogatz_graph(node_count, 5, 0.5)

    def generate_transactions(self, tr_count=10):
        """
        Generate some transactions in the graph between two random nodes
        :param tr_count:
        :return:
        """
        transactions = []
        for t in range(tr_count):
            node1 = randint(0, self.node_count-1)
            node2 = choice(self.gr.adj[node1].keys())
            transactions.append({'uploader': node1,
                                 'downloader': node2,
                                 'amount': 50 + random() * 50})
        return transactions

    def generate_local_transactions(self, local_node, tr_count=1,
                                    upload_bias_prob=0.5,
                                    by_topology=True):
        """
        Generate transaction between local and a random node
        :param local_node: Target node
        :param tr_count: Number of transactions to generate
        :param upload_bias_prob: Upload/download bias.
        The probability that local_node will be an uploader in a transaction
        :param by_topology: Are the nodes chosen according to the network topology?
        :return:
        """
        transactions = []
        for t in range(tr_count):
            node1 = local_node
            choose_set = self.gr.adj[node1].keys() if by_topology else list(self.gr.nodes())
            node2 = choice(choose_set)

            if random() < upload_bias_prob:
                transactions.append({'uploader': node1,
                                     'downloader': node2,
                                     'amount': 50 + random() * 50})
            else:
                transactions.append({'uploader': node2,
                                     'downloader': node1,
                                     'amount': 50 + random() * 50})
        return transactions
