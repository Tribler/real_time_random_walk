import FakeNetwork as fk
import networkx as nx
from NodeVision import NodeVision
import urllib
import json
from random import random 


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
        else:
            url = "http://localhost:8085/trustchain/recent?limit={}".format(tr_count * 100)
            response = urllib.urlopen(url)
            data = json.loads(response.read())['blocks']

            transactions = []
            for d in data:
                if random() > 0.01:
                    continue
                if int(d['transaction']['down']) > 0:
                    transactions.append({'downloader': d['public_key'],
                                         'uploader': d['link_public_key'],
                                         'amount': int(d['transaction']['down']) / 200000000.0})
                if int(d['transaction']['up']) > 0:
                    transactions.append({'downloader': d['link_public_key'],
                                         'uploader': d['public_key'],
                                         'amount': int(d['transaction']['up']) / 200000000.0})
            return transactions
