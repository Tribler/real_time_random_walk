from abc import abstractmethod, ABCMeta
import urllib
import json
from random import random
from FakeNetwork import GraphTransactionGenerator
from NodeVision import NodeVision

import sqlite3
from sqlite3 import Error

from pyipv8.ipv8.attestation.trustchain.block import TrustChainBlock


class TransactionDiscovery(object):
    """
    Class for fetching transactions
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def read_transactions(self, tr_count=50):
        """
        Read transactions
        :param tr_count: Number of transactions
        :return: Transaction list
        """
        pass


class GeneratedTransactionDiscovery(TransactionDiscovery):

    def __init__(self, node_count=300):
        self.fg = GraphTransactionGenerator(node_count)
        self.Gw = NodeVision()
        # Add some initial transactions to the graph: 1 % of node count
        self.Gw.add_transactions(self.fg.generate_transactions(0.01 * node_count))

    def read_transactions(self, tr_count=50, local_percentage=1):
        """
        :param tr_count: Number of transaction to read
        :param local_percentage: Percent of transaction of tr_count from root node (from 0 to 100)
        :return:
        """
        local_txs = local_percentage * tr_count / 100
        glob_trs = self.fg.generate_transactions(tr_count - local_txs)
        self.Gw.add_transactions(glob_trs)
        loc_trs = self.fg.generate_local_transactions(self.Gw.root_node, local_txs)
        self.Gw.add_transactions(loc_trs)
        return glob_trs + loc_trs


class CrawlerTransactionDiscovery(TransactionDiscovery):

    def __init__(self, url="http://localhost:8085"):
        self.url = url + "/trustchain/recent?limit={}"

    def read_transactions(self, tr_count=50):
        url = self.url.format(tr_count * 100)
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


class SQLiteTransactionDiscovery(TransactionDiscovery):

    def create_connection(self):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            conn = sqlite3.connect(self.db_dir)
            return conn
        except Error as _:
            # Print error
            return None

    def close_connection(self):
        self.conn.close()

    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.conn = self.create_connection()
        self.offset = 0

    def read_transactions(self, tr_count=50):
        if self.conn is None:
            # possibly throw exception
            return None
        else:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM blocks "
                        "ORDER BY block_timestamp "
                        "LIMIT {},{}".format(self.offset, tr_count))
            self.offset += tr_count
            res = []
            for db_val in cur.fetchall():
                block = TrustChainBlock(db_val)
                if block.type == 'tribler_bandwidth':
                    if int(block.transaction['down']) > 0:
                        res.append({'downloader': block.public_key,
                                    'uploader': block.link_public_key,
                                    'amount': int(block.transaction['down']) / 200000000.0})
                    if int(block.transaction['up']) > 0:
                        res.append({'downloader': block.link_public_key,
                                    'uploader': block.public_key,
                                    'amount': int(block.transaction['up']) / 200000000.0})

            return res
