from TransactionDiscovery import SQLiteTransactionDiscovery


def run_db_crawl():
    db_path = "/Volumes/Elements/Databases/trustchain.db"
    disc = SQLiteTransactionDiscovery(db_path)
    tx_val = disc.read_transactions(10)
    for t in tx_val:
            print(t)

    tx_val = disc.read_transactions(10)
    for t in tx_val:
        print(t)

    disc.close_connection()


if __name__ == '__main__':
    run_db_crawl()
