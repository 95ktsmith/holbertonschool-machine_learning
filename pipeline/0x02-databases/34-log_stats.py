#!/usr/bin/env python3
""" Log stats """
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient()
    db = client.logs.nginx

    print("{} logs".format(db.count_documents({})))
    print("Methods:")
    print("\tmethod GET: {}".format(
        db.count_documents({"method": "GET"})
    ))
    print("\tmethod POST: {}".format(
        db.count_documents({"method": "POST"})
    ))
    print("\tmethod PUT: {}".format(
        db.count_documents({"method": "PUT"})
    ))
    print("\tmethod PATCH: {}".format(
        db.count_documents({"method": "PATCH"})
    ))
    print("\tmethod DELETE: {}".format(
        db.count_documents({"method": "DELETE"})
    ))
    print("{} status check".format(
        db.count_documents({"method": "GET", "path": "/status"})
    ))
