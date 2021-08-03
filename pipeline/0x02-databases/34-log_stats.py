#!/usr/bin/env python3
""" Log stats """
from pymongo import MongoClient


if __name__ == "__main__":
    db = MongoClient()
    logs = db.logs.nginx

    print("{} logs".format(db.count()))
    print("Methods:")
    print("\tmethod GET: {}".format(
        db.count({"method": "GET"})
    ))
    print("\tmethod POST: {}".format(
        db.find({"method": "POST"}).count()
    ))
    print("\tmethod PUT: {}".format(
        db.find({"method": "PUT"}).count()
    ))
    print("\tmethod PATCH: {}".format(
        db.find({"method": "PATCH"}).count()
    ))
    print("\tmethod DELETE: {}".format(
        db.find({"method": "DELETE"}).count()
    ))
    print("{} status check".format(
        db.find({"method": "GET", "path": "/status"}).count()
    ))
