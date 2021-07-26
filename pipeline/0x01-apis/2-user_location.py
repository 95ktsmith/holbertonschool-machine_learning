#!/usr/bin/env python3
""" Github user location """
import requests
import sys
import time


if __name__ == "__main__":

    url = sys.argv[1]
    data = requests.get(url)

    if data.status_code == 404:
        print("Not found")

    elif data.status_code == 403:
        reset_time = int(data.headers['X-RateLimit-Reset'])
        current_time = int(time.time())
        minutes_until_reset = int((reset_time - current_time) / 60)
        print("Reset in {} min".format(minutes_until_reset))

    else:
        print(data.json()['location'])
