#!/usr/bin/env python3
""" Rocket frequency """
import requests

if __name__ == "__main__":
    launches = requests.get("https://api.spacexdata.com/v5/launches").json()
    rockets = {}
    rocketURL = "https://api.spacexdata.com/v4/rockets/"
    for launch in launches:
        rocketData = requests.get(rocketURL + launch['rocket']).json()
        if rocketData['name'] in rockets.keys():
            rockets[rocketData['name']] += 1
        else:
            rockets[rocketData['name']] = 1

    rockets_list = [(k, v) for k, v in rockets.items()]
    sorted_list = sorted(rockets_list, key=lambda x: x[1], reverse=True)

    for rocket in sorted_list:
        print("{}: {}".format(rocket[0], rocket[1]))
