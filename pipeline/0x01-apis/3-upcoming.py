#!/usr/bin/env python3
""" Upcoming launch """
import requests


if __name__ == "__main__":

    data = requests.get("https://api.spacexdata.com/v5/launches/next").json()

    launch_name = data['name']
    date = data['date_local']
    rocketURL = "https://api.spacexdata.com/v4/rockets/" + data['rocket']
    padURL = "https://api.spacexdata.com/v4/launchpads/" + data['launchpad']

    rocketData = requests.get(rocketURL).json()
    rocketName = rocketData['name']

    padData = requests.get(padURL).json()
    padName = padData['name']
    padLoc = padData['locality']

    print("{} ({}) {} - {} ({})".format(
        launch_name,
        date,
        rocketName,
        padName,
        padLoc
    ))
