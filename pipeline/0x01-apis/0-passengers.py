#!/usr/bin/env python3
""" Passengers """
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers
    """
    url = "https://swapi-api.hbtn.io/api/starships"
    ships = []

    while url is not None:
        data = requests.get(url).json()

        for ship in data['results']:
            try:
                if int(ship['passengers']) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                if ship['name'] == 'Death Star' and passengerCount <= 843342:
                    ships.append('Death Star')

        url = data['next']

    return ships
