#!/usr/bin/env python3
""" Sentient Planets """
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species"
    planets = []

    while url is not None:
        data = requests.get(url).json()

        for species in data['results']:
            if (species['designation'] == 'sentient' or
                    species['classification'] == 'sentient') and\
                    species['homeworld'] is not None:
                planetData = requests.get(species['homeworld']).json()
                if planetData['name'] not in planets:
                    planets.append(planetData['name'])

        url = data['next']

    return planets
