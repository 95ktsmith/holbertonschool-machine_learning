#!/usr/bin/env python3
""" Sentient Planets """
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/people"
    planets = []
    checkedSpecies = []

    while url is not None:
        data = requests.get(url).json()

        for person in data['results']:
            for species in person['species']:
                if species not in checkedSpecies:
                    checkedSpecies.append(species)
                    speciesData = requests.get(species).json()
                    if speciesData['designation'] == 'sentient':
                        homeWorld = requests.get(person['homeworld']).json()
                        if homeWorld['name'] not in planets:
                            planets.append(homeWorld['name'])
                            if homeWorld['name'] == 'Tatooine':
                                print(person['name'])
                                print(speciesData['name'])
                                return []

        url = data['next']

    return planets
