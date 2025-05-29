## Calculer et afficher une trajectoire d'orbite d'un satellite - exemple 9

import os
chemin = "/Users/nganle2911/Documents/2425_FRANCE/SCIENCES-DES-DONNEES/IUT-Perpignan-Carcassonne/BUT1_2425/Semestre_2/SAE/Project-statistique-astrophysique_R2-06/satellites-par-constellations/afficher-une-trajectoire"

os.chdir(chemin)

from skyfield.api import load, wgs84
from skyfield.api import EarthSatellite

# de421 donne la position des planètes dans l'espace
eph = load('de421.bsp')
earth = eph['earth']

# Définit un objet date d'observation en date UTC (temps universel)
ts = load.timescale()

# Réglage de l'heure d'observation
# C'est 1h de moins en hiver par rapport à l'heure locale en FR
Jour = 2
Mois = 3
Annee = 2025
Heure = 0
Minute = 0
Seconde = 0

# Lieu d'observation : latitude, longitude, altitude
latOBS = 45.0
longOBS = 0
altitudeOBS = 100

from skyfield.api import N,S,E,W
lieu = wgs84.latlon(latOBS, longOBS, elevation_m=altitudeOBS)
vecteurlieuobservation = earth + lieu

lemplacement = f"Emplacement : Latitude={latOBS} / Longitude={longOBS}"


########## Utilisation de la bibliothèque Hipparcos
## https://rhodesmill.org/skyfield/stars.html

from skyfield.data import hipparcos
# Le fichier de données Hipparcos contient la position des étoiles
f = load.open(hipparcos.URL)
stars = hipparcos.load_dataframe(f)
f.close()

from skyfield.api import Star
liste_etoiles = Star.from_dataframe(stars)

########## Constellations

from skyfield.api import position_of_radec, load_constellation_map, load_constellation_names
d = dict(load_constellation_names())

# Fonction de détermination de la constellation
constellation_at = load_constellation_map()

# Chargement depuis le disque
import json
f = open("constellationship.json", "rt")
constellations = json.load(f)
f.close()

###########################
print()
# Créer un objet itérable contenant toutes les étoiles
# de type catalogue skyfield.starlib.Star
from skyfield.api import Star
liste_etoiles = Star.from_dataframe(stars)


########### Définition d'une projection sphérique
# Set up stereographic projection
observateur = lieu.at(ts.utc(Annee, Mois, Jour, Heure, Minute, Seconde))
ra, dec, distance = observateur.radec()
center_object = Star(ra=ra, dec=dec)
center = earth.at(ts.utc(Annee, Mois, Jour, Heure, Minute, Seconde)).observe(center_object)

from skyfield.projections import build_stereographic_projection
projection = build_stereographic_projection(center)

# Star magnitude filter
limiting_magnitude = 5
bright_stars = (stars['magnitude'] <= limiting_magnitude)
magnitude = stars['magnitude'][bright_stars]
marker_size = (0.5 + limiting_magnitude - magnitude) ** 2.0

# Load satellite data
import pandas as pd
df = pd.read_csv(f"{chemin}/Satellite-avec-categorie.csv", sep=";", encoding="latin1")
df = df.head(50)
df = df[pd.notna(df["TLE_text"])]

# Initialize satellite trajectories
satellite_trajectories = {}

# Loop over time to compute satellite positions
for Minute in range(0,10):
    for Seconde in range(0, 60, 5):
        t = ts.utc(Annee, Mois, Jour, Heure, Minute, Seconde)
        print(t.ut1_strftime())
        ladate = f"Date : {Jour}/{Mois}/{Annee} à {Heure}h{Minute}min{Seconde}s UTC (faire -1h en hiver en France)\n"
        star_positions = vecteurlieuobservation.at(t).observe(liste_etoiles)

        # Compute visible satellites
        listeSATELLITESvisibles = []
        listeNomSat = []

        for numero, ligne in df.iterrows():
            noradID = ligne["NORAD_number"]
            Name = ligne["Name"]
            TLE_text = ligne["TLE_text"]

            lignesTLE = TLE_text.split("/")

            satellite = EarthSatellite(lignesTLE[1], lignesTLE[2], Name, ts)
            difference = satellite - lieu
            topocentric = difference.at(t)
            alt, az, distance = topocentric.altaz()

            if alt.degrees > 0:
                satellite_astrometrique = vecteurlieuobservation.at(t).observe(earth + satellite)
                satellite_ra, satellite_dec, distance = topocentric.radec()

                position_a_regarder = position_of_radec(satellite_ra.hours, satellite_dec.degrees, distance_au=distance.au)
                nomconstellation = d[constellation_at(position_a_regarder)]

                listeSATELLITESvisibles.append([Name,
                                                int(ligne['NORAD_number']),
                                                ligne['OWNER'],
                                                ligne['LAUNCH_DATE'],
                                                ligne['LAUNCH_SITE'],
                                                ligne['Period(minutes)'],
                                                ligne['Apogee(km)'],
                                                ligne['Perigee(km)'],
                                                ligne['categorie'],
                                                satellite_astrometrique,
                                                satellite_ra._degrees,
                                                satellite_dec.degrees,
                                                alt.degrees,
                                                az.degrees,
                                                distance.km,
                                                nomconstellation])

                listeNomSat.append(Name)

                # Store trajectory position for this satellite
                x, y = projection(satellite_astrometrique)
                if Name not in satellite_trajectories:
                    satellite_trajectories[Name] = {'x': [], 'y': []}
                satellite_trajectories[Name]['x'].append(x)
                satellite_trajectories[Name]['y'].append(y)

            print(listeNomSat)
            nbsatvisibles = f"Nombre de satellites visibles : {len(listeSATELLITESvisibles)}"
            print(nbsatvisibles, "\n")

    # Save satellite data to CSV
    dsatvisibles = pd.DataFrame(listeSATELLITESvisibles)
    dsatvisibles.rename(columns={0: "Name",
                                    1: "NORAD_number",
                                    2: "OWNER",
                                    3: "LAUNCH_DATE",
                                    4: "LAUNCH_SITE",
                                    5: "Period(minutes)",
                                    6: "Apogee(km)",
                                    7: "Perigee(km)",
                                    8: "categorie",
                                    9: "astrometrique",
                                    10: "RA_en_J2000(°)",
                                    11: "DEC_en_J2000(°)",
                                    12: "ALT(°)",
                                    13: "AZ(°)",
                                    14: "Distance(km)",
                                    15: "Nom_constellation"}, inplace=True)
    del dsatvisibles["astrometrique"]

    nomfichier = f"carte_du_ciel_lat={latOBS}&long={longOBS}__{Annee}-{Mois:02d}-{Jour:02d}_{Heure:02d}h{Minute:02d}m{Seconde:02d}s"
    dsatvisibles.to_csv(f"./fichier/{nomfichier}.csv", sep=";", index=False, encoding="latin1")

# Compute planet positions (at final timestamp)
# liste_planetes = [
#     ('Lune', eph['moon']),
#     ('Mercure', eph['mercury']),
#     ('Vénus', eph['venus']),
#     ('Mars', eph['mars']),
#     ('Jupiter', eph['jupiter barycenter']),
#     ('Saturne', eph['saturn barycenter']),
#     ('Soleil', eph['sun']),
#     ('Uranus', eph['uranus barycenter']),
#     ('Neptune', eph['neptune barycenter']),
#     ('Pluto', eph['pluto barycenter'])
# ]
#
# planetes_x = []
# planetes_y = []
# planetes_nom = []
# for nomplanete, planete in liste_planetes:
#     planete_astrometrique = vecteurlieuobservation.at(t).observe(planete)
#     app = planete_astrometrique.apparent()
#     planete_alt, planete_az, planete_distance = app.altaz()
#     if planete_alt.degrees >= 0:
#         x, y = projection(planete_astrometrique)
#         planetes_x.append(x)
#         planetes_y.append(y)
#         planetes_nom.append(nomplanete)

# Plot the sky chart
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(25, 12.5))

# Sky background
border = plt.Circle((0, 0), 1, color='navy', fill=True)
ax.add_patch(border)

# Plot stars
stars['x'], stars['y'] = projection(star_positions)
ax.scatter(stars['x'][bright_stars], stars['y'][bright_stars], s=marker_size, marker='.', color='w')

# Plot satellite trajectories
for name, traj in satellite_trajectories.items():
    if traj['x'] and traj['y']:
        ax.plot(traj['x'], traj['y'], color='red', marker='.', linestyle='-', linewidth=1, label=name)
        ax.text(traj['x'][-1], traj['y'][-1], name, fontsize=8, color='yellow', fontweight="bold")

# Plot constellations
import numpy as np
from matplotlib.collections import LineCollection
edges = [edge for name, edges in constellations for edge in edges]
edges_star1 = [star1 for star1, star2 in edges]
edges_star2 = [star2 for star1, star2 in edges]

xy1 = stars[['x', 'y']].loc[edges_star1].values
xy2 = stars[['x', 'y']].loc[edges_star2].values
lines_xy = np.rollaxis(np.array([xy1, xy2]), 1)
ax.add_collection(LineCollection(lines_xy, colors="#4e8e89"))

for i in range(len(constellations)):
    coord = stars[['x', 'y']].loc[constellations[i][1][0][0]].values
    if coord[0]**2 + coord[1]**2 <= 1:
        ax.text(coord[0], coord[1], d[constellations[i][0]], color="lightblue", fontsize="small")

# Plot planets
# ax.scatter(planetes_x, planetes_y, s=300, color='yellow', marker='.')
# for i in range(len(planetes_x)):
#     ax.text(planetes_x[i], planetes_y[i], planetes_nom[i], color='#EB8317', fontweight="bold")

# Clip to sky disk
horizon = plt.Circle((0, 0), radius=1, transform=ax.transData)
for col in ax.collections:
    col.set_clip_path(horizon)
for col in ax.texts:
    col.set_clip_path(horizon)

# Finalize plot
ax.set_aspect('equal', adjustable='box')
ax.set_title(f"{lemplacement}\n\n{ladate}\n{nbsatvisibles}")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.grid(False)
plt.axis('off')

# Save figure
fig.savefig(f'./fichier/{nomfichier}_all_trajectories.png', bbox_inches='tight')