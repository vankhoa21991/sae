## Calculer et afficher une trajectoire d'orbite d'un satellite - exemple 9

import os
chemin = "/home/vankhoa@median.cad/code/github/sae/data"

# os.chdir(chemin)

from skyfield.api import load, wgs84

# de421 donne la position des planètes dans l'espace
eph = load(chemin+'/de421.bsp')

# lune = eph['moon'], mars = eph['mars'],.., soleil = eph['sun']
earth = eph['earth']

# Définit un objet date d'observation en date UTC (temps universel)
ts = load.timescale()

# Si on veut l'heure actuelle
#t = ts.from_datetime(utc_dt)

# Réglage de l'heure d'observation
# C'est 1h de moins en hiver par rapport à l'heure locale en FR
Jour=14
Mois=5
Annee=2025
Heure=0
Minute=00
Seconde=00


t = ts.utc(Annee, Mois, Jour, Heure, Minute, Seconde)


# Lieu d'observation : latitude, longitude, altitude
latOBS = 45.0
longOBS = 0
altitudeOBS = 100

from skyfield.api import N,S,E,W
lieu = wgs84.latlon(latOBS, longOBS, elevation_m=altitudeOBS)
vecteurlieuobservation = earth + lieu

lemplacement = "Emplacement : Latitude="+str(latOBS)+" / Longitude="+str(longOBS)

########## Utilisation de la bibliothèque Hipparcos
## https://rhodesmill.org/skyfield/stars.html

from skyfield.data import hipparcos
# Le fichier de données Hipparcos contient la position des étoiles
url = hipparcos.URL
# https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat

f = load.open(hipparcos.URL)
stars = hipparcos.load_dataframe(f)
f.close()

from skyfield.api import Star
liste_etoiles = Star.from_dataframe(stars)
star_positions = vecteurlieuobservation.at(t).observe(liste_etoiles)


########## Constellations

from skyfield.api import position_of_radec, load_constellation_map, load_constellation_names
d = dict(load_constellation_names())

# Fonction de détermination de la constellation
constellation_at = load_constellation_map()

# Chargement depuis le disque
import json
f = open(chemin+"/constellationship.json","rt")
constellations = json.load(f)
f.close()


###########################
print()
# Créer un objet itérable contenant toutes les étoiles
# de type catalogue skyfield.starlib.Star
from skyfield.api import Star
liste_etoiles = Star.from_dataframe(stars)

########### Définition d'une projection sphérique

# Définit la position du lieu d'observation
# en utilisant le système de données géocentriques
observateur = lieu.at(t)

# On récupère les coordonnées en RA/DEC du lieu d'observation
# dans le système géocentrique : comme si on était placé au centre
# de la Terre et qu'elle était transparente, et qu'on observait vers le ciel
# dans la direction où se trouve le lieu d'observation au sol
# (donc à distance du rayon terrestre, soit 6371 km en moyenne, variant de 6 352 km à  6 384 km)
ra, dec, distance = observateur.radec()
print(ra, dec, distance)

# On crée un objet fictif dans le ciel qui est aux coordonnées RA et DEC de l'observateur
# mais qui est situé au-dessus du niveau du sol. Si onn'indique pas de distance, par défaut la
# distance de l'objet créé est de 1 gigaparsec = 1 000 000 de parsecs
# 1 parsec = 648 000 / pi unités astronomiques = environ 3,26 années-lumières
# 1 unit astronomique = 149 597 870 700 mètres
center_object = Star(ra=ra, dec=dec)

# Centre de la projection :
center = earth.at(t).observe(center_object)

from skyfield.projections import build_stereographic_projection
projection = build_stereographic_projection(center)

# Limite de magnitude maximale visible
limiting_magnitude = 5

# Création d'une liste de booléen pour filtre correspondant à la limite
bright_stars = (stars['magnitude'] <= limiting_magnitude)
# Application du filtre
magnitude = stars['magnitude'][bright_stars]
# Création d'une liste de tailles associées aux magnitudes
marker_size = (0.5 + limiting_magnitude - magnitude) ** 2.0

from skyfield.api import EarthSatellite

#### créer ici une boucle si on veut étudier plusieurs satellites

import pandas as pd
df = pd.read_csv(f"{chemin}/Satellite-avec-categorie.csv",sep=";",encoding="latin1")
# df = df.head(20)
df = df[pd.notna(df["TLE_text"])]

# print(df["NORAD_number"].value_counts())

########### Partie ajout des étoiles

stars['x'], stars['y'] = projection(star_positions)


######################################################################
######################################################################

listeposXISS = []
listeposYISS = []

satellite_trajectories = {}

# Loop over 24 hours, every 3 hours
for hour_block in range(0, 24, 3):
    for minute in range(0, 60):
        for second in [0, 30]:  # Two times in each minute
            t = ts.utc(Annee, Mois, Jour, hour_block, minute, second)
            print(t.ut1_strftime())
            ladate = f"Date : {Jour}/{Mois}/{Annee} à {hour_block}h{minute}min{second}s UTC (faire -1h en hiver en France)\n"
            star_positions = vecteurlieuobservation.at(t).observe(liste_etoiles)
            listeSATELLITESvisibles = []
            listeNomSat = []

            for numero, ligne in df.iterrows():
                noradID = ligne["NORAD_number"]

                if noradID not in [25371, 27508, 27441]:
                    continue

                Name = TLE_text = ligne["Name"]
                TLE_text = ligne["TLE_text"]
                lignesTLE = TLE_text.split("/")
                satellite = EarthSatellite(lignesTLE[1], lignesTLE[2], Name, ts)
                difference = satellite - lieu
                topocentric = difference.at(t)
                alt, az, distance = topocentric.altaz()

                if alt.degrees > 0:
                    try:
                        satellite_astrometrique = vecteurlieuobservation.at(t).observe(earth+satellite)
                    except ValueError as e:
                        print(f"Skipping {Name} at {t.utc_iso()} due to: {e}")
                        continue
                    satellite_ra, satellite_dec, distance = topocentric.radec()
                    position_a_regarder = position_of_radec(satellite_ra.hours, satellite_dec.degrees, distance_au=distance.au)
                    nomconstellation = d[constellation_at(position_a_regarder)]
                    listeSATELLITESvisibles.append([Name,
                                                    ligne["International_designation"],
                                                    int(ligne['NORAD_number']),
                                                    ligne['OPS_STATUS_CODE'],
                                                    ligne['OWNER'],
                                                    ligne['LAUNCH_DATE'],
                                                    ligne['LAUNCH_SITE'],
                                                    ligne['Period(minutes)'],
                                                    ligne['Apogee(km)'],
                                                    ligne['Perigee(km)'],
                                                    ligne['RCS'],
                                                    ligne['Classification'],
                                                    ligne['Epoch'],
                                                    ligne['1st_derivation'],
                                                    ligne['2nd_derivative'],
                                                    ligne['Drag'],
                                                    ligne['Set_number'],
                                                    ligne['Inclination(°)'],
                                                    ligne['Right_ascension(°)'],
                                                    ligne['Eccentricity'],
                                                    ligne['Argument_periastre(°)'],
                                                    ligne['Mean_anomaly(°)'],
                                                    ligne['Revolution_number'],
                                                    ligne['categorie'],
                                                    satellite_astrometrique,
                                                    satellite_ra._degrees,
                                                    satellite_dec.degrees,
                                                    alt.degrees,
                                                    az.degrees,
                                                    distance.km,
                                                    nomconstellation])
                    listeNomSat.append(Name)
                    x, y = projection(listeSATELLITESvisibles[0][24])
                    if Name not in satellite_trajectories:
                        satellite_trajectories[Name] = {'x': [], 'y': []}
                    satellite_trajectories[Name]['x'].append(x)
                    satellite_trajectories[Name]['y'].append(y)

            # print(listeNomSat)
            nbsatvisibles = "Nombre de satellites visibles : "+str(len(listeSATELLITESvisibles))
            print(nbsatvisibles,"\n")
            dsatvisibles = pd.DataFrame(listeSATELLITESvisibles)
            dsatvisibles.rename(columns={0:"Name",
                                        1:"International_designation",
                                        2:"NORAD_number", # 25371 27508 27441
                                        3:"OPS_STATUS_CODE",
                                        4:"OWNER",
                                        5:"LAUNCH_DATE",
                                        6:"LAUNCH_SITE",
                                        7:"Period(minutes)",
                                        8:"Apogee(km)",
                                        9:"Perigee(km)",
                                        10:"RCS",
                                        11:"Classification",
                                        12:"Epoch",
                                        13:"1st_derivation",
                                        14:"2nd_derivative",
                                        15:"Drag",
                                        16:"Set_number",
                                        17:"Inclination(°)",
                                        18:"Right_ascension(°)",
                                        19:"Eccentricity",
                                        20:"Argument_periastre(°)",
                                        21:"Mean_anomaly(°)",
                                        22:"Revolution_number",
                                        23:"categorie",
                                        24:"astrometrique",
                                        25:"RA_en_J2000(°)",
                                        26:"DEC_en_J2000(°)",
                                        27:"ALT(°)",
                                        28:"AZ(°)",
                                        29:"Distance(km)",
                                        30:"Nom_constellation"}, inplace=True)
            if "astrometrique" in dsatvisibles.columns:
                del dsatvisibles["astrometrique"]
            nomfichier = f"carte_du_ciel_lat={latOBS}&long={longOBS}__{Annee}-{Mois}-{Jour}_{hour_block}h{minute}m{second}s"
            os.makedirs(f"./output/fichier", exist_ok=True)
            dsatvisibles.to_csv(f"./output/fichier/{nomfichier}.csv", sep=";", index=False, encoding="latin1")



########### Partie ajout des planètes visibles
liste_planetes = [
                    ('Lune',eph['moon']),
                    ('Mercure',eph['mercury']),
                    ('Vénus',eph['venus']),
                    ('Mars',eph['mars']),
                    ('Jupiter',eph['jupiter barycenter']),
                    ('Saturne',eph['saturn barycenter']),
                    ('Soleil', eph['sun']),
                    ('Uranus', eph['uranus barycenter']),
                    ('Neptune', eph['neptune barycenter']),
                    ('Pluto', eph['pluto barycenter'])
                ]

planetes_x = []
planetes_y = []
planetes_nom = []
for nomplanete,planete in liste_planetes:
    planete_astrometrique = vecteurlieuobservation.at(t).observe(planete)

    app = planete_astrometrique.apparent()
    planete_alt, planete_az, planete_distance = app.altaz()

    # On ajouter la planète à la liste des planètes à afficher seulement si elle est visible
    if planete_alt.degrees>=0:
        x,y = projection(planete_astrometrique)
        planetes_x.append(x)
        planetes_y.append(y)
        planetes_nom.append(nomplanete)

#########################################################################
########### Partie réalisation du graphique

### Début du tracé graphique
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(25, 12.5))

# Tracé du disque du fond de ciel en bleu navy
border = plt.Circle((0, 0), 1, color='navy', fill=True)
ax.add_patch(border)


### Tracé des étoiles
# stars['x'][bright_stars],stars['y'][bright_stars] : on filtre pour garder seulement les mêmes étoiles les plus brillantes de magnitude <= limiting_magnitude
ax.scatter(stars['x'][bright_stars], stars['y'][bright_stars], s=marker_size,marker='.',color='w')

### Tracé des satellites
# ax.plot(listeposXISS, listeposYISS, color='red', marker='.', label=Name)
#
# if listeposXISS and listeposYISS:  # make sure the lists are not empty
#     ax.text(listeposXISS[-1], listeposYISS[-1], Name, fontsize=9, color='yellow', fontweight="bold")

for name, traj in satellite_trajectories.items():
    if traj['x'] and traj['y']:  # Ensure there are positions to plot
        ax.plot(traj['x'], traj['y'], color='red', marker='.', linestyle='-', linewidth=1, label=name)
        # Label only the final position
        ax.text(traj['x'][-1], traj['y'][-1], name, fontsize=8, color='yellow', fontweight="bold")

# for i in range(len(listeSATELLITESvisibles)):
#     ax.text(satellites_x[i],satellites_y[i],satellites_nom[i],color='pink')


#### Tracé des Constellations
## https://rhodesmill.org/skyfield/example-plots.html#neowise-chart

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
    if coord[0]**2+coord[1]**2<=1:
        ax.text(coord[0],coord[1],d[constellations[i][0]],color="lightblue",fontsize="small")

#######################################################

### Tracé des planètes
ax.scatter(planetes_x, planetes_y,s=300, color='yellow', marker='.')
for i in range(len(planetes_x)):
    ax.text(planetes_x[i],planetes_y[i],planetes_nom[i],color='#EB8317',fontweight="bold")


### Coupe tout ce qui dans le dessin sort du disque du fond du ciel
horizon = plt.Circle((0, 0), radius=1, transform=ax.transData)
for col in ax.collections:
    col.set_clip_path(horizon)

for col in ax.texts:
    col.set_clip_path(horizon)

### Fin du tracé graphique
ax.set_aspect('equal', adjustable='box')
ax.set_title(lemplacement+"\n\n"+ladate+"\n"+nbsatvisibles)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.grid(False)
plt.axis('off')

fig.savefig(f'./output/fichier/{noradID}.png',bbox_inches='tight')
# plt.show()