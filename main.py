import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# Find all CSV files
data_files = glob.glob('/home/vankhoa@median.cad/code/github/sae/data/45nord/*/fichiers/*.csv')
print(len(data_files))
print(data_files[0])

# Aggregate all data
all_dfs = []
for file in tqdm(data_files):
    # print(file)
    df = pd.read_csv(file, sep=';', encoding='latin-1')
    # print(df.columns)
    df = df[['Name', 'Nom_constellation', 'categorie', 'RA_en_J2000(°)', 'DEC_en_J2000(°)']].drop_duplicates()


    df["hour"] = int(file.split('/')[-1].split('_')[-1].split('h')[0])
    df["minute"] = int(file.split('/')[-1].split('_')[-1].split('h')[1].split('m')[0])
    df["second"] = int(file.split('/')[-1].split('_')[-1].split('h')[1].split('m')[1].split('s')[0])
    df['time(s)'] = df['hour']*3600 + df['minute']*60 + df['second']
    
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = full_df.dropna(subset=['categorie'])

data = defaultdict(list)
nrow = 0

for i,row in tqdm(full_df.iterrows()):
    # print(row)
    num_categorie = len(row['categorie'].split(','))
    for categorie in row['categorie'].split(','):
        data[nrow] = {
            'categorie': categorie,
            'Nom_constellation': row['Nom_constellation'],
            'Name': row['Name'],
            'time(s)': row['time(s)'],
            'RA_en_J2000(°)': row['RA_en_J2000(°)'],
            'DEC_en_J2000(°)': row['DEC_en_J2000(°)']
        }
        nrow += 1

data_df = pd.DataFrame.from_dict(data, orient='index')
print(data_df.head())

# exit()

# Group and count unique Name
heatmap_data = data_df.groupby(['categorie', 'Nom_constellation'])['Name'].count().reset_index()

# Pivot for heatmap
pivot_table = heatmap_data.pivot(index='Nom_constellation', columns='categorie', values='Name').fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 8))
plt.title('Total Number of Satellites by Categorie and Constellation')
plt.xlabel('Categorie')
plt.ylabel('Constellation')
plt.imshow(pivot_table, aspect='auto', cmap="Blues")
plt.colorbar(label='Number of Satellites')
plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)
plt.tight_layout()
plt.savefig('output/heatmap_number_of_satellites_by_constellation_and_categorie.png')
plt.show()

# Find the top 10 categories with the most unique satellites
category_counts = data_df.groupby('categorie')['Name'].count().sort_values(ascending=False)
top_categories = category_counts.index[:10]

# Plot: number of satellites over time for the top 10 categories
plt.figure(figsize=(12, 8))
for categorie, group in data_df.groupby('categorie'):
    if categorie not in top_categories:
        continue
    satellites_over_time = group.groupby('time(s)')['Name'].count().sort_index()
    plt.plot(satellites_over_time.index, satellites_over_time.values, label=categorie)
plt.xlabel('Time (s)')
plt.ylabel('Number of Satellites')
plt.title('Number of Satellites Over Time for Top 10 Categories')
plt.legend(title='Categorie', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('output/satellites_over_time_by_top10_categories.png')
plt.show()

# Boxplot: x = categorie, y = total number of satellites (count, not nunique)
boxplot_data = data_df.groupby(['categorie', 'time(s)'])['Name'].count().reset_index()
print(boxplot_data.head())
plt.figure(figsize=(14, 8))
sns.boxplot(data=boxplot_data, x='categorie', y='Name')
plt.xlabel('Categorie')
plt.ylabel('Number of Satellites (per time point, total count)')
plt.title('Distribution of Number of Satellites per Categorie (Total Count)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/boxplot_satellites_by_categorie_total_count.png')
plt.show()

# Boxplot: x = Nom_constellation, y = total number of satellites (count, not nunique)
boxplot_constellation = data_df.groupby(['Nom_constellation', 'time(s)'])['Name'].count().reset_index()
print(boxplot_constellation.head())
plt.figure(figsize=(16, 8))
sns.boxplot(data=boxplot_constellation, x='Nom_constellation', y='Name')
plt.xlabel('Nom_constellation')
plt.ylabel('Number of Satellites (per time point, total count)')
plt.title('Distribution of Number of Satellites per Constellation (Total Count)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/boxplot_satellites_by_constellation_total_count.png')
plt.show()

# Boxplot: x = categorie, y = number of unique satellites (per time point)
boxplot_unique_categorie = data_df.groupby(['categorie', 'time(s)'])['Name'].nunique().reset_index()
print(boxplot_unique_categorie.head())
plt.figure(figsize=(14, 8))
sns.boxplot(data=boxplot_unique_categorie, x='categorie', y='Name')
plt.xlabel('Categorie')
plt.ylabel('Number of Unique Satellites (per time point)')
plt.title('Distribution of Number of Unique Satellites per Categorie')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/boxplot_unique_satellites_by_categorie.png')
plt.show()

# Boxplot: x = Nom_constellation, y = number of unique satellites (per time point)
boxplot_unique_constellation = data_df.groupby(['Nom_constellation', 'time(s)'])['Name'].nunique().reset_index()
print(boxplot_unique_constellation.head())
plt.figure(figsize=(16, 8))
sns.boxplot(data=boxplot_unique_constellation, x='Nom_constellation', y='Name')
plt.xlabel('Nom_constellation')
plt.ylabel('Number of Unique Satellites (per time point)')
plt.title('Distribution of Number of Unique Satellites per Constellation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/boxplot_unique_satellites_by_constellation.png')
plt.show()

# Select top 10 Name with most count
name_counts = data_df['Name'].value_counts().head(10)
top10_names = name_counts.index.tolist()

# Plot time vs RA for top 10 Names
plt.figure(figsize=(14, 8))
for name in top10_names:
    subset = data_df[data_df['Name'] == name]
    plt.plot(subset['time(s)'], subset['RA_en_J2000(°)'], marker='o', linestyle='-', label=name)
plt.xlabel('Time (s)')
plt.ylabel('RA_en_J2000(°)')
plt.title('Time vs RA for Top 10 Names')
plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('output/time_vs_RA_top10_names.png')
plt.show()

# Plot time vs DEC for top 10 Names
plt.figure(figsize=(14, 8))
for name in top10_names:
    subset = data_df[data_df['Name'] == name]
    plt.plot(subset['time(s)'], subset['DEC_en_J2000(°)'], marker='o', linestyle='-', label=name)
plt.xlabel('Time (s)')
plt.ylabel('DEC_en_J2000(°)')
plt.title('Time vs DEC for Top 10 Names')
plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('output/time_vs_DEC_top10_names.png')
plt.show()

# Plot DEC vs RA for top 10 Names
plt.figure(figsize=(10, 8))
for name in top10_names:
    subset = data_df[data_df['Name'] == name]
    plt.plot(subset['RA_en_J2000(°)'], subset['DEC_en_J2000(°)'], marker='o', linestyle='-', label=name)
plt.xlabel('RA_en_J2000(°)')
plt.ylabel('DEC_en_J2000(°)')
plt.title('DEC vs RA for Top 10 Names')
plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('output/DEC_vs_RA_top10_names.png')
plt.show()

# Boxplots for top 3 constellations: x=categorie, y=number of satellites (count) per time point
constellation_counts = data_df['Nom_constellation'].value_counts().head(3)
top3_constellations = constellation_counts.index.tolist()

for constellation in top3_constellations:
    subset = data_df[data_df['Nom_constellation'] == constellation]
    boxplot_data = subset.groupby(['categorie', 'time(s)'])['Name'].count().reset_index()
    print(f"Boxplot data for constellation {constellation}:")
    print(boxplot_data.head())
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=boxplot_data, x='categorie', y='Name')
    plt.xlabel('Categorie')
    plt.ylabel('Number of Satellites (per time point, total count)')
    plt.title(f'Distribution of Number of Satellites per Categorie\nConstellation: {constellation}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    safe_constellation = str(constellation).replace('/', '_').replace(' ', '_')
    plt.savefig(f'output/boxplot_satellites_by_categorie_for_constellation_{safe_constellation}.png')
    plt.show()

# For each of the top 3 constellations, plot time vs RA, time vs DEC, and DEC vs RA for top 10 Names in that constellation
for constellation in top3_constellations:
    subset = data_df[data_df['Nom_constellation'] == constellation]
    name_counts = subset['Name'].value_counts().head(10)
    top10_names = name_counts.index.tolist()

    # Time vs RA
    plt.figure(figsize=(14, 8))
    for name in top10_names:
        name_subset = subset[subset['Name'] == name]
        plt.plot(name_subset['time(s)'], name_subset['RA_en_J2000(°)'], marker='o', linestyle='-', label=name)
    plt.xlabel('Time (s)')
    plt.ylabel('RA_en_J2000(°)')
    plt.title(f'Time vs RA for Top 10 Names\nConstellation: {constellation}')
    plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    safe_constellation = str(constellation).replace('/', '_').replace(' ', '_')
    plt.savefig(f'output/time_vs_RA_top10_names_{safe_constellation}.png')
    plt.show()

    # Time vs DEC
    plt.figure(figsize=(14, 8))
    for name in top10_names:
        name_subset = subset[subset['Name'] == name]
        plt.plot(name_subset['time(s)'], name_subset['DEC_en_J2000(°)'], marker='o', linestyle='-', label=name)
    plt.xlabel('Time (s)')
    plt.ylabel('DEC_en_J2000(°)')
    plt.title(f'Time vs DEC for Top 10 Names\nConstellation: {constellation}')
    plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'output/time_vs_DEC_top10_names_{safe_constellation}.png')
    plt.show()

    # DEC vs RA
    plt.figure(figsize=(10, 8))
    for name in top10_names:
        name_subset = subset[subset['Name'] == name]
        plt.plot(name_subset['RA_en_J2000(°)'], name_subset['DEC_en_J2000(°)'], marker='o', linestyle='-', label=name)
    plt.xlabel('RA_en_J2000(°)')
    plt.ylabel('DEC_en_J2000(°)')
    plt.title(f'DEC vs RA for Top 10 Names\nConstellation: {constellation}')
    plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'output/DEC_vs_RA_top10_names_{safe_constellation}.png')
    plt.show()

# For each of the top 3 constellations, plot number of satellites over time for top 10 categories
for constellation in top3_constellations:
    subset = data_df[data_df['Nom_constellation'] == constellation]
    category_counts = subset.groupby('categorie')['Name'].count().sort_values(ascending=False)
    top_categories = category_counts.index[:10]

    plt.figure(figsize=(12, 8))
    for categorie, group in subset.groupby('categorie'):
        if categorie not in top_categories:
            continue
        satellites_over_time = group.groupby('time(s)')['Name'].count().sort_index()
        plt.plot(satellites_over_time.index, satellites_over_time.values, label=categorie)
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Satellites')
    plt.title(f'Number of Satellites Over Time for Top 10 Categories\nConstellation: {constellation}')
    plt.legend(title='Categorie', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    safe_constellation = str(constellation).replace('/', '_').replace(' ', '_')
    plt.savefig(f'output/satellites_over_time_by_top10_categories_{safe_constellation}.png')
    plt.show()











