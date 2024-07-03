import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

GROUPSIZE=5
INPUTFILE=sys.argv[1]

df = pd.read_excel(INPUTFILE)
df = df.drop('Start time', axis=1)
df = df.drop('Completion time', axis=1)
df = df.drop('Email', axis=1) #Eller Name
df = df.drop('Id', axis=1)

print(df.head())
print(list(df.columns))
only_numbers_cols = list(df.columns)[1:-1]

amb_trans = {'Lavt':1, 'Middel/lavt':2, 'Middel': 3, 'Middel/højt':4, 'Højt':5}
age_trans = {'mindre end 25':1, 'mellem 25 og 30':2, 'mellem 30 og 35': 3, 'mellem 35 og 40': 4, 'mere end 40': 5}
dist_trans = {'Mindre end 5 km':1, '5-10 km': 2, '10-20 km': 3, '20-30 km': 4, 'Mere end 40 km': 5}

df['Hvad er dit ambitionsniveau for at studere IT-teknolog?\n\nLavt: Jeg vil bare lige akkurat klare mig igennem uddannelsen og lave absolut minimum for at komme igennem.\n\nHøjt: Jeg er villig til at bruge m'] = df['Hvad er dit ambitionsniveau for at studere IT-teknolog?\n\nLavt: Jeg vil bare lige akkurat klare mig igennem uddannelsen og lave absolut minimum for at komme igennem.\n\nHøjt: Jeg er villig til at bruge m'].map(amb_trans)
df['Hvad er din alder år?\n.'] = df['Hvad er din alder år?\n.'].map(age_trans)
df['Hvor langt bor du fra KEA i kilometer?\n.Afstand:'] = df['Hvor langt bor du fra KEA i kilometer?\n.Afstand:'].map(dist_trans)

CLUSTERS=df.shape[0]//GROUPSIZE
# Make name s.t. it can be identified afterwards
print(df.head())
print(list(df.columns))

random_state = 100 # Change this to get a slightly other result
common_params = {
            "n_init": "auto",
            "random_state": random_state,
        }

only_numbers = df[only_numbers_cols].to_numpy()

y_pred = KMeans(n_clusters=CLUSTERS, **common_params).fit_predict(only_numbers)

print(y_pred)

data = list(zip(df['Name'],y_pred, df['Hvem kunne du godt tænke dig at være i gruppe med?\n(Bemærk der er ikke er garanti for at man kommer i gruppe med ønskede medlemmer)\n'])) #TODO Add note field about wanted group members to zip as well

print(*data,sep='\n')

df_res = pd.DataFrame(data, columns =['Name', 'Group number', 'Wanted member'])
df_res.to_excel('groups.xlsx')

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
axs.scatter(only_numbers[:, 0], only_numbers[:, 1], c=y_pred)
axs.set_title("Grupperne")
plt.show()

