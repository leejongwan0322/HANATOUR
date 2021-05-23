import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#https://towardsdatascience.com/3-awesome-visualization-techniques-for-every-dataset-9737eecacbe8

# We dont Probably need the Gridlines. Do we? If yes comment this line
sns.set(style="ticks")
player_df = pd.read_csv("./data.csv")
numcols = [
    'Overall',
    'Potential',
    'Crossing',
    'Finishing',
    'ShortPassing',
    'Dribbling',
    'LongPassing',
    'BallControl',
    'Acceleration',
    'SprintSpeed',
    'Agility',
    'Stamina',
    'Value',
    'Wage']

catcols = ['Name','Club','Nationality','Preferred Foot','Position','Body Type']
print(numcols+catcols)
player_df = player_df[numcols+catcols]
print(player_df.head(5))

def wage_split(x):
    try:
        return int(x.split("K")[0][1:])
    except:
        return 0

player_df['Wage'] = player_df['Wage'].apply(lambda x : wage_split(x))

def value_split(x):
    try:
        if 'M' in x:
            return float(x.split("M")[0][1:])
        elif 'K' in x:
            return float(x.split("K")[0][1:])/1000
    except:
        return 0

player_df['Value'] = player_df['Value'].apply(lambda x : value_split(x))

#Heatmap
corr = player_df.corr()
g = sns.heatmap(corr, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
sns.despine()
g.figure.set_size_inches(14, 10)
player_df = player_df.fillna(0)
plt.show()

# import pandas as pd
#
# dataframe=pd.DataFrame({'Attendance': {0: 60, 1: 100, 2: 80,3: 78,4: 95},
#                         'Name': {0: 'Olivia', 1: 'John', 2: 'Laura',3: 'Ben',4: 'Kevin'},
#                         'Obtained Marks': {0: 90, 1: 75, 2: 82, 3: 64, 4: 45}})
# print("The Original Data frame is: \n")
# print(dataframe)
#
# dataframe1 = dataframe.corr()
# print("The Correlation Matrix is: \n")
# print(dataframe1)

#Pairplots
filtered_player_df = player_df[(player_df['Club'].isin(['FC Barcelona', 'Paris Saint-Germain','Manchester United', 'Manchester City', 'Chelsea', 'Real Madrid','FC Porto','FC Bayern München'])) &
                      (player_df['Nationality'].isin(['England', 'Brazil', 'Argentina',
       'Brazil', 'Italy','Spain','Germany']))
                     ]

# Single line to create pairplot
g = sns.pairplot(filtered_player_df[['Value','SprintSpeed','Potential','Wage']])
g = sns.pairplot(filtered_player_df[['Value','SprintSpeed','Potential','Wage','Club']],hue = 'Club')
plt.show()


#SwarmPlot
g = sns.swarmplot(
    y = "Club",
    x = 'Wage',
    data = filtered_player_df,
    # Decrease the size of the points to avoid crowding
    size = 7)

# remove the top and right line in graph
sns.despine()
g.figure.set_size_inches(14,10)
plt.show()


g = sns.boxplot(
    y = "Club",
    x = 'Wage',
    data = filtered_player_df, whis=np.inf)

g = sns.swarmplot(
    y = "Club",
    x = 'Wage',
    data = filtered_player_df,
    # Decrease the size of the points to avoid crowding
    size = 7,
    color = 'black')

# remove the top and right line in graph
sns.despine()
g.figure.set_size_inches(12,8)
plt.show()


max_wage = filtered_player_df.Wage.max()
max_wage_player = filtered_player_df[(player_df['Wage'] == max_wage)]['Name'].values[0]
g = sns.boxplot(y = "Club",
              x = 'Wage',
              data = filtered_player_df, whis=np.inf)
g = sns.swarmplot(y = "Club",
              x = 'Wage',
              data = filtered_player_df,
              # Decrease the size of the points to avoid crowding
              size = 7,color='black')
# remove the top and right line in graph
sns.despine()
# Annotate. xy for coordinate. max_wage is x and 0 is y. In this plot y ranges from 0 to 7 for each level
# xytext for coordinates of where I want to put my text
plt.annotate(s = max_wage_player,
             xy = (max_wage,0),
             xytext = (500,1),
             # Shrink the arrow to avoid occlusion
             arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03},
             backgroundcolor = 'white')
g.figure.set_size_inches(12,8)
plt.show()
