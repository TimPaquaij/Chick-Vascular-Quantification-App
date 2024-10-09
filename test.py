

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from pandas import DataFrame
fig_input_2, ax2 = plt.subplots(figsize = (2,2))
number_list = [0,2,3,4,5,6]
part_list = [0.8,0.3,0.4,0.5,0.6,0.7]
data_diameter = {'Diameter (\u03BCm)': np.array(number_list[1:]), 'Part of total vessel area' : np.array(part_list)[1:]} #Making dictionary
df1 = DataFrame(data_diameter,columns=['Diameter (\u03BCm)','Part of total vessel area']) #Making dataframe of dictionary

color = sns.color_palette("viridis", 1+len(np.array(df1['Diameter (\u03BCm)'])))

print(len(color))
rank =  np.array(df1['Diameter (\u03BCm)']).tolist()
rank.insert(0,0)
ax = sns.barplot(x='Diameter (\u03BCm)', y='Part of total vessel area', data=df1,ax=ax2,palette=np.array(color))
print(rank)
plt.show()


