# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

# %%
file_path = 'production data.xlsx'
df_production = pd.read_excel(file_path)

# %%
df_production.info()

# %%
df_production.head(5)

# %%
A= sns.heatmap(df_production.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.xticks(rotation = 45, ha = 'right')

# %%
df_production['WELL_TYPE'].unique()

# %%
df_production['NPD_WELL_BORE_CODE'].unique()

# %%
# filter production well which has well type as OP
df_production[df_production['WELL_TYPE'] == 'OP']['NPD_WELL_BORE_CODE'].unique()

# %%
# filtered dataframe only has data for producer type well
df_filtered =df_production[df_production['WELL_TYPE'] == 'OP']

# %%
# some colume in filtered dataframe not useful for this study, drop these column
df_filtered.drop(labels=['WELL_BORE_CODE',
                         'NPD_WELL_BORE_NAME',
                         'NPD_FIELD_CODE',
                         'NPD_FIELD_NAME',
                         'NPD_FACILITY_CODE',
                         'NPD_FACILITY_NAME',
                        'AVG_CHOKE_UOM',
                        'FLOW_KIND',
                        'WELL_TYPE'], axis=1)

# %%
# use heatmap to visualize null data in our dataframe after drop unuseful columns
sns.heatmap(df_filtered.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.xticks(rotation = 45, ha = 'right')

# %%
# counts number of data available for each well id
df_filtered['NPD_WELL_BORE_CODE'].value_counts()

# %%
#from the seaborn library , we can use the Empirical Cumulative Distribution Function , and plot the oil production from all those wells having code as producers 

sns.ecdfplot(x='BORE_OIL_VOL',data=df_filtered, hue='NPD_WELL_BORE_CODE',palette=['r', 'g', 'b', 'y', 'brown', 'magenta'])

plt.show()

# %% [markdown]
# As seen in the above ECDF plot, we can see that for the well 7405, almost 40% data is zero Bore_OIL_VOL production and similarly for Well 7289, almost 20% data is zero Bore_OIL_VOL production. Similar case is with the well 5769. In All these three wells , the total Cumulative production is also significatly less compared to other producing wells. So let us ignore these three wells during our production prediction Machine learning model training

# %%
## Another Scatter plot to show the oil production from these producer wells. 
#Clearly the wells #7405, # 5769, #7289 production volume is insignificant
df_filtered["NPD_WELL_BORE_CODE"] = df_filtered["NPD_WELL_BORE_CODE"].astype(str)
fig = px.scatter(df_filtered, x="DATEPRD", y="BORE_OIL_VOL",
                 color="NPD_WELL_BORE_CODE",
                 hover_name="NPD_WELL_BORE_CODE")
fig.show()

# %%
fig = px.line(df_filtered, x="DATEPRD", y="BORE_OIL_VOL", color='NPD_WELL_BORE_CODE')
fig.show()

# %%
fig = px.line(df_filtered, x="DATEPRD", y="BORE_WAT_VOL", color='NPD_WELL_BORE_CODE')
fig.show()

# %%
fig = px.line(df_filtered, x="DATEPRD", y="BORE_GAS_VOL", color='NPD_WELL_BORE_CODE')
fig.show()

# %%
df_filtered.info()