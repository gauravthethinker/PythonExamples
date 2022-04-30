import plotly
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

excel_file = '/Users/gauravpandey/Downloads/sample.xls'
df = pd.read_excel(excel_file)
#print(df)

plt.scatter(df["Voltage"], df["Current"])
plt.show()

#data = [go.Scatter(x=df['Voltage'], y=df['Current'])]
#fig = go.Figure(data)
#fig.show()