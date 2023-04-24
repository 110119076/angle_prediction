import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle

st.write("""# Angle Visualization""")

load_reg=pickle.load(open('SV_reg.pkl', 'rb'))
df = pd.read_csv('Data_Pred.csv')
st.write(df.iloc[:,1:].head())

st.write("""# Model Perfromance""")
st.write(px.line(df[(df.index>0) & (df.index<100)],x='index',y=['Angle','SVR_Pred_Angle','LR_Pred_Angle','RF_Pred_Angle','SGD_Pred_Angle']))

st.sidebar.header('User Input Features')
def user_input_features():
    current_range = st.sidebar.slider('Current (A)', -2.5,2.5,0.0,0.1)
    voltage_range = st.sidebar.slider('Voltage (V)', -10.0,10.0,0.0,0.1)

    data = {'Current ': current_range, 'voltage': voltage_range}
    features = pd.DataFrame(data,index=[0])
    return features

in_df = user_input_features()
prediction = load_reg.predict(in_df)

st.subheader('Angle Prediction')
st.write(prediction[0])

# if selected_day == 'Tue':
#     date=df_ice.loc[df_ice.Day_Name == selected_day,'Date(DD/MM/YYYY)']
#     data=df_ice.loc[df_ice.Day_Name == selected_day].copy()

#     fig = make_subplots(rows=16, cols=1, subplot_titles=tuple(date.unique()))
#     r,c=1,1
#     for v_idx,v in enumerate(date.unique()[:]):
#         temp=data[date==v]
#         fig.append_trace(go.Scatter(x=temp['Time'],y=temp['KWH'],name='KWH_ICE'),row=r, col=c)
#         r+=1
#     fig.update_layout(height=3000, width=1000)
#     fig.show()

