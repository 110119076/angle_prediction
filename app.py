import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.write("""# Angle Visualization""")

load_reg=pickle.load(open('SV_reg.pkl', 'rb'))
df = pd.read_csv('Data_Pred.csv')
st.write(df.iloc[:,1:].head())

st.write("""# Model Performance""")
st.write(px.line(df[(df.index>0) & (df.index<100)],x='index',y=['Angle','SVR_Pred_Angle','LR_Pred_Angle','RF_Pred_Angle','SGD_Pred_Angle']))

df = pd.read_excel('Data.xlsx')
# df.head()

X_train, X_test, y_train, y_test = train_test_split(df[['Current ','voltage']],df['Angle'],test_size=0.33)
scaler = StandardScaler()
scaler = scaler.fit(X_train)
df_scale = scaler.transform(X_train)
train_scale=df_scale

scaler1 = StandardScaler()
scaler1 = scaler1.fit(X_test)
test_scale = scaler1.transform(X_test)



st.sidebar.header('User Input Features')
def user_input_features():
    current = st.number_input('Current -2.5 to 2.5A')
    voltage = st.number_input('Voltage -10 to 10V')


    data = {'Current ': current, 'voltage': voltage}
    features = pd.DataFrame(data,index=[0])
    return features

classifier = st.sidebar.selectbox('Select Classifier', ('SVM', 'RF', 'SGD', 'LinearRegression'))
cv_count = st.sidebar.slider('Cross-validation count', 2, 5, 3)
st.sidebar.write('---')
# clf_name=['SVM','SGD','LR','RF']
def get_classifier(clf_name):
    model = None
    parameters = None
    
    if clf_name == 'SVM':
        st.sidebar.write("**Kernel Type**")
        st.sidebar.write('Specifies the kernel type to be used in the algorithm.')
        kernel_type = st.sidebar.multiselect('', options=['linear', 'rbf', 'poly'], default=['linear', 'rbf', 'poly'])
        st.sidebar.subheader('')
        
        st.sidebar.write('**Regularization Parameter**')
        st.sidebar.write('The strength of the regularization is inversely proportional to C.')
        c1 = st.sidebar.slider('C1', 1, 7, 1)
        c2 = st.sidebar.slider('C2', 8, 14, 10)
        # c3 = st.sidebar.slider('C3', 15, 20, 20)
        
        # parameters = {'C':[c1, c2, c3], 'kernel':kernel_type}
        parameters = {'C':[c1, c2], 'kernel':kernel_type}
        model = svm.SVR()
    elif clf_name == 'RF':
        st.sidebar.write('**Number of Estimators**')
        st.sidebar.write('The number of trees in the forest.')
        n1 = st.sidebar.slider('n_estimators1', 1, 40, 5)
        n2 = st.sidebar.slider('n_estimators2', 41, 80, 50)
        # n3 = st.sidebar.slider('n_estimators3', 81, 120, 100)
        st.sidebar.header('')
        
        st.sidebar.write('**Max depth**')
        st.sidebar.write('The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.')
        md1 = st.sidebar.slider('max_depth1', 1, 7, 1)
        md2 = st.sidebar.slider('max_depth2', 8, 14, 10)
        # md3 = st.sidebar.slider('max_depth3', 15, 20, 20)
        
        # parameters = {'n_estimators':[n1, n2, n3], 'max_depth':[md1, md2, md3]}
        parameters = {'n_estimators':[n1, n2], 'max_depth':[md1, md2]}
        model = RandomForestRegressor()
    
    elif clf_name == 'SGD':
        st.sidebar.write("**Penalty**")
        st.sidebar.write('Used to specify the norm used in the penalization.')
        penalty = st.sidebar.multiselect('', options=['l1', 'l2'], default=['l1', 'l2'])
        st.sidebar.subheader('')
        
        st.sidebar.write('**Learning Rate**')
        st.sidebar.write("The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.01.")
        lr = st.sidebar.selectbox('learning_rate', ['constant', 'optimal', 'invscaling' ,'adaptive'])
        
        parameters = {'penalty':penalty, 'learning_rate':[lr]}
        model = SGDRegressor()
    else:
        parameters = {}
        model = LinearRegression()
        
    return model, parameters

model, parameters = get_classifier(classifier)

clf = GridSearchCV(estimator=model, param_grid=parameters, cv=cv_count, return_train_score=False)
clf.fit(train_scale, y_train)

df1 = pd.DataFrame(clf.cv_results_)

st.header('Tuning Results')
results_df = st.multiselect('', options=['mean_fit_time', 'std_fit_time', 'mean_score_time', 
                                         'std_score_time', 'split0_test_score', 'split1_test_score', 
                                         'std_test_score', 'rank_test_score'], 
                            default=['mean_score_time', 'std_score_time', 
                                     'split0_test_score', 'split1_test_score'])
df_results = df1[results_df]
st.write(df_results)

st.subheader('**Parameters and Mean test score**')
st.write(df1[['params', 'mean_test_score']])
st.write('Best Score:', clf.best_score_)
st.write('Best Parameters:', clf.best_params_)


in_df = user_input_features()
prediction = clf.predict(in_df)

st.write("""## Angle Prediction""")
st.write(prediction[0])



# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import pickle

# st.write("""# Angle Visualization""")

# load_reg=pickle.load(open('SV_reg.pkl', 'rb'))
# df = pd.read_csv('Data_Pred.csv')
# st.write(df.iloc[:,1:].head())

# st.write("""# Model Perfromance""")
# st.write(px.line(df[(df.index>0) & (df.index<100)],x='index',y=['Angle','SVR_Pred_Angle','LR_Pred_Angle','RF_Pred_Angle','SGD_Pred_Angle']))

# st.sidebar.header('User Input Features')
# def user_input_features():
#     current_range = st.sidebar.slider('Current (A)', -2.5,2.5,0.0,0.1)
#     voltage_range = st.sidebar.slider('Voltage (V)', -10.0,10.0,0.0,0.1)

#     data = {'Current ': current_range, 'voltage': voltage_range}
#     features = pd.DataFrame(data,index=[0])
#     return features

# in_df = user_input_features()
# prediction = load_reg.predict(in_df)

# st.subheader('Angle Prediction')
# st.write(prediction[0])

# # if selected_day == 'Tue':
# #     date=df_ice.loc[df_ice.Day_Name == selected_day,'Date(DD/MM/YYYY)']
# #     data=df_ice.loc[df_ice.Day_Name == selected_day].copy()

# #     fig = make_subplots(rows=16, cols=1, subplot_titles=tuple(date.unique()))
# #     r,c=1,1
# #     for v_idx,v in enumerate(date.unique()[:]):
# #         temp=data[date==v]
# #         fig.append_trace(go.Scatter(x=temp['Time'],y=temp['KWH'],name='KWH_ICE'),row=r, col=c)
# #         r+=1
# #     fig.update_layout(height=3000, width=1000)
# #     fig.show()

