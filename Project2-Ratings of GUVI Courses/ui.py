# Importing the libraries
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import plotly.graph_objects as go

#---------------- Select the options -------------------------------------#
levels = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
subject_dict ={'Subject: Web Development':3,'Business Finance':0,'Graphic Design':1,'Musical Instruments':2}
#------------------------------------------------------------------------#


#--------------------------- Functions ---------------------------------#
def set_page_config():
    icon = Image.open("images/icon.png")
    st.set_page_config(page_title= "Rating of Guvi Courses",
                        page_icon= icon,
                        layout= "wide",
                        initial_sidebar_state= "expanded",
                        menu_items={'About': """# This dashboard app is created by Aastha Mukherjee!"""})
        
    st.markdown(""" 
            <style>
                    .stApp,[data-testid="stHeader"] {
                        background: url("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700014252.jpg");
                        background-size: cover
                    }

                    
                    .stSpinner,[data-testid="stMarkdownContainer"],.uploadedFile{
                       color:black !important;
                    }

                    [data-testid="stSidebar"]{
                       background: url("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700014168.jpg");
                       background-size: cover
                    }

                    .stButton > button,.stDownloadButton > button {
                        background-color: #f54260;
                        color: black;
                    }

                    #custom-container {
                        background-color: #0B030F !important;
                        border-radius: 10px; /* Rounded corners */
                        margin: 20px; /* Margin */
                        padding: 20px;
                    }

            </style>""",unsafe_allow_html=True)
        

def style_submit_button():
    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #37a4de;
                                                        color: white!important;
                                                        width: 45%}
                    </style>
                """, unsafe_allow_html=True)
        
def home_page():
        left,right = st.columns((1,3))
        with right:
            st.markdown('<p style="color: black; font-size:45px; font-weight:bold">Singapore Resale Price Prediction</p>',unsafe_allow_html=True)
            st.markdown("""<p style="color: black; font-size:20px; font-weight:bold"> This application is mainly used to predict the HDB Flats managed by Singapore Government Agency and also do  some data analysis,exploration and vizualizations.</p>""",unsafe_allow_html=True)
            st.markdown('<br>',unsafe_allow_html=True)
            st.markdown("""<p style="color: black; font-size:18px; font-weight:bold">Click on the <span style="color: red; font-size:18px; font-weight:bold">Sidebar Menus</span> option to start exploring.</p>""",unsafe_allow_html=True)
            st.markdown('<p style="color: black; font-size:25px; font-weight:bold">TECHNOLOGIES USED :</p>',unsafe_allow_html=True)
            st.markdown("""
                            <p style="color: black; font-size:18px; font-weight:bold">*<span style="color: red; font-size:18px; font-weight:bold"> Python</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Streamlit</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Matplotlib</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Seaborn</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Scikit-Learn</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Pickle</span></p>""",unsafe_allow_html=True)





def bar_charts():
    df = pd.read_csv("final_data.csv")
    st.title("Some Bar Charts :")
    col1,col2,col3 = st.columns([1,0.2,1],gap="small")
    with col1:
        st.subheader("Flat Types present in Towns")
        fig = px.bar(df[0:7000],
                    title='Town Vs Flat Type',
                    x="town",
                    y="flat_type",
                    color="flat_type",
                    orientation='v',
                    color_continuous_scale=px.colors.sequential.Inferno)
        fig.update_layout(height=500,width=500)
        st.plotly_chart(fig,use_container_width=False)
    with col3:
        st.subheader("Resale Price based on Flat Type")
        fig = px.bar(df[0:7000],
                        title='Resale Price Vs Flat Type',
                        x="flat_type",
                        y="resale_price_log",
                        color="flat_type",
                        orientation='v',
                        color_continuous_scale=px.colors.sequential.Inferno)
        fig.update_layout(height=500,width=500)
        st.plotly_chart(fig,use_container_width=False)

    st.subheader("Count of HDB Flats by Town")
    fig = px.histogram(df, nbins=30, x="town", 
                       color="town", title="Flats Vs Town", 
                       color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(
        xaxis_title='Towns',
        yaxis_title='No. Of HDB Flats'
    )
    fig.update_layout(height=500,width=800)
    st.plotly_chart(fig,use_container_width=False)

    block = df['block'].value_counts().index
    df_sorted = df.sort_values(by='block', ascending=False)
    top_100_descending = df_sorted.head(100)
    st.subheader("Count of HDB Flats for Resale based on Top 100 Blocks")
    fig = px.histogram(top_100_descending, nbins=len(block), x='block',
                       color="block", title="No. of Flats Vs Block", 
                       color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(
        xaxis_title='Block Name',
        yaxis_title='Count Of HDB Flats'
    )
    fig.update_layout(height=500,width=800)
    st.plotly_chart(fig,use_container_width=False)






def pie_charts():
     df = pd.read_csv("final_data.csv")
     col1,col2 = st.columns([1,1],gap="small")
     with col1:
        st.subheader("Floor Area Based on Flat Model")
        grouped_data = df.groupby('flat_model')['floor_area_sqm_log'].sum()
        fig = px.pie(grouped_data, 
                                    title='Floor Area vs Flat Model',
                                    names=grouped_data.index,
                                    color_discrete_sequence=px.colors.sequential.Agsunset,
                                    values='floor_area_sqm_log')
        fig.update_layout(height=650,width=500)
        st.plotly_chart(fig,use_container_width=True)
     with col2:
        st.subheader("Floor Area Based on Flat Type")
        grouped_data = df.groupby('flat_type')['floor_area_sqm_log'].sum()
        fig = px.pie(grouped_data, 
                                    title='Floor Area vs Flat Type',
                                    names=grouped_data.index,
                                    color_discrete_sequence=px.colors.sequential.RdBu,
                                    values='floor_area_sqm_log')
        fig.update_layout(height=650,width=500)
        st.plotly_chart(fig,use_container_width=True)


     st.subheader("Resale Price Based on Town")
     fig = go.Figure(data=[go.Pie(labels=df['town'], values=df['resale_price_log'], hole=0.5)])
     fig.update_layout(title_text='Resale Price Vs Town')
     fig.update_layout(height=650,width=500)
     st.plotly_chart(fig)




def set_sidebar():
        with st.sidebar:
            selected = option_menu('Menu', ['Home Page',"Some Visualizations","Predict Rating"],
                    icons=["house",'geo-fill','gear','flag','star'],
                    menu_icon= "menu-button-wide",
                    default_index=0,
                    styles={"nav-link": {"font-size": "15px", "text-align": "left", "margin": "-2px", "--hover-color": "#6F36AD"},
                            "nav-link-selected": {"background-color": "#B1A3F7"}})

        if selected == 'Home Page':
            home_page()

        if selected == 'Some Visualizations':
            bar_charts()
            pie_charts()
        
        
        if selected == 'Predict Value':
            st.markdown('<p style="color: black; font-size:45px; font-weight:bold">Predicting the Resale Price</p>',unsafe_allow_html=True)
            
            with st.form("Predict_Rating"):
                    col1,col2,col3 = st.columns([0.5,0.1,0.5])
                    # -----New Data inputs from the user for predicting the resale price-----
                    with col1:
                        course_id = st.text_input("Course Id")
                        course_name = st.text_input("Course Name")
                        subject = st.selectbox(label='Subject', options=list(subject_dict.keys()))
                        price = st.text_input("Price")
                        subscriber_no = st.text_input("No. of Subscribers")
                    with col3:
                        review_no = st.text_input("No. of Reviews")
                        lecture_no = st.text_input("No. of Lectures")
                        content_duration = st.text_input("Content Duration")
                        level = st.selectbox(label='Level', options=levels)
                        
                    submit_button = st.form_submit_button(label="PREDICT RATING")

                    if submit_button:
                        with st.spinner('Please wait, Work in Progress.....'):
                            time.sleep(2)
                            with open(r"regression_Rating_model.pkl", 'rb') as file:
                                print("loading regression model")
                                loaded_model = pickle.load(file)

                            with open(r'scaler.pkl', 'rb') as f:
                                print("loading scaler")
                                scaler_loaded = pickle.load(f)

                            # -----Sending the user enter values for prediction to our model-----
                            new_sample = np.array(
                                [[subject_dict[subject], price, subscriber_no, review_no, lecture_no, content_duration, level]])
                            new_sample = scaler_loaded.transform(new_sample[:, :7])

                            new_pred = loaded_model.predict(new_sample)[0]
                            rating=str(np.exp(new_pred))
                            s="<p style='color: #8B120E; font-size:45px; font-weight:bold'>Predicted Rating: "+rating+"</p>"
                            st.balloons()
                            st.markdown(s,unsafe_allow_html=True)
                            
                        
                            # evalution metrics
                            with st.container():
                                actual = pd.DataFrame(index=[0],data=[307500])
                                predicted = pd.DataFrame(index=[0],data=[np.exp(new_pred)])
                                # Flattening the data
                                actual_values = actual.values.flatten()
                                predicted_values = predicted.values.flatten()
                                mse = mean_squared_error(actual_values,predicted_values)
                                # Normalization Method : to bring the value between 0 and 1
                                normalized_mse = mse / max(np.square(actual_values), np.square(predicted_values))
                                
                                mae = mean_absolute_error(actual,predicted)
                                # Normalization Method : to bring the value between 0 and 1
                                normalized_mae = mae / max(actual_values, predicted_values)

                                rmse = np.sqrt(normalized_mse)
                                st.write(" ")
                                st.markdown("<p style='color: black; font-size:30px; font-weight:bold'>Evaluation Metrics</p>",unsafe_allow_html=True)
                                st.markdown("<p style='color: #2A7DEF; font-size:25px; font-weight:bold'>Mean squared error:&emsp;"+str(float(normalized_mse))+"</p",unsafe_allow_html=True)
                                st.markdown("<p style='color: #2A7DEF; font-size:25px; font-weight:bold'>Mean absolute error:&emsp;"+str(float(normalized_mae))+"</p",unsafe_allow_html=True)
                                st.markdown("<p style='color: #2A7DEF; font-size:25px; font-weight:bold'>Root mean squared error:&emsp;"+str(float(rmse))+"</p",unsafe_allow_html=True)
                        
                        

            
            
        
           
                    


#------------ Run the app --------------#
set_page_config()
set_sidebar()