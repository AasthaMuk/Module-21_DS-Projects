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
                        background: url("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700008778-1200x675.jpg");
                        background-size: cover
                    }
                    p{
                      color:white;
                    }

            </style>""",unsafe_allow_html=True)
        


        
def home_page():
        left,right = st.columns((3,1))
        with left:
            st.markdown('<p style="color: #f2d583; font-size:45px; font-weight:bold">Prediction of Guvi Course Ratings</p>',unsafe_allow_html=True)
            st.markdown("""<p style="color: white; font-size:20px; font-weight:bold"> This application is mainly used to predict the ratings of Guvi courses and also do  some data analysis,exploration and vizualizations.</p>""",unsafe_allow_html=True)
            st.markdown('<br>',unsafe_allow_html=True)
            st.markdown("""<p style="color: white; font-size:18px; font-weight:bold">Click on the <span style="color: red; font-size:18px; font-weight:bold">Tabs below</span> to start exploring.</p>""",unsafe_allow_html=True)
            st.markdown('<p style="color: white; font-size:25px; font-weight:bold">TECHNOLOGIES USED :</p>',unsafe_allow_html=True)
            st.markdown("""
                            <p style="color: white; font-size:18px; font-weight:bold">*<span style="color: red; font-size:18px; font-weight:bold"> Python</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Streamlit</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Matplotlib</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Seaborn</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Scikit-Learn</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Pickle</span></p>""",unsafe_allow_html=True)

        with right:
             st.image("images/img.gif",use_column_width=True)



def charts():
    st.markdown('<p style="color: white; font-size:30px; font-weight:bold">Some EDA Charts</p>',unsafe_allow_html=True)
    df = pd.read_csv(r'C:\Users\aasth\OneDrive\Desktop\Guvi_Projects_Module_21\dataset\3.1-data-sheet-guvi-courses.csv')
    col1,col2,col3 = st.columns([1,0.2,1],gap="small")
    with col1:
        st.markdown('<p style="color: white; font-size:20px; font-weight:bold">Courses for each level</p>',unsafe_allow_html=True)
        fig = px.histogram(df, nbins=30, x="level", 
                       color="level",
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(height=500,width=500)
        st.plotly_chart(fig,use_container_width=False)
    with col3:
        st.markdown('<p style="color: white; font-size:20px; font-weight:bold">Rating for 10 Courses</p>',unsafe_allow_html=True)
        fig = px.bar(df[0:10],
                        x="course_id",
                        y="Rating",
                        color="Rating",
                        orientation='v',
                        color_continuous_scale=px.colors.sequential.Inferno)
        fig.update_layout(height=500,width=500)
        st.plotly_chart(fig,use_container_width=False)

    
    st.markdown('<p style="color: white; font-size:20px; font-weight:bold">Courses for each subject</p>',unsafe_allow_html=True)
    subject = df['subject'].value_counts().index
    fig = px.histogram(df, nbins=len(subject), x='subject',
                        color="subject",
                        color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(
            xaxis_title='Subjects',
            yaxis_title='No. of Courses'
        )
    fig.update_layout(height=500,width=500)
    st.plotly_chart(fig,use_container_width=False)
    
    st.markdown('<p style="color: white; font-size:20px; font-weight:bold">Courses based on content duration</p>',unsafe_allow_html=True)
    df1 = df.copy()
    df1.dropna(inplace=True)
    st.line_chart(df1,x='course_id',y='content_duration',color='subject',height=500,width=900,use_container_width=False) 
        
    with col1:
        st.markdown('<p style="color: white; font-size:20px; font-weight:bold">No. of Lectures based on Level</p>',unsafe_allow_html=True)
        grouped_data = df.groupby('level')['num_lectures'].sum()
        fig = px.pie(grouped_data, 
                                    names=grouped_data.index,
                                    color_discrete_sequence=px.colors.sequential.Darkmint_r,
                                    values='num_lectures')
        fig.update_layout(height=400,width=300)
        st.plotly_chart(fig,use_container_width=True)
    with col3:
        st.markdown('<p style="color: white; font-size:20px; font-weight:bold">Content Duration based on Subject</p>',unsafe_allow_html=True)
        grouped_data = df.groupby('subject')['content_duration'].sum()
        fig = px.pie(grouped_data,
                                    names=grouped_data.index,
                                    color_discrete_sequence=px.colors.sequential.RdBu,
                                    values='content_duration')
        fig.update_layout(height=400,width=300)
        st.plotly_chart(fig,use_container_width=True)


    # st.subheader("Resale Price Based on Town")
    # fig= go.Figure(data=[go.Pie(labels=df['town'], values=df['resale_price_log'], hole=0.5)])
    # fig.update_layout(title_text='Resale Price Vs Town')
    # fig.update_layout(height=650,width=500)
    # st.plotly_chart(fig)     






# def pie_charts():
#      df = pd.read_csv("final_data.csv")
#      col1,col2 = st.columns([1,1],gap="small")
#      with col1:
#         st.subheader("Floor Area Based on Flat Model")
#         grouped_data = df.groupby('flat_model')['floor_area_sqm_log'].sum()
#         fig = px.pie(grouped_data, 
#                                     title='Floor Area vs Flat Model',
#                                     names=grouped_data.index,
#                                     color_discrete_sequence=px.colors.sequential.Agsunset,
#                                     values='floor_area_sqm_log')
#         fig.update_layout(height=650,width=500)
#         st.plotly_chart(fig,use_container_width=True)
#      with col2:
#         st.subheader("Floor Area Based on Flat Type")
#         grouped_data = df.groupby('flat_type')['floor_area_sqm_log'].sum()
#         fig = px.pie(grouped_data, 
#                                     title='Floor Area vs Flat Type',
#                                     names=grouped_data.index,
#                                     color_discrete_sequence=px.colors.sequential.RdBu,
#                                     values='floor_area_sqm_log')
#         fig.update_layout(height=650,width=500)
#         st.plotly_chart(fig,use_container_width=True)


#      st.subheader("Resale Price Based on Town")
#      fig = go.Figure(data=[go.Pie(labels=df['town'], values=df['resale_price_log'], hole=0.5)])
#      fig.update_layout(title_text='Resale Price Vs Town')
#      fig.update_layout(height=650,width=500)
#      st.plotly_chart(fig)




def set_main_page():
        home_page()
        tab1,tab2 = st.tabs(["Visualizations", "Predict Rating"])
        
        with tab1:
            charts()
        
        with tab2:
            st.markdown('<p style="color: white; font-size:30px; font-weight:bold">Predicting Rating</p>',unsafe_allow_html=True)
            
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
                        
                    submit_button = st.form_submit_button(label="PREDICT RATING",type="primary")

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
                                [[int(subject_dict[subject]), np.log(int(price)), np.log(int(subscriber_no)), np.log(int(review_no)), np.log(int(lecture_no)), np.log(int(content_duration)), int(level)]])
                            new_sample = scaler_loaded.transform(new_sample[:, :8])

                            new_pred = loaded_model.predict(new_sample)[0]
                            rating=str(np.exp(new_pred))
                            s="<p style='color: #8B120E; font-size:45px; font-weight:bold'>Predicted Rating: "+rating+"</p>"
                            st.balloons()
                            st.markdown(s,unsafe_allow_html=True)
                            
                        
                            # evalution metrics
                            with st.container():
                                actual = pd.DataFrame(index=[0],data=[0.7])
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
                                st.markdown("<p style='color: white; font-size:30px; font-weight:bold'>Evaluation Metrics</p>",unsafe_allow_html=True)
                                st.markdown("<p style='color: white; font-size:25px; font-weight:bold'>Mean squared error:&emsp;"+str(float(normalized_mse))+"</p",unsafe_allow_html=True)
                                st.markdown("<p style='color: white; font-size:25px; font-weight:bold'>Mean absolute error:&emsp;"+str(float(normalized_mae))+"</p",unsafe_allow_html=True)
                                st.markdown("<p style='color: white; font-size:25px; font-weight:bold'>Root mean squared error:&emsp;"+str(float(rmse))+"</p",unsafe_allow_html=True)
                        
                        

            
            
        
           
                    


#------------ Run the app --------------#
set_page_config()
set_main_page()