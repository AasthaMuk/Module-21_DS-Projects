# Ratings of GUVI Courses

### Problem Statement
The following project is about Guvi Courses. The dataset for this project contains information
about Guvi courses in various categories, including course title, URL, price, number of
subscribers, number of reviews, number of lectures, course level, rating, content duration,
published timestamp, and subject. With this dataset, we can track the performance of courses
and uncover opportunities to generate revenue.

It contains the following 6 fields:

1. course_title : The title of the Guvi course. (String)
2. url : The URL of the Guvi course. (String)
3. price : The price of the Guvi course. (Float)
4. num_subscribers :The number of subscribers for the Guvi course. (Integer)
5. num_reviews : The number of reviews for the Guvi course.(Integer)
6. num_lectures : The number of lectures in the Guvi course.(Integer)
7. level : The level of the Guvi course. (String)
8. Rating : The rating of the Guvi course. (Float)
9. content_duration : The content duration of the Guvi course.(Float)
10. published_timestamp : The timestamp of when the Guvi course was published. (Datetime)
11. subject : The subject of the Guvi course. (String)
    
   Design a regression model to predict the ratings given by the learners to the course.

### Tools Used

1. Python
2. Scikit-Learn
3. Matplotlib
4. Seaborn
5. Pickle

### Steps Followed

1. Data is preprocessed.
2. Feature Engineering is performed to select only relevant features for the model.
3. Model is selected and then optimized

## Regression Model Used
1. We have used Random Forest Regressor for predicting the rating.
2. Evaluation Metrics :
   
   a) Mean squared error :  0.07290885128581288
   
   b) Mean Absolute Error :  0.2061041045419825
   
   c) Root Mean squared error :  0.2700163907725101
