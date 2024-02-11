# Twitter_Sentiment_Analysis

### Problem Statement

The following project is about analyzing the sentiments of tweets on social networking website
‘Twitter’. The dataset for this project is scraped from Twitter. It contains 1,600,000 tweets
extracted using Twitter API. It is a labeled dataset with tweets annotated with the sentiment (0 =
negative, 2 = neutral, 4 = positive).
It contains the following 6 fields:

1. target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
2. ids: The id of the tweet .
3. date: The date of the tweet (Sat May 16 23:58:44 UTC 2009)
4. flag: The query. If there is no query, then this value is NO_QUERY.
5. user: The user that tweeted
6. text: The text of the tweet.
   Design a classification model that correctly predicts the polarity of the tweets provided in the
   dataset.

### Tools Used

1. Python
2. Scikit-Learn
3. NLTK
4. Seaborn
5. Matplotlib

### Steps Followed

1. First the tweets column is preprocessed before estimating the polarity or sentiment.
2. Some amount of Exploratory Data Analysis is done on the preprocessed tweets column.
3. Vectorization Methods ( CountVectorizer & TFIDFVectorizer ) is used to convert the sentences in the tweets column into numerical vectors that will be required for the ML Model.
4. Finally, a classification model is built to predict the polarity.

### Classification Model Used
Logistic Regression is used for this usecase.
Model Reports :
From various models built and tested, 
* Two vectorizer Method gives almost same result.
* Logistic Regression with tfidf Vectorizer/Count Vectorizer has  the following Metrics for evaluation :
   1. Confusion Matrix:
      
      [[181728  57633]
      
       [ 48476 192163]]
      
   2.  Classification Report:
      
                            precision    recall  f1-score   support
       
           0                   0.79      0.76      0.77    239361
           4                   0.77      0.80      0.78    240639
           
           accuracy             -           -       0.78   480000
           macro avg 0.78      0.78       0.78      0.78   480000
           weighted avg        0.78       0.78       0.78  480000


   3. Accuracy: 0.7789395833333334
   4.  Precision: 0.7692797322615254
   5.  F1_score: 0.7836430923567852
   6.  ROC :  0.7788872231512126
   7.   ROC Curve :
![image](https://github.com/AasthaMuk/Module-21_DS-Projects/assets/53363503/dd065523-ad9e-4e24-989f-1e174558934f)


