from enum import unique

from altair.vegalite.v4.api import value
from pandas.core.frame import DataFrame
import streamlit as st

import pandas as pd
import json

from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt

import altair as alt
import pandas as pd


import string

import pyodbc

st.title('How the sentiment of feedback changes overtime with changes in app version')

@st.cache(suppress_st_warning=True)

def user_read():
    with open('usr.json') as f:
        data = json.load(f)
    usrs  = pd.read_json(data, orient='records')
    usrs_pivot =  usrs.pivot(index='user', columns='version', values='date')
    return usrs_pivot[['alpha01_date','beta01_date', 'alpha02_date',  'beta02_date']]

usrs_pivot = user_read()

df=pd.read_csv('df_.csv')

df = pd.merge(df, usrs_pivot, on = 'user', how = 'left')

df = df.replace({'pintrest':'theapp', 'pintrest':'theapp', 'Dropbox':'theapp', 'dropbox':'theapp', 'Evernote':'theapp','evernote':'theapp', 'Yelp':'theapp', 'Foursquare':'theapp', 'T rip advisor':'theapp','yelp':'theapp', 'foursquare':'theapp', 'trip advisor':'theapp'})
df['alpha01_date'] = pd.to_datetime(df['alpha01_date'], errors='coerce')
df['beta01_date'] = pd.to_datetime(df['beta01_date'], errors='coerce')
df['alpha02_date'] = pd.to_datetime(df['alpha02_date'], errors='coerce')
df['beta02_date'] = pd.to_datetime(df['beta02_date'], errors='coerce')
df['date_review'] = pd.to_datetime(df['date_review'], errors='coerce')

df['alpha01']=df['date_review'] <df['beta01_date'] 
df['beta01']=(df['date_review'] <df['alpha02_date']) *( df['date_review'] >=df['beta01_date'] )
df['alpha02']=(df['date_review'] <df['beta02_date']) *( df['date_review'] >=df['alpha02_date'] )
df['beta02']=(df['date_review'] >=df['beta02_date']) 

df['alpha01'] = df['alpha01']*1
df['beta01'] = df['beta01']*1
df['alpha02'] = df['alpha02']*1
df['beta02'] = df['beta02']*1

def get_version(row):
    for c in ['alpha01','beta01','alpha02','beta02']:
        if row[c]==1:
             return c

version = df.apply(get_version, axis=1)


df['version'] = version 

df=df[['id','review','class','user','date_review','rating','version']]


def no_punkt(strin):
    punc = string.punctuation
    for ele in strin:  
        if ele in punc:  
            strin = strin.replace(ele, "") 
    return strin
df['review'] = df['review'].apply(lambda x: no_punkt(x) )

df['review'] = df['review'].apply(lambda x: x.lower().split())


stop = ['yours', 'nor', 'theirs', 'shouldn\'t', 'just', 'how', 'from', 'same', 'it\'s', 'what', 'and', 'below', 'had', 'at', 'ain', 'about', 'themselves', 'needn', 'doesn\'t', 'into', 'no', 'she\'s', 'shouldn', 'himself', 'when', 'doing', 'you\'d', 'yourselves', 'will', 'ourselves', 'isn', 'our', 'own', 'your', 'yourself', 'that', 'wouldn\'t', 'until', 'been', 'such', 'aren', 'mustn', 'why', 'there', 'don\'t', 'any', 'being', 'up', 'before', 'herself', 'he', 'only', 'can', 'that\'ll', 'i', 'my', 'do', 'her', 's', 'other', 'couldn', 'but', 'needn\'t', 'this', 'll', 'hers', 'with', 'each', 'haven\'t', 'over', 'she', 'its', 'be', 'we', 'weren', 'all', 'here', 'as', 'after', 'hadn\'t', 'won\'t', 'didn', 'am', 're', 'does', 'isn\'t', 'above', 'mustn\'t', 'itself', 'then', 'very', 'whom', 'most', 'down', 'did', 'or', 'again', 'which', 'both', 'should', 'further', 'won', 'ours', 'through', 'more', 'once', 'doesn', 'didn\'t', 'who', 'm', 'shan', 'you', 'shan\'t', 'under', 'hadn', 'should\'ve', 'you\'re', 'out', 'few', 'me', 'them', 'their', 'these', 'don', 'having', 'haven', 'if', 'than', 'now', 'so', 'you\'ll', 'o', 'you\'ve', 'hasn\'t', 'mightn', 'wasn', 'while', 'are', 'is', 'some', 'wasn\'t', 'they', 'weren\'t', 'wouldn', 'it', 'those', 'of', 'to', 'have', 't', 'for', 'hasn', 'the', 'during', 'aren\'t', 'has', 'd', 'by', 'against', 'was', 'because', 'off', 'not', 'his', 'ma', 'y', 'a', 'couldn\'t', 'mightn\'t', 'him', 'in', 'where', 'myself', 'were', 'an', 'too', 'on', 've', 'between']
df['review'] = df['review'].apply(lambda x: [item for item in x if item not in stop])

def ngrams(input, n):
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
       
    return output



chart1 = alt.Chart(df).mark_bar(opacity = 0.5).encode(
    x='date_review:T'
    , y=alt.Y('count():Q', stack="normalize"),
    color='version'
)



chart2 = alt.Chart(df).mark_bar().encode(
    x='date_review:T',
    y='count():Q',
    color='rating:N'
)


def get_image(df,   n):
    
    ngram_n = n
    df['ngrams'] = df['review'].apply(lambda x: ngrams(x, int(ngram_n)))
    main_dict = {}
    for i, r in df.iterrows():
        for key, v in r['ngrams'].items():
            if key  not in main_dict:
                main_dict[key] = v
            else:
                main_dict[key] += v
    lstk=[]
    lstv=[]
    for k, v in main_dict.items():
        lstk.append("_".join(k.split()))
        lstv.append(v)
    
    data = {'ngram': lstk, 'count': lstv}
    
    ngram_df = pd.DataFrame.from_dict(data)
    wc = WordCloud(width=1000, height=1000).generate_from_text(  " ".join(ngram_df['ngram'] ))
    plt.figure(figsize=(40,40) )
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    


def get_table(df,  n, k):
    ngram_n = n 
    ngrumnk = k 
    
    df['ngrams'] = df['review'].apply(lambda x: ngrams(x, int(ngram_n)))
    
    main_dict = {}
    for i, r in df.iterrows():
        for key, v in r['ngrams'].items():
            if key  not in main_dict:
                main_dict[key] = v
            else:
                main_dict[key] += v
    lstk=[]
    lstv=[]
    for k, v in main_dict.items():
        lstk.append("_".join(k.split()))
        lstv.append(v)
    data = {'ngram': lstk, 'count': lstv}
    return pd.DataFrame.from_dict(data).sort_values(by = 'count', ascending=False).head(int(ngrumnk))










my_button = st.sidebar.radio("Some numbers", ('Home','Data Sources', 'Preprocessing / N-grams', 'TimeSeries', 'Word Clouds', 'Conclusion')) 

if my_button == 'Home':
    st.write('Software release life cycle have several stages alpha and beta, while one of product development approaches is to tell customers what they want, the alternative approach is to collect feedback, feedback could be collected through survey and open ended questions, more interesting and useful approach is to use feedback that customer gives willingly. It could be a new future proposal, complaint about a feature, all mentioned in a review, dedicated facebook group, feedback page or forum. Product managers and Engineering Managers are interested to know what version of software custom is using at the time of the review  and if the text is one of the classes: Feature Request, Bug report/Problem Discovery), User Experience (UE) reviews as Rating (RT). What is the count and what is the overarching theme of  review of a certain class. Dataset also contains star rating so I will be able to see how successful are new software versions at addressing previously identified problems, or how often do they create new problems')

     

elif my_button == 'Data Sourses': 
   
    st.write('Software release life cycle have several stages alpha and beta, while one of product development approaches is to tell customers what they want, the alternative approach is to collect feedback, feedback could be collected through survey and open ended questions, more interesting and useful approach is to use feedback that customer gives willingly. It could be a new future proposal, complaint about a feature, all mentioned in a review, dedicated facebook group, feedback page or forum. Product managers and Engineering Managers are interested to know what version of software custom is using at the time of the review  and if the text is one of the classes: Feature Request, Bug report/Problem Discovery), User Experience (UE) reviews as Rating (RT). What is the count and what is the overarching theme of  review of a certain class. Dataset also contains star rating so I will be able to see how successful are new software versions at addressing previously identified problems, or how often do they create new problems')


elif my_button == 'TimeSeries':  
     st.write('The chart is showing proportions of positive and negative reviews, backgraound is also incoded in color as showen in labeling. As we can see by th e chart alpha01 have got more reaction a lot of it positive, some negative, beta01 have got less reaction and most negative reaction came at the beginning of teh update alpha and beta second version got th eleast amount of negative reaction and a lot of positive reaction to invrstigate the sentiment we can explore ngrams.')
    # Works
     st.altair_chart((chart1 + chart2).resolve_scale(y='independent').properties(width=650,height=400))
elif my_button == "Preprocessing / N-grams":
    
    st.write('Preprocessing of the text included tokanizing removing stop words, lemmatizng, stemming. After preprocessing it is posible to see most polpular words adjust scrollers and selctors below to see most poular words and combinations of words.')

    x = st.slider('How many ngrams?', min_value=1, value=2, max_value=5)   # 👈 this is a widget
    y = st.slider('How many rows of data?',min_value=1, value=5, max_value=55) 
    col1,col2 = st.beta_columns(2)
    with col1:
        
        
        
        option_rating_= st.slider('What is teh rating?', min_value=1, value=2, max_value=5) 
        option_version_ = st.selectbox( 'Which version?',  ('alpha01', 'beta01', 'alpha02', 'beta02'))

    with col2:
        pdf = df[df['rating']==option_rating_][df['version']== option_version_].copy()
        st.write(get_table(pdf, x, y))

    
   
    

elif my_button == 'Word Clouds':  
    

    st.title('Which combination of features would you like to see?')
    
    col1,col2 = st.beta_columns(2)
    with col1:
        option_class = st.selectbox( 'Which type of feedback?', ('PD', 'RT', 'UE', 'FR'))
        
        option_rating= st.slider('What is teh rating?', min_value=1, value=2, max_value=5) 
        option_version = st.selectbox( 'Which version?',  ('alpha01', 'beta01', 'alpha02', 'beta02'))
        ngram_ = st.slider('How many ngrams?', min_value=1, value=2, max_value=10) 

    with col2:
        pdf = df[df['class']== option_class][df['rating']==int(option_rating)][df['version']== option_version].copy()
        get_image(pdf, ngram_)



elif my_button == 'Data Prep for model':  
    pass
    





elif my_button == 'Conclusion': 
    st.write('From visualizations in previous pages it is apparent that sentiment does change overtime, with the changes in releases, whiel alpha and better have got a better engagement and better overal reviews, first releases have bigger proportions of negative reviews in alpha01 and at the beginning of beta02, investigation shows what are the themes of the reviews that causes negative review:  For example expolring ngrams in alpha01 we can see taht main concern was reading of excel files (Filter FR, rating 1, vesion alpha 01) ')


else: 

    st.write('I am using mobile application feedback (I am doing that to avoid breaking my NDA) I will load it to SQL so it will look like something I would use in a real workplace. ') 

    if st.checkbox('Show users dataset'):
        st.write(usrs_pivot.head())
    st.write('Secondly I am going to use a randomly generated table that resembles real tables that I would use to understand customers’ software build version')
    
    if st.checkbox('Show reviews dataset'):
        st.write(df.head())

