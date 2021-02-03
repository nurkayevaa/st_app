from enum import unique
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import json
import nltk
nltk.download('punkt')

from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import plotly.tools
import altair as alt
import pandas as pd


import string

import pyodbc
print( set(stopwords.words('english')))

st.title('How the sentiment of feedback changes overtime with changes in app version')

@st.cache(suppress_st_warning=True)
def user_read():
    with open('usr.json') as f:
        data = json.load(f)
    usrs  = pd.read_json(data, orient='records')
    usrs_pivot =  usrs.pivot(index='user', columns='version', values='date')
    return usrs_pivot[['alpha01_date','beta01_date', 'alpha02_date',  'beta02_date']]

usrs_pivot = user_read()

@st.cache(suppress_st_warning=True)
def get_data():
    server = 'appreviews.database.windows.net'
    database = 'appreviews'
    username = 'anel'
    password = 'Exbnmcz3hfpf!'   
    driver= '{ODBC Driver 17 for SQL Server}'
    lst = []
    lst2 = []
    lst3 =[]
    lst4 = []
    lst5 = []
    lst6 = []
    with pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT  id, review, class,  usr, date_review, rating  FROM    apps_maalej_") #SELECT  * FROM dbo.apps_pan_1000 UNION ALL 
            row = cursor.fetchone()
            while row:
                lst.append(str(row[0]))
                lst2.append(str(row[1]))
                lst3.append(str(row[2]))
                lst4.append(str(row[3]))
                lst5.append(str(row[4]))
                lst6.append(str(row[5]))
                row = cursor.fetchone()
    
    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    return  pd.DataFrame(list(zip(lst, lst2, lst3, lst4, lst5, lst6)),  columns =['id', 'review', 'class', 'user', 'date_review', 'rating']) 

df = get_data()


df = pd.merge(df, usrs_pivot, on = 'user', how = 'left')

df = df.replace({'pintrest':'theapp', 'pintrest':'theapp', 'Dropbox':'theapp', 'Dropbox':'theapp', 'Evernote':'theapp','evernote':'theapp'})
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
    for c in df.columns:
        if row[c]==1:
             return c

version = df.apply(get_version, axis=1)


df['version'] = version 

df_=df[['id','review','class','user','date_review','rating','version']]





def no_punkt(strin):
    punc = string.punctuation
    for ele in strin:  
        if ele in punc:  
            strin = strin.replace(ele, "") 
    return strin
df['review'] = df['review'].apply(lambda x: no_punkt(x) )




w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df['review'] = df['review'].apply(lambda x: lemmatize_text(x) )

# Use English stemmer.
stemmer = SnowballStemmer("english")
df['review'] = df['review'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.

stop = set(stopwords.words('english'))
#df['review'] = df['review'].str.lower().str.split()
df['review'] = df['review'].apply(lambda x: [item for item in x if item not in stop])

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





@st.cache(suppress_st_warning=True)
def get_table(df1, rating , class_, version,  n, k):
    df1 = df1[df1['version']== version]
    if rating  =='none' and class_ =='none':
        df = df1.copy()
        
    elif  rating =='none' and class_!='none' :
        df = df1[df1['class']==class_].copy()
    elif  rating !='none'  and class_ =='none':
        df = df1[df1['rating']==rating].copy()
    elif  rating !='none'  and class_ !='none':
        df = df1[df1['rating']==rating][df1['class']==class_].copy()
    lst = []
    for i in (df['review']):
        lst.extend(i)
    token = nltk.word_tokenize(" ".join(lst))
    bigrams = ngrams(token,n)
    st.write(pd.DataFrame.from_dict(Counter(bigrams), orient='index').reset_index().rename(columns={'index' :'ngram', 0:"count" }).sort_values(by = 'count', ascending=False).head(k))

def convert_ngram(mydict):
    df = pd.DataFrame.from_dict(mydict).reset_index()
    df['ngram'] = df[df.columns[1:].tolist()].apply(lambda x: '_'.join(x), axis = 1) 
    return df

@st.cache(suppress_st_warning=True)
def get_image(df1, rating , class_, version,  ngram_ = 'bigrams'):
    df1 = df1[df1['version']== version]
    if rating  =='none' and class_ =='none':
        df = df1.copy()
        
    elif  rating =='none' and class_!='none' :
        df = df1[df1['class']==class_].copy()
    elif  rating !='none'  and class_ =='none':
        df = df1[df1['rating']==rating].copy()
    elif  rating !='none'  and class_ !='none':
        df = df1[df1['rating']==rating][df1['class']==class_].copy()
    lst = []
    
    for i in (df['review']):
        lst.extend(i)
    token = nltk.word_tokenize(" ".join(lst))
    bigrams = ngrams(token,2)
    trigrams = ngrams(token,3)
    fourgrams = ngrams(token,4)
    fivegrams = ngrams(token,5)
    bigrams = convert_ngram(bigrams)
    trigrams = convert_ngram(trigrams)
    fourgrams = convert_ngram(fourgrams)
    fivegrams = convert_ngram(fivegrams)
    ngram_df =( bigrams if ngram_ == 'bigrams' else   trigrams if ngram_ == 'trigrams' else  fourgrams if ngram_ == 'fourgrams' else fivegrams)
    wc = WordCloud(width=1000, height=1000).generate_from_text(  " ".join(ngram_df['ngram'] ))
    plt.figure(figsize=(40,40) )
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



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

    x = st.slider('How many ngrams?')  # 👈 this is a widget
    y = st.slider('How many rows of data?')
    col1,col2 = st.beta_columns(2)
    with col1:
        option_class_ = st.selectbox( 'Which type of feedback?',('PD', 'RT', 'UE', 'FR', 'none'))
        option_rating_= st.selectbox( 'Which rating?', ('1','2','3','4','5', 'none'))
        option_version_ = st.selectbox( 'Which version?',  ('alpha01', 'beta01', 'alpha02', 'beta02'))

    with col2:
        get_table(df, option_rating_ ,option_class_, option_version_, x,y)

    
   
    

elif my_button == 'Word Clouds':  
    

    st.title('Which combination of features would you like to see?')
    
    col1,col2 = st.beta_columns(2)
    with col1:
        
        option_ngram= st.selectbox('Which ngrams?', ('bigrams','trigrams','fourgrams','fivegrams'))
        option_class = st.selectbox( 'Which type of feedback?', ('PD', 'RT', 'UE', 'FR', 'none'))
        option_rating= st.selectbox( 'Which rating?',  ('1','2','3','4','5', 'none'))
        option_version = st.selectbox( 'Which version?',  ('alpha01', 'beta01', 'alpha02', 'beta02'))

    with col2:
        
        get_image(df, option_rating ,option_class, option_version, ngram_ = 'fivegrams')


elif my_button == 'Conclusion': 
    st.write('From visualizations in previous pages it is apparent that sentiment does change overtime, with the changes in releases, whiel alpha and better have got a better engagement and better overal reviews, first releases have bigger proportions of negative reviews in alpha01 and at the beginning of beta02, investigation shows what are the themes of the reviews that causes negative review:  For example expolring ngrams in alpha01 we can see taht main concern was reading of excel files (Filter FR, rating 1, vesion alpha 01) ')


else: 

    st.write('I am using mobile application feedback (I am doing that to avoid breaking my NDA) I will load it to SQL so it will look like something I would use in a real workplace. ') 

    if st.checkbox('Show users dataset'):
        st.write(users.head())
    st.write('Secondly I am going to use a randomly generated table that resembles real tables that I would use to understand customers’ software build version')
    
    if st.checkbox('Show reviews dataset'):
        st.write(df.head())

