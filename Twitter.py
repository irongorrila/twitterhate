
import pandas as pd
import matplotlib.pyplot as plt
import re, string


from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

###################################
'''Loading the Data & prelim analysis'''

df = pd.read_csv('TwitterHate.csv')
df.head()

df.shape
df.isnull().sum()
df['label'].value_counts()

# adding tweet length as a column
df['length'] = df.tweet.str.len()
df.head()
#####################################

'''Cleanup'''


def cleaner(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\(.*?\)', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\"+', '', text)
    text = re.sub('(\&amp\;)', '', text)
    text = re.sub('(@[^\s]+)', '', text)
    text = re.sub('(#[^\s]+)', '', text)
    text = re.sub('(rt)', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    return text


df['cleaned_tweets'] = df['tweet'].apply(lambda x: cleaner(x))
df.head(30)


tknzr = TweetTokenizer(strip_handles=True, reduce_len=True,)

# tokenizing and cleaning up terms with a length of 1
def tokenizer(text):
    tokens = tknzr.tokenize(text)
    tokens = [token for token in tokens if (token not in stopwords.words('english') and len(token) > 1)]
    
    return tokens

df['tokens'] = df["cleaned_tweets"].apply(lambda x: tokenizer(x))
df.head()

######################
#Step 5. Check out the top terms in the tweets:
######################

terms =[term for token in df.tokens for term in token]
FreqDist(terms).most_common()[:10]


######################
#6. Data formatting for predictive modeling:
#####################

lemmatizer = WordNetLemmatizer() 
lemmatized_output = []

for l in df.tokens:
    lemmed = ' '.join([lemmatizer.lemmatize(w) for w in l])
    lemmatized_output.append(lemmed)

#Assign X & y
X = lemmatized_output
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 5)

#########################
#7. TFIDF vectorization
########################
a = stopwords.words('english')

tfidf = TfidfVectorizer(stop_words=a,max_df=5000 )

# transforming tokenized data into sparse matrix format with 20K stored elements
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

########################
#8. Model building: Ordinary Logistic Regression
# class weights balanced here itself
########################

logreg = LogisticRegression(class_weight='balanced')

logreg.fit(X_train_tfidf, y_train)
pred= logreg.predict(X_test_tfidf)


#######################
#9. Model evaluation: Accuracy, recall, f_1 score
#######################

accuracy_score(y_test,pred)
print(classification_report(y_test, pred))



# Gridsearch with Hyper parameter tuning

params = {'penalty':['l1', 'l2'],'solver':['liblinear', 'newton-cg','sag'],'C':[100, 10, 1.0, 0.1, 0.01]
}

grid_cv = GridSearchCV(logreg, params, scoring= 'f1', verbose=1)
grid_cv.fit(X_train_tfidf, y_train)

grid_cv.best_score_
grid_cv.best_params_

##########################
#Find the parameters with the best recall in cross-validation
##########################

grid_cv = GridSearchCV(logreg, params, scoring= 'recall', verbose=1, cv=4)
grid_cv.fit(X_train_tfidf, y_train)

grid_cv.best_score_
grid_cv.best_params_

# creating stratified k-fold object
strat = StratifiedKFold(n_splits=4,shuffle=True, random_state=5)

strat_cv = cross_val_score(logreg,X_train_tfidf, y_train, cv=strat, n_jobs=1)

#maximum score with k fold cv
max(strat_cv)

# BEST PARAMETERS
grid_cv.best_params_

log_reg_best = LogisticRegression(C=0.1,penalty='l2',solver='liblinear')

log_reg_best.fit(X_train_tfidf, y_train)
pred_best = log_reg_best.predict(X_test_tfidf)

print(classification_report(y_test, pred_best))