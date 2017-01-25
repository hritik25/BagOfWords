
"""
This script will give a bag of words representation of the reviews in th IMDB movie review dataset
A classifier will then need to be trained using these features, for the purpose of sentiment analysis
"""
import pandas as pd
# for removing HTML markup, I am using the BeautifulSoup library
from bs4 import BeautifulSoup
import re
import nltk
# getting the stopwords list form nltk, will be used below
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer 

train = pd.read_csv('labeledTrainData.tsv', delimiter = '\t', quoting = 3)

def processedReview(rawReview):
    
    # using the library BeautifulSoup for removing html markup from the raw review string
    withoutMarkup = BeautifulSoup(rawReview)
    withoutMarkup = withoutMarkup.get_text()

    # removing punctuation(which may take away smilies used in the review), and numbers for simplicity
    lettersOnly = re.sub('[^a-zA-Z]', ' ', withoutMarkup)
    # getting all words in lower case
    lettersOnly = lettersOnly.lower()
    words = lettersOnly.split()
    
    # Stop words are words which occur frequently in the language and don't carry much meaninig
    # using sets to store stop words as they are faster for membership tests than lists
    stopWords = set(stopwords.words('english'))
    # removing the stopwords of english language from the words occured in the review
    withoutStopWords = [w for w in words if w not in stopWords]
    processedReview = ' '.join(withoutStopWords)

    return processedReview

cleanMovieReivews = []
for i in xrange(train['review'].size):
    cleanReview = processedReview(train['review'][i])
    cleanMovieReivews.append(cleanReview)   

cv = CountVectorizer(analyzer = 'word', tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
vectorizedCounts = cv.fit_transform(cleanMovieReivews)  

vectorizedCounts = vectorizedCounts.toarray()


# let's train a classifier on these bag of words features
print "Train a Random Forest Classifier..."
from sklearn.ensemble import RandomForestClassifier
# initializing a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( vectorizedCounts, train["sentiment"] )

# now, it's time to check our classifier's performance on the test data set
test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting=3 )
cleanMovieReivewsTest = []
for i in xrange(test['review'].size):
    cleanReviewTest = processedReview(test['review'][i])
    cleanMovieReivewsTest.append(cleanReviewTest)

testVectorizedCounts = cv.transform(cleanMovieReivewsTest)
testVectorizedCounts = testVectorizedCounts.toarray()

result = forest.predict(testVectorizedCounts)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "bowsubmission.csv", index=False, quoting=3 )