from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer as TTransformer
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import metrics
import numpy as np

TEST = False
#categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
categories = None

classifiers = [
    MNB(),
    DTC(),
    SGDC()]
fw = [
     ('vect',CountVectorizer()), 
      #("tf",TTransformer(use_idf=False)),
      ("tfidf",TTransformer())]

def main():
    print("Starting data sklearn data text analaytics.")
    print()
    set = RetrieveData(1)

    if TEST:
        TestingSet(set)

    for k in range(len(classifiers)):
       clas = classifiers[k] 
       print()
       print("clf: "+str(clas))
       for i in range(2,3):
          print(i)
          pip = fw[:i]
          pip.append(('clf',clas))
          print("Pipeline: "+str(pip))
          clf = Pipeline(pip)
          clf.fit(set.data,set.target)
          evaluation(clf)

def RetrieveData(selector):
    if selector == 1:
        twenty_Train = fetch_20newsgroups(subset="train",categories=categories,shuffle=True, random_state=42)
        return twenty_Train
    return None

def TestingSet(set):
    data = set.data
    print("TESTING DATA")
    print("Data size: "+str(len(data)))
    print()
    print("Targets: "+str(set.target_names))
    print()
    print("First lines first file:")
    print("\n".join(data[0].split("\n")[:3]))
    print()

def WordVector(data):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    print(X_train_counts.shape)
    print(count_vect.vocabulary_.get(u'algorithm'))
    return count_vect

def OcToFr(data,CV,target,names):
    tf_transformer = TTransformer(use_idf=False).fit(data)
    X_train_tf = tf_transformer.transform(data)
    print(X_train_tf.shape)
    ttransformer = TTransformer()
    X_train_tfidf = ttransformer.fit_transform(data)

    Xtrain = X_train_tfidf    
    clf = MNB().fit(Xtrain,target)
    docs_new = ['God is love','OpenGL in the GPU is fast']
    X_new_counts = CV.transform(docs_new)
    X_new_tfidf = ttransformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, names[category]))

def evaluation(text_clf):
     twenty_test = fetch_20newsgroups(subset='test',
        categories=categories, shuffle=True, random_state=42)
     docs_test = twenty_test.data
     predicted = text_clf.predict(docs_test)
     print(np.mean(predicted == twenty_test.target))            
     print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))

main()
