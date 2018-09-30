from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

TEST = True

def main():
    print("Starting data sklearn data text analaytics.")
    print()
    set = RetrieveData(1)

    if TEST:
        TestingSet(set)
    WordVector(set.data)

def RetrieveData(selector):
    if selector == 1:
        categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
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

main()
