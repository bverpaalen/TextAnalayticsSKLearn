from sklearn.datasets import fetch_20newsgroups



def main():
    print("Starting data sklearn data text analaytics.")
    set = retrieveData(1)
    print(len(set.data))

def retrieveData(selector):
    if selector == 1:
        categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
        twenty_Train = fetch_20newsgroups(subset="train",categories=categories,shuffle=True, random_state=42)
        return twenty_Train
    return None

main()