import ndjson
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

model = spacy.load('en_core_web_sm')


def assemble():
    print("Prepare data set")
    x, y = prepare_data()

    print("Split into train and tast data set")
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print("Create CountVectorizer")

    vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
    vect.fit(x)

    print("Transform to bag of words")
    x_train = vect.transform(x_train)

    print("Cross validation")
    scores = cross_val_score(LogisticRegression(max_iter=300), x_train, y_train, cv=2)
    print("Średnia dokładność walidacji krzyżowej: {:.2f}".format(np.mean(scores)))


def prepare_data():
    with open('data/Software.json') as f:
        data = ndjson.load(f)

    reviews_df = pd.DataFrame(data)
    reviews_df = reviews_df.dropna(subset=['overall', 'reviewText'])

    x = reviews_df.loc[:1000, 'reviewText'].values

    y = reviews_df.loc[:1000, 'overall'].values
    y = map(int, y)
    y = list(y)

    return x, y


def custom_tokenizer(doc):
    tokens = model(doc)
    return [token.lemma_ for token in tokens]


if __name__ == '__main__':
    assemble()
