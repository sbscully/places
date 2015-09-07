from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import spacy.en

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sknn.mlp import Classifier, Layer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

logging.info('loading spaCy')
nlp = spacy.en.English()

def repvecs(words, nlp):
  tokens = [nlp.vocab[word] for word in words]
  return [token.repvec for token in tokens if token.has_repvec]

cities = [line.split(',')[0] for line in open('uk_geo_cities.csv')]
areas = [line.split(',')[0] for line in open('uk_geo_areas.csv')]
gazetteer = [line.split(',')[0] for line in open('uk_geo_gazetteer.csv')]
world = [line.split(',')[0] for line in open('world_geo_cities.csv')]
places = list(set(cities + areas + gazetteer + world))

non_places = [line.split(',')[0] for line in open('uk_nongeo_corncob.csv')]
non_places = list(set(non_places) - set(places))

logging.info('%s place names and %s other words' % (len(places), len(non_places)))

place_vectors = repvecs(places, nlp)
non_place_vectors = repvecs(non_places, nlp)

logging.info('%s place name vectors and %s other word vectors' % (len(place_vectors), len(non_place_vectors)))

place_outputs = [1] * len(place_vectors)
non_place_outputs = [0] * len(non_place_vectors)

x = place_vectors + non_place_vectors
y = place_outputs + non_place_outputs

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=984)

logging.info('training classifier')

clf = Classifier(
  layers=[
    Layer("Rectifier", units=100),
    Layer("Softmax")
  ],
  regularize='dropout',
  dropout_rate=0.5,
  learning_rate=0.02,
  n_iter=20)
clf.fit(np.asarray(x_train), np.asarray(y_train))

print(classification_report(np.asarray(y_test), clf.predict(np.asarray(x_test))))
