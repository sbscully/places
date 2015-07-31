from __future__ import print_function

import re
import random
import string
import os.path
import logging
import matplotlib.pyplot as plt
import multiprocessing

from mpl_toolkits.mplot3d import Axes3D
from HTMLParser import HTMLParser
from textblob import TextBlob
from gensim.models import Word2Vec, Phrases
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_cores = multiprocessing.cpu_count()

class FullDescriptions(object):
    def __init__(self, filename, blobs=False):
        self.filename = filename
        self.blobs = blobs

    def __iter__(self):
        for line in open(self.filename, 'r'):
            if self.blobs:
                for sentence in TextBlob(line.decode('utf8')).sentences:
                    yield [word.lower() for word in sentence.words]
            else:
                yield line.split()

model_fname = 'uk_full_descriptions_1M'

# Generate word vectors and save model
if os.path.isfile(model_fname + '.clean.vectors'):
    model = Word2Vec.load(model_fname + '.clean.vectors')
else:
    # Generate bigrams and save cleaned txt
    bigram = Phrases(FullDescriptions(model_fname + '.txt', True))

    with open(model_fname + '.clean.txt', 'wb') as output:
        logging.info('cleaning text and writing to file')
        sentence_index = 0

        for line in open(model_fname + '.txt'):
            for sentence in TextBlob(line.decode('utf8')).sentences:
                if sentence_index % 10000 == 0:
                    logging.info('PROGRESS: at sentence #' + str(sentence_index))
                sentence_index += 1

                words = [ word.lower() for word in sentence.words ]
                words = ' '.join(bigram[words]).encode('utf8')
                print(words, file=output)

    model = Word2Vec(FullDescriptions(model_fname + '.clean.txt'), workers=num_cores)
    model.save(model_fname + '.clean.vectors')

words = []
vectors = []
targets = []
for fname in ['uk_nongeo_corncob.csv', 'uk_geo_words_bigrams.csv', 'uk_geo_cities.csv', 'uk_geo_areas.csv']:
    for line in open(fname):
        word, isgeo = line.rstrip().split(',')
        try:
            vector = model[word]
            words.append(word)
            vectors.append(vector)
            targets.append(int(isgeo))
        except:
            pass

# pca = PCA(n_components=3)
# components = pca.fit_transform(vectors)
# results = random.sample(zip(components, targets, words), 200)

# geo = [[result[0], result[2]] for result in results if result[1]]
# nongeo = [[result[0], result[2]] for result in results if not result[1]]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim3d(-1.5, 1.5)
# ax.set_ylim3d(-1.5, 1.5)
# ax.set_zlim3d(-1.5, 1.5)

# for coords, word in geo:
#     ax.text(coords[0], coords[1], coords[2], word.decode('utf8'), color='green')

# for coords, word in nongeo:
#     ax.text(coords[0], coords[1], coords[2], word.decode('utf8'), color='red')

# plt.show()

# geo = [arr[0] for arr in geo]
# nongeo = [arr[0] for arr in nongeo]

# geo = map(list, zip(*geo))
# nongeo = map(list, zip(*nongeo))

# # 3D PCA plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(geo[0], geo[1], geo[2], marker='o', c='g')
# ax.scatter(nongeo[0], nongeo[1], nongeo[2], marker='o', c='r')
# plt.show()

# # 2D PCA plot
# plt.clf()
# plt.plot(geo[0], geo[1], 'go', markersize=1, mec='g')
# plt.plot(nongeo[0], nongeo[1], 'ro', markersize=1, mec='r')
# plt.title('Place Names (green) vs Non-Place Names (red)')
# plt.show()

words_train, words_test, targets_train, targets_test = train_test_split(
    vectors, targets, test_size=0.3, random_state=21)

print("Number of place names found: " + str(sum(targets_train)) + " (train) " + str(sum(targets_test)) + " (test)")

classifier = RandomForestRegressor(n_estimators=100, n_jobs=num_cores)
classifier.fit(words_train, targets_train)

predictions_test = classifier.predict(words_test)

print(classification_report(targets_test, [round(prediction) for prediction in predictions_test]))

test_words = []
for line in open('uk_full_descriptions_10k.clean.txt'):
    test_words += line.split(' ')
test_words = set(test_words)
test_vectors = []
new_test_words = []
for word in test_words:
    try:
        vector = model[word]
        new_test_words.append(word)
        test_vectors.append(vector)
    except:
        pass
test_words = new_test_words
test_predictions = classifier.predict(test_vectors)
results = sorted(zip(test_words, test_predictions), key=lambda pair: pair[1])

threshold = 0.8
place_dict = {}
for word, prediction in results:
    if float(prediction) >= threshold:
        place_dict[word] = 1

for fname in ['uk_nongeo_corncob.csv', 'uk_geo_words_bigrams.csv', 'uk_geo_cities.csv', 'uk_geo_areas.csv']:
    for line in open(fname):
        word, isgeo = line.rstrip().split(',')
        place_dict[word] = 0

new_places = [place for place in place_dict.keys() if place_dict[place] != 0]

with open('classifier_new_places.csv', 'wb') as output:
    output.write("\n".join(new_places))

with open('classifier_results.csv', 'wb') as output:
    for word, prediction in results:
        print(("%.2f " % prediction) + word, file=output)

precision, recall, _ = precision_recall_curve(targets_test, predictions_test)

plt.clf()
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision_score(targets_test, predictions_test)))
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()