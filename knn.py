import csv
import random

import numpy as np


def get_knn_sorted(data, k, row):
    dists = [np.linalg.norm(np.array(data_row[1:]) - np.array(row[1:])) for data_row in data]
    dists = list(enumerate(dists))
    dists.sort(key=lambda x: x[1])
    dists = dists[:k]
    perm = list(map(lambda x: x[0], dists))
    return list(np.array(data)[perm])


def most_common(lst):
    items = list(set(lst))
    counts = [lst.count(item) for item in items]
    counts = list(enumerate(counts))
    counts.sort(key=lambda x: x[1], reverse=True)
    counts = list(filter(lambda x: x[1] == counts[0][1], counts))
    return items[random.choice(counts)[0]]


def predict_label(data, k, row):
    knn_sorted = get_knn_sorted(data, k, row)
    labels = list(map(lambda x: x[0], knn_sorted))
    return [most_common(labels[:k1]) for k1 in range(1, k + 1)]


def loo(data, k):
    cnt = [0] * k
    for i, row in enumerate(data):
        predicts = [x != row[0] for x in predict_label(data[:i] + data[i + 1:], k, row)]
        cnt = [cnt[i] + predicts[i] for i in range(k)]
    return [x / len(data) for x in cnt]


def run_and_save(dataset_path, answer_path):
    reader = csv.reader(open(dataset_path))
    reader.__next__()  # skip the header
    data = []
    for row in reader:
        data.append([row[0]] + list(map(float, row[1:])))
    loo_errors = loo(data, 10)
    with open(answer_path, 'w') as f:
        f.write('k\tx\n')
        for k, x in enumerate(loo_errors):
            f.write('{}\t{}\n'.format(k + 1, x))


if __name__ == '__main__':
    run_and_save('datasets/cancer.csv', 'answers/cancer.txt')
    run_and_save('datasets/spam.csv', 'answers/spam.txt')
