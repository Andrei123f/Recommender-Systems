# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from cmath import sqrt

import codecs

import math

import os


# GLOBAL CONSTANTS
LOWER_BOUND = 0.5
HIGHER_BOUND = 5
NEGHBOURHOOD_MAX_NUMBER = 5
"""
THIS IS USING ITEM-BASED FILTERING

For calculating the sim between items I1 and I2 we need (using adjusted cosine similarity):
- the users that rated both item I1 and I2
- each user average rating
- the rating of user for each item

For calculating the pred of user U of item I we need:
- the neighbourhood of that item
- the similarity of the neighbours of that item
- the rating of that user of the neighbour

After all this we can conclude that we need:
- to keep track of user's average rating
- to keep track of each item rating by each user
- to keep track of each item neighbourhood
- to keep track of the sims of the items



ITEMS_USERS_RATING_HM = {item_id: {user_id: rating_of_user}}
SIMILARITIES_TUPLE_HM = {(item_id_1, item_id_2): sim_between_item_1_and_item_2}
ITEMS_NEIGHBOURHOOD_HM = {item_id: [item_id_1, item_id_2]} <- to keep track of the neighbours of the item
USER_AVERAGE_RATING_HM = {user_id: user_average}
USER_ITEMS_RATING_HM = {user_id: {item_id1: user_rating }} <- to keep track of the items that were rated by each user


"""
ITEMS_USERS_RATING_HM = {}
SIMILARITIES_TUPLE_HM = {}
ITEMS_NEIGHBOURHOOD_HM = {}
USER_AVERAGE_RATING_HM = {}
USER_ITEMS_RATING_HM = {}

NEIGHBOURHOOD_MAX_N = 5
MIN_RATING = 0.5
MAX_RATING = 5.0

#   0              1              2                 3
# user_id (int), item_id (int), rating (float), timestamp (int)


def formatTrainData(train_data):
    global ITEMS_USERS_RATING_HM
    global USER_AVERAGE_RATING_HM
    global USER_ITEMS_RATING_HM

    # O(n) formatting the data
    for row in train_data:
        if len(row.strip()) > 0:
            listParts = row.strip().split(',')
            if listParts[1] not in ITEMS_USERS_RATING_HM:
                ITEMS_USERS_RATING_HM[listParts[1]] = {}
            if listParts[0] not in USER_ITEMS_RATING_HM:
                USER_ITEMS_RATING_HM[listParts[0]] = {}

            ITEMS_USERS_RATING_HM[listParts[1]][listParts[0]] = listParts[2]
            USER_ITEMS_RATING_HM[listParts[0]][listParts[1]] = listParts[2]

    # O(n) calculating the user average
    for user_id in USER_ITEMS_RATING_HM:
        sum = 0.0
        for item_id in USER_ITEMS_RATING_HM.get(user_id):
            sum += float(USER_ITEMS_RATING_HM.get(user_id).get(item_id))

        items_rated = USER_ITEMS_RATING_HM.get(user_id)
        average_rating = sum / (len(items_rated))
        USER_AVERAGE_RATING_HM[user_id] = average_rating


# for calculating the similarity we are going to use adjusted cosine similarity
def calcSim(item_id_1, item_id_2):
    global ITEMS_USERS_RATING_HM
    global USER_ITEMS_RATING_HM
    global USER_AVERAGE_RATING_HM

    users_who_rated_item_1 = ITEMS_USERS_RATING_HM.get(item_id_1)
    users_who_rated_item_2 = ITEMS_USERS_RATING_HM.get(item_id_2)

    users_who_rated_both_items = []

    # O(n) filtering
    for user_id_who_rated_item_1 in users_who_rated_item_1:
        if user_id_who_rated_item_1 in users_who_rated_item_2:
            users_who_rated_both_items.append(user_id_who_rated_item_1)

    SIM = 0.0

    if len(users_who_rated_both_items) > 0:
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0

        for user_id in users_who_rated_both_items:
            user_rating_of_item_1 = float(
                USER_ITEMS_RATING_HM.get(user_id).get(item_id_1))
            user_rating_of_item_2 = float(
                USER_ITEMS_RATING_HM.get(user_id).get(item_id_2))
            user_average_rating = float(USER_AVERAGE_RATING_HM.get(user_id))

            s1 += (user_rating_of_item_1 - user_average_rating) * \
                (user_rating_of_item_2 - user_average_rating)
            s2 += pow((user_rating_of_item_1 - user_average_rating), 2)
            s3 += pow((user_rating_of_item_2 - user_average_rating), 2)

        if s2 != 0 and s3 != 0:
            SIM = s1 / (sqrt(s2) * sqrt(s3))

    return SIM

# for determining the neighbourhood of a user based on the similarities. I am just using the ones that the user is the most close to.
# user_id <- the user we are determining the neighbourhood for


def detNeigh(item_id, items_rated_by_same_user):
    global ITEMS_NEIGHBOURHOOD_HM
    global NEIGHBOURHOOD_MAX_N
    global SIMILARITIES_TUPLE_HM
    similarities_tuples = []

    for item_id_other in items_rated_by_same_user:
        similarities_tuples.append((
            item_id_other, SIMILARITIES_TUPLE_HM.get(item_id, item_id_other)))

    similarities_tuples.sort(key=lambda tup: tup[1], reverse=True)
    neighbours = []

    i = 0
    # using KNN
    while i != NEIGHBOURHOOD_MAX_N:
        if i >= len(similarities_tuples):
            break
        neighbours.append(similarities_tuples[i][0])
        i += 1

    ITEMS_NEIGHBOURHOOD_HM[item_id] = neighbours


#   0              1              2
# user_id (int), item_id (int), timestamp (int)
def doPred(data):
    global ITEMS_USERS_RATING_HM
    global SIMILARITIES_TUPLE_HM
    global USER_AVERAGE_RATING_HM
    global ITEMS_NEIGHBOURHOOD_HM
    global USER_ITEMS_RATING_HM
    global MIN_RATING
    global MAX_RATING
    f = open('./results_item_based.csv', "a")

    for row in data:
        if len(row.strip()) > 0:
            listParts = row.strip().split(',')
            user_id_to_pred = listParts[0]
            item_id_to_pred = listParts[1]

            PRED = MIN_RATING
            # check if we had that item during the training session. if we did not, then the PRED is the minimum one.
            if ITEMS_USERS_RATING_HM.get(item_id_to_pred) != None:
                # determine the neighbourhood of the item
                # get the items that that user has rated
                items_rated_by_user_to_pred = []
                for item_id in USER_ITEMS_RATING_HM.get(user_id_to_pred):
                    items_rated_by_user_to_pred.append(item_id)

                detNeigh(item_id_to_pred, items_rated_by_user_to_pred)
                neighbours = ITEMS_NEIGHBOURHOOD_HM.get(item_id_to_pred)

                s1 = 0.0
                s2 = 0.0

                for item_neighbour_id in neighbours:
                    sim_between_neighbour_and_item_to_pred = float(
                        SIMILARITIES_TUPLE_HM.get(item_neighbour_id, item_id_to_pred))
                    rating_of_neighbour_by_user_to_pred = float(
                        USER_ITEMS_RATING_HM.get(user_id_to_pred).get(item_neighbour_id))

                    s1 += sim_between_neighbour_and_item_to_pred * \
                        rating_of_neighbour_by_user_to_pred
                    s2 += sim_between_neighbour_and_item_to_pred

                if s2 != 0:
                    PRED = s1 / s2

                if PRED < MIN_RATING:
                    PRED = MIN_RATING
                elif PRED > MAX_RATING:
                    PRED = MAX_RATING

            PRED_str = str(PRED)
            print("PREDICTION: USER ", user_id_to_pred,
                  " WILL RATE ITEM ", item_id_to_pred, " AS ", PRED)
            s = str(user_id_to_pred) + ',' + str(item_id_to_pred) + \
                ',' + PRED_str + ',' + listParts[2] + '\n'
            f.write(s)
    f.close()


def doSim():
    global ITEMS_USERS_RATING_HM
    global SIMILARITIES_TUPLE_HM

    # O(n^2) time for calculating the sim

    for item_id_1 in ITEMS_USERS_RATING_HM:
        for item_id_2 in ITEMS_USERS_RATING_HM:
            # if we have already calculated the similarity, exclude the tuple
            if (item_id_1, item_id_2) in SIMILARITIES_TUPLE_HM or (item_id_2, item_id_1) in SIMILARITIES_TUPLE_HM or item_id_1 == item_id_2:
                continue;
            sim = calcSim(item_id_1, item_id_2)
            SIMILARITIES_TUPLE_HM[(item_id_1, item_id_2)] = sim
            SIMILARITIES_TUPLE_HM[(item_id_2, item_id_1)] = sim


if __name__ == '__main__':
    test_csv_values_name = 'comp3208_100k_test_withoutratings.csv'
    train_csv_values_name = 'comp3208_100k_train_withratings.csv'
    file_dir = os.path.dirname(__file__)

    fileHandler_test = codecs.open(
        file_dir + os.sep + test_csv_values_name, 'r', 'utf-8', errors='replace')
    fileHandler_train = codecs.open(
        file_dir + os.sep + train_csv_values_name, 'r', 'utf-8', errors='replace')

    listLines_test = fileHandler_test.readlines()
    listLines_train = fileHandler_train.readlines()

    fileHandler_test.close()
    fileHandler_train.close()

    formatTrainData(listLines_train)
    doSim()
    doPred(listLines_test)
