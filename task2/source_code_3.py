# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from cmath import sqrt

import codecs

import math

import os

"""
THIS IS USING USER-BASED FILTERING



For calculating the sim between user U1 and user U2 we need (using Pearson)
- the set of items rated by both U1 and U2, 
- the rating of user U1 and U2 for the item i from I that is the set from previous point
- the average rating of each user

For calculating the prediction of user U and item i we need
- user U average rating
- Neighbourhood of the user U
- the similarity between the user U and neighbour N from the Neighbourhood
- the rating of the neighbourhood N of the item i

After all of this we can conclude that we need:
- keep track of which item is rated by whom and it's rating
- keep track of what is the average rating of a user
- keep track of what item a user has rated
- keep track of the similarities between the users
- create a neighbourhood such that the neighbours that are selected have already rated that item



USER_ITEMS_RATING_HM = {user_id: {item_id: user_rating_for_item, item_id2: user_rating_for_item_2}}
SIMILARITIES_TUPLE_HM = {(user_id_1, user_id_2) : similarity_between_user_id_1 and similarity_between_user_id_2 }
USER_AVERAGE_RATING_HM = {user_id: user_average_rating}
ITEMS_USER_HM = {item_id = [user_id_1, user_id_2]} <- to keep track of what were the users that an item was rated for
USERS_NEIGHBOURHOOD_HM = {user_id: [user_id_1, user_id_2]} <- to keep track of the neighbours of the user
"""
USER_ITEMS_RATING_HM = {}
SIMILARITIES_TUPLE_HM = {}
USER_AVERAGE_RATING_HM = {}
ITEMS_USER_HM = {}
USERS_NEIGHBOURHOOD_HM = {}

NEIGHBOURHOOD_MAX_N = 15
MIN_RATING = 0.5;
MAX_RATING = 5.0;

#   0              1              2                 3
# user_id (int), item_id (int), rating (float), timestamp (int)


def formatTrainData(train_data):
    global USER_ITEMS_RATING_HM
    global USER_AVERAGE_RATING_HM
    global ITEMS_USER_HM

    # O(n) formatting the data
    for row in train_data:
        if len(row.strip()) > 0:
            listParts = row.strip().split(',')
            if listParts[0] not in USER_ITEMS_RATING_HM:
                USER_ITEMS_RATING_HM[listParts[0]] = {}
            if listParts[1] not in ITEMS_USER_HM:
                ITEMS_USER_HM[listParts[1]] = []

            USER_ITEMS_RATING_HM[listParts[0]][listParts[1]] = listParts[2]
            ITEMS_USER_HM[listParts[1]].append(listParts[0])

    # O(n) calculating the user average
    for user_id in USER_ITEMS_RATING_HM:
        sum = 0.0
        for item_id in USER_ITEMS_RATING_HM.get(user_id):
            sum += float(USER_ITEMS_RATING_HM.get(user_id).get(item_id))

        items_rated = USER_ITEMS_RATING_HM.get(user_id)
        average_rating = sum / (len(items_rated))
        USER_AVERAGE_RATING_HM[user_id] = average_rating

# for calculating the similarity we are going to user Pearson


def calcSim(user_id_1, user_id_2):
    global USER_ITEMS_RATING_HM
    global ITEMS_USER_HM
    global USER_AVERAGE_RATING_HM

    user_1_items = USER_ITEMS_RATING_HM.get(user_id_1)

    # O(n) filtering
    # get the items that were rated by the user 1, get the users that rated that item and check if user2 is in that array
    items_pool = []
    for item_id in user_1_items:
        users_rted_item_idArr = ITEMS_USER_HM.get(item_id)
        if user_id_2 in users_rted_item_idArr:
            items_pool.append(item_id)

    # calculate sim using Pearson
    user_1_average_rating = float(USER_AVERAGE_RATING_HM.get(user_id_1))
    user_2_average_rating = float(USER_AVERAGE_RATING_HM.get(user_id_2))

    s1 = 0
    s2 = 0
    s3 = 0
    for item_id in items_pool:
        user_1_rating_for_item = float(
            USER_ITEMS_RATING_HM.get(user_id_1).get(item_id))
        user_2_rating_for_item = float(
            USER_ITEMS_RATING_HM.get(user_id_2).get(item_id))
        d1 = user_1_rating_for_item - user_1_average_rating
        d2 = user_2_rating_for_item - user_2_average_rating

        s1 += d1 * d2
        s2 += pow(d1, 2)
        s3 += pow(d2, 2)

    if(s2 == 0 or s3 == 0):
        return 0.0

    SIM = s1 / (math.sqrt(s2) * math.sqrt(s3))
    return SIM

# for determining the neighbourhood of a user based on the similarities. I am just using the ones that the user is the most close to.
# user_id <- the user we are determining the neighbourhood for
# similarities_tuples <- the similarities that he has with the other users in an array of  tuples, something like (other_user_id, similarity_between_user_id_and_other_user_id)
def detNeigh(user_id, similarities_tuples):
    global USERS_NEIGHBOURHOOD_HM
    global NEIGHBOURHOOD_MAX_N

    similarities_tuples.sort(key=lambda tup: tup[1], reverse=True)
    neighbours = []

    i = 0
    #using KNN 
    while i != NEIGHBOURHOOD_MAX_N:
        if i >= len(similarities_tuples):
            break
        neighbours.append(similarities_tuples[i][0])
        i += 1

    USERS_NEIGHBOURHOOD_HM[user_id] = neighbours


#   0              1              2
# user_id (int), item_id (int), timestamp (int)
def doPred(data):
    global ITEMS_USER_HM
    global SIMILARITIES_TUPLE_HM
    global USER_AVERAGE_RATING_HM
    global USERS_NEIGHBOURHOOD_HM
    global USER_ITEMS_RATING_HM
    global MIN_RATING
    global MAX_RATING
    f = open('./results_user_based.csv', "a")

    for row in data:
        if len(row.strip()) > 0:
            listParts = row.strip().split(',')
            user_id_to_pred = listParts[0]
            item_id_to_pred = listParts[1]
            # first, get the users that rated that item
            users_that_rated_the_item = ITEMS_USER_HM.get(item_id_to_pred)
            sim_user_users = []  # then, get the similarity between the user we are trying to determine and those who already rated the item
            PRED = MIN_RATING;

            if users_that_rated_the_item != None: #this would happen if the item has not been in the training data.
                for user_id in users_that_rated_the_item:
                    sim_user_users.append((user_id, SIMILARITIES_TUPLE_HM.get(user_id_to_pred, user_id)))

                # determine the neighbours
                detNeigh(user_id_to_pred, sim_user_users)

                user_to_pred_average_rating = float(
                    USER_AVERAGE_RATING_HM.get(user_id_to_pred))
                neighbourhood_user_to_pred = USERS_NEIGHBOURHOOD_HM.get(
                    user_id_to_pred)

                s1 = 0
                s2 = 0

                for neighbour_id in neighbourhood_user_to_pred:
                    sim_between_users = float(
                        SIMILARITIES_TUPLE_HM.get(user_id_to_pred, neighbour_id))
                    rating_of_neighbour_to_item = float(
                        USER_ITEMS_RATING_HM.get(neighbour_id).get(item_id_to_pred))
                    average_rating_neighbour = float(
                        USER_AVERAGE_RATING_HM.get(neighbour_id))

                    s1 += sim_between_users * \
                        (rating_of_neighbour_to_item - average_rating_neighbour)
                    s2 += sim_between_users

                    PRED = user_to_pred_average_rating + (s1/s2);

                    if PRED < MIN_RATING:
                        PRED = MIN_RATING;
                    elif PRED > MAX_RATING:
                        PRED = MAX_RATING;

            print("PREDICTION: USER ", user_id_to_pred,
                  " WILL RATE ITEM ", item_id_to_pred, " AS ", PRED)     
            PRED_str = str(PRED);
            s = str(user_id_to_pred) + ',' + str(item_id_to_pred) + ',' + PRED_str + ',' + listParts[2] + '\n';
            f.write(s)
    f.close()



def doSim():
    global USER_ITEMS_RATING_HM
    global SIMILARITIES_TUPLE_HM

    # O(n^2) time for calculating the sim and also determining the neighbourhood
    for user_id_1 in USER_ITEMS_RATING_HM:
        for user_id_2 in USER_ITEMS_RATING_HM:
            # if we have already calculated the similarity, exclude the tuple
            if (user_id_1, user_id_2) in SIMILARITIES_TUPLE_HM or (user_id_2, user_id_1) in SIMILARITIES_TUPLE_HM or user_id_1 == user_id_2:
                continue

            sim = calcSim(user_id_1, user_id_2)
            SIMILARITIES_TUPLE_HM[(user_id_1, user_id_2)] = sim
            SIMILARITIES_TUPLE_HM[(user_id_2, user_id_1)] = sim


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

    formatTrainData(listLines_train);
    doSim();
    doPred(listLines_test);
