# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, logging, os, shutil, subprocess, sqlite3, traceback, random

if __name__ == '__main__':
    #the names of the files that we are going to use in order to calculate MSE, RMSE , MAE 
    predicted_csv_values_name = 'comp3208_micro_pred.csv';
    actual_csv_values_name = 'comp3208_micro_gold.csv';
    file_dir = os.path.dirname(__file__);

    fileHandler_pred = codecs.open( file_dir + '/' + predicted_csv_values_name, 'r', 'utf-8', errors = 'replace' );
    fileHandler_act = codecs.open( file_dir + '/' + actual_csv_values_name, 'r', 'utf-8', errors = 'replace' );

    listLines_pred = fileHandler_pred.readlines();
    listLines_act = fileHandler_act.readlines();

    fileHandler_pred.close();
    fileHandler_act.close();

    n = 0;
    sum = 0;
    sum_sq = 0;
    j = 0;
    for strLine_pred in listLines_pred:
        if len(strLine_pred.strip()) > 0:
            strLine_act = listLines_act[n];            
            if len(strLine_act.strip()) > 0:
                listParts_act = strLine_act.strip().split(',');
                listParts_pred = strLine_pred.strip().split(',');
                #check if we are referring to the same purchase
                if(listParts_act[0] == listParts_pred[0] and listParts_act[1] == listParts_pred[1]):
                    # calculate the sum for the MAE, RMSE and MSE formula
                    sum += abs(float(listParts_pred[2]) - float(listParts_act[2]));
                    sum_sq += pow(float(listParts_pred[2]) - float(listParts_act[2]), 2);
                    j += 1;

            
            listParts = strLine_pred.strip().split(',');
            n += 1;
    
    MAE = 1/n * sum;
    MSE = 1/n * sum_sq;
    RMSE = math.sqrt(1/n * sum_sq); # or math.sqrt(MSE)


    with open("task1_submit.csv", "w") as file:
        file.write(str(MSE) + "," + str(RMSE) + "," + str(MAE));
