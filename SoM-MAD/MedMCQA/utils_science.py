
import os
import pandas as pd

import re

def science_question_selection(subset,i):
    #print(question_with_details)
    science_question = subset.iloc[i]["question"]
    answer_options =  subset.iloc[i]["all_options"]
    correct_label = subset.iloc[i]['correct_option']
    print("science_question:", science_question)
    print("answer_options:", answer_options, "correct_option:", correct_label )
    return science_question, answer_options, correct_label



# Function to compare labels
def compare_labels(row):
    if len(row['AI_label']) == 1:
        return row['AI_label'] == row['correct_label']
    else:
        return None

def science_analysis(df_result):
    df_result['comparison_result'] = df_result.apply(compare_labels, axis=1)
    return df_result
