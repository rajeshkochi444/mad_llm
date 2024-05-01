
import os
import pandas as pd
from fractions import Fraction
import re

def math_question_selection(subset,i):
    question_with_details = subset.iloc[i]
    #print(question_with_details)
    math_question = subset.iloc[i]["Problem"]
    answer_options =  subset.iloc[i]["options"]
    correct_answer = subset.iloc[i]['correct_numeric']
    print("math_question:", math_question)
    print("answer_options:", answer_options, "correct_answer:", correct_answer )
    return math_question, answer_options, correct_answer

# This pattern looks for an equals sign followed by spaces, and captures:
# - Negative or positive fractions (e.g., -1/2, 1/2)
# - Negative or positive floating point numbers (e.g., -3.14), ignoring commas for thousand separators
# - Negative or positive integers (e.g., -42), ignoring commas for thousand separators
# It will capture the last numeric value after the equals sign, if present,
# or the first numeric value otherwise.

# Function to extract the final numeric value after '=' if present, taking into account commas
def extract_final_value(text):
    # Find all matches in the text
    pattern = r'(?:= ?)?(-?\d+/\d+|-?\d+(?:,\d+)*\.\d+|-?\d+(?:,\d+)*)'
    matches = re.findall(pattern, text)
    # Check if an equals sign is present and return the last numeric value
    for match in matches[::-1]:  # Start from the end of the list
        match = match.replace(',', '')  # Remove commas from the match
        if '/' in match or '.' in match or re.match(r'-?\d+', match):  # Check if it's a fraction, float, or integer
            return match  # Return the match
    return None  # If no match, return None


def convert_to_float(value):
    try:
        if '/' in value:
            return float(Fraction(value))
        else:
            return float(value)
    except ValueError:
        return None  # or some default value like 0.0

def math_analysis(result_round):
    # Apply the function to the 'AI_answer' column
    df_result = pd.DataFrame(result_round, columns = ['AI_answer', 'correct_answer']) 

    try:
        df_result['Extracted_Value'] = df_result['AI_answer'].apply(extract_final_value)
        df_result['correct_numeric'] = df_result['correct_answer'].apply(convert_to_float).round(2)
        df_result['agent_numeric'] = df_result['Extracted_Value'].apply(convert_to_float).round(2)
        df_result['difference'] = df_result['agent_numeric'] - df_result['correct_numeric'] 

        # Create a boolean column based on the threshold
        threshold = 0.05
        df_result['bool'] = df_result['difference'].abs() <= threshold  

        #bool_counts = df_result['bool'].value_counts()
    except:
        f'''Error in math analysis'''
    bool_counts = df_result['bool'].value_counts()

   
    return df_result, bool_counts

