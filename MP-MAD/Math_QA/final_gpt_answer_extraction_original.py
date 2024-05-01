import pandas as pd
from crewai import Agent, Task, Crew
from crewai.process import Process
from textwrap import dedent
import os

from langchain.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI

# %%
base_fname = "XXXXX"
file_ext = ".csv"
filename = base_fname + file_ext
df = pd.read_csv(filename)
df

# %%
os.environ["OPENAI_API_KEY"] = "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"
#os.environ['hUGGINGFACEHUB_API_TOKEN'] = "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"
hugg_key = "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"
#lm = HuggingFaceHub(repo_id='google/gemma-2b-it', huggingfacehub_api_token = hugg_key, model_kwargs={"max_new_tokens": 1000})

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1)
#llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8)

# %%
def agent_analyst(llm):
    return Agent(
        role = dedent(f"""\
                    You are an expert AI agent who can extract answers from a given text"""
                    ),
        goal = dedent(f"""\
                    Extract the single numerical value from the given text."""
                    ),
        backstory = dedent("""\
                        You are an expert AI agent who can easily solve the given task"""
                        ),
        llm = llm,
        verbose=True,
        allow_delegation=False,
        #tools=[math_tool],
        memory = True
        )


# %%
def task_analysis(text,agent):
    return Task(
                  description= dedent(f"""\
                                      Extract the final answer from the given text: {text} which is final output/answer for a given math question. 
                                      You need to read the text and extract  a single numerical answer. You are tasked to output this numerical value.
                                      Do not output any other details or letters. 
                                      If there are no numerical value in the given final text, output None """
                                      ),
                  expected_output = dedent(f""" \
                                           A single numeric value.  Do not add any prefix or suffix or pharases like "The final answer". Just the final numeric value.
                                           If your questions ask for interger, report integer values and for floating values, use one or two decimal places.
                                           Fractions must be reported as a/b format.
                                           Example output: 32.8
                                           """
                                           ),
                  agent=agent,
                  output_file = "output_task.txt"
                  )


# %%
def Crew4task(llm, text):
  analyst = agent_analyst(llm)
  analysis_task = task_analysis(text, analyst)
  
  crew = Crew(
    agents= [analyst],
    tasks=[analysis_task],
    process=Process.sequential,
    #full_output = True,
    verbose=2, # You can set it to True, 1 or 2 to different logging levels
    output_log_file=True
  )

  # Get your crew to work!
  result = crew.kickoff()
  output_analysis =  analysis_task.output.raw_output

  return result, output_analysis

  


# %%
result_bag = []
for i in range(len(df)):
    text = df.iloc[i]['AI_answer']
    result, output_analysis = Crew4task(llm, text)

    result_bag.append(result)

    print("#############################RESULT#################################################################")
    print(result, '\n\n')
    print(f""" Output_analysis: {output_analysis} \n \n """)

print(result_bag)



# %%
df['AI_answer_analyzed'] = result_bag

df


# %%
print(df["AI_answer_analyzed"].isna().sum())

# %%
# Assuming 'df' is your DataFrame and 'AI_answer_analyzed' is the column of interest
none_string_count = (df['AI_answer_analyzed'] == 'None').sum()

print("Total 'None' string values in 'AI_answer_analyzed':", none_string_count)


# %%
'''
import pandas as pd

# Sample DataFrame initialization (replace this with your actual DataFrame)
# df = ...

# Step 1: Convert string numerical values to floats
# Use the `pd.to_numeric` function to convert strings to numbers, errors='coerce' will replace non-convertible strings with NaN
df['AI_answer_analyzed'] = pd.to_numeric(df['AI_answer_analyzed'], errors='coerce')
df['correct_answer'] = pd.to_numeric(df['correct_answer'], errors='coerce')

# Rounding to two decimal places if conversion is successful
df['AI_answer_analyzed'] = df['AI_answer_analyzed'].round(2)
df['correct_answer'] = df['correct_answer'].round(2)

# Step 2: Compare the float values within a threshold of 0.02
# Create a comparison function
def compare_within_threshold(row):
    if pd.isna(row['AI_answer_analyzed']) or pd.isna(row['correct_answer']):
        return False  # If either value is NaN, return False
    return abs(row['AI_answer_analyzed'] - row['correct_answer']) <= 0.02

# Step 3: Create a new column with the result of the comparison
df['comparison_result'] = df.apply(compare_within_threshold, axis=1)

# Step 4: Count the number of True and False values
true_count = df['comparison_result'].sum()  # True is interpreted as 1
false_count = len(df) - true_count

# Output the counts
print("Total True values:", true_count)
print("Total False values:", false_count)

# If you need to see the DataFrame
df
'''

# %%
import pandas as pd
from fractions import Fraction

# Define a function to convert strings to floats, including fractions like '1/2'
def string_to_float(value):
    try:
        # Attempt to convert regular numbers to float first
        return float(value)
    except ValueError:
        # If ValueError is raised, try converting fractions
        try:
            return float(Fraction(value))
        except ValueError:
            # If it still fails, return NaN
            return None

# Assuming 'df' is your DataFrame with 'AI_answer_analyzed' and 'correct_answer' columns

# Apply the conversion function to the DataFrame columns
df['AI_answer_analyzed'] = df['AI_answer_analyzed'].apply(string_to_float)
df['correct_answer'] = df['correct_answer'].apply(string_to_float)

# Perform rounding to two decimal places
df['AI_answer_analyzed'] = df['AI_answer_analyzed'].round(2)
df['correct_answer'] = df['correct_answer'].round(2)

# Perform the comparison within a threshold of 0.02
df['comparison_result'] = df.apply(lambda row: abs(row['AI_answer_analyzed'] - row['correct_answer']) <= 0.02 if pd.notnull(row['AI_answer_analyzed']) and pd.notnull(row['correct_answer']) else False, axis=1)

# Count the True and False values in the new 'comparison_result' column
true_count = df['comparison_result'].sum()
false_count = len(df) - true_count

# Output the counts
print(f"Total True values: {true_count}")
print(f"Total False values: {false_count}")

# If you want to see the DataFrame with the new 'comparison_result' column
df



# %%
fname_analysis = base_fname + "_gpt_analysis.csv"
df.to_csv(fname_analysis)


