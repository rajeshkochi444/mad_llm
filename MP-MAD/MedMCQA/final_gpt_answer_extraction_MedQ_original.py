# %%
import pandas as pd
from crewai import Agent, Task, Crew
from crewai.process import Process
from textwrap import dedent
import os

from langchain.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI

# %%
# %%
base_fname = "XXXXX"
file_ext = ".csv"
filename = base_fname + file_ext
df = pd.read_csv(filename)
#df


# %%
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
                                      Extract the final answer label from the given text: {text} which is final output/answer label for a given science question. 
                                      You need to read the text and extract  a single label "a", or "b", "c", "d". You are tasked to output this single label.
                                      Do not output any other details or letters. 
                                      If you cannot extract any single lavel from the given text as answer, output None """
                                      ),
                  expected_output = dedent(f""" \
                                           A single letter "a", or "b", "c", "d".  
                                           Do not add any prefix or suffix or pharases like "The final answer". Just the final letter.
                                           Example output: d
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
    text = df.iloc[i]['AI_label']
    result, output_analysis = Crew4task(llm, text)

    result_bag.append(result)

    print("#############################RESULT#################################################################")
    print(result, '\n\n')
    print(f""" Output_analysis: {output_analysis} \n \n """)

print(result_bag)


# %%
# %%
df['AI_answer'] = result_bag
del df['AI_label']
#df


# %%
# Adding a new column 'Is_Correct' to check if both answers are the same
df['Is_Correct'] = df.apply(lambda x: x['correct_label'] == x['AI_answer'] if x['AI_answer'] is not None else False, axis=1)

print(df)

# %%
# Calculating the metrics
total_true = df['Is_Correct'].sum()
total_false = (~df['Is_Correct']).sum()
total_none = (df['AI_answer'] == 'None').sum()

# Displaying the results
print("Total True: ", total_true)
print("Total False: ", total_false)
print("Total None: ", total_none)


