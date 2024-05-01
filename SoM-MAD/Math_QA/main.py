import os
import pandas as pd
from crew import Crew4task
from dotenv import load_dotenv
#from langchain.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM, HuggingFaceHub, HuggingFaceEndpoint
from utils_math import math_analysis, math_question_selection

n_examples = 100
n_rounds = 3

df = pd.read_csv('cleaned_math_dataset.csv')
subset = df.iloc[:n_examples]

# LLM setup 
os.environ["OPENAI_API_KEY"] = "Your Key"
#os.environ['hUGGINGFACEHUB_API_TOKEN'] = "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"
hugg_key = "Your KEY"

llm = ChatOpenAI(model='gpt-3.5-turbo')


#model = "mistralai/Mistral-7B-Instruct-v0.2"

'''
llm = VLLM(
    model= model,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=1024,
    download_dir = "/home/mila/r/rajesh.raju/scratch/.cache/huggingface",
    top_k=10,
    top_p=0.95,
    temperature=0.7,
    dtype='float16',
    #gpu_memory_utilization=0.8,
)
llm = HuggingFaceEndpoint(
    repo_id=model,
    #endpoint_url="http://localhost:8010/",
    max_new_tokens=1024,
    top_k=10,
    top_p=0.95,
    #typical_p=0.95,
    temperature=0.7,
    #repetition_penalty=1.03,
    huggingfacehub_api_token="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"
)
'''

result_round0, result_round1, result_round2, result_round3 = [[] for _ in range(n_rounds+1)]
for i in range(n_examples):
   print(i)
   math_question, answer_options, correct_answer = math_question_selection(subset, i)

   result, output_round0, output_round1, output_round2, output_round3 = Crew4task(llm, math_question, n_rounds)
   
   result_round0.append([output_round0, correct_answer])
   result_round1.append([output_round1, correct_answer])
   result_round2.append([output_round2, correct_answer])
   result_round3.append([output_round3, correct_answer])
   
   print("#############################RESULT#################################################################")
   print(result, '\n\n\n')

   print(f""" Output_Round0: {output_round0} \n \n """)
   print(f""" Output_Round1: {output_round1} \n \n """)
   print(f""" Output_Round2: {output_round2} \n \n """)
   print(f""" Output_Round3: {output_round3} \n \n """)
   

print(f""" Final Result: {result} \n \n""")


### Analysis
result_bag = {}
result_bag['result_round0'] = result_round0
result_bag['result_round1'] = result_round1
result_bag['result_round2'] = result_round2
result_bag['result_round3'] = result_round3

for i, round in enumerate([result_round0, result_round1, result_round2, result_round3]):
    df_before_analysis = pd.DataFrame(round, columns = ['AI_answer', 'correct_answer'])
    df_before_analysis.to_csv('result_round_'+str(i)+'_before_math_analysis.csv', index=False)

