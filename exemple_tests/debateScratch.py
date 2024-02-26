# pip install langchain huggingface_hub transformers --quiet

import os
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# don't forget to write your key bellow
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '...'

agent1 = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 250},
)

# feel free to change the models
agent2 = HuggingFacePipeline.from_model_id(
    model_id="mosaicml/mpt-7b-chat",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 250},
)


# first round debate template
template_1 = """Question : {question}

Answer: let's give argument and reasonning endorsing the answer.
"""

prompt_1 = PromptTemplate(template=template_1, input_variables=["question"])

# middle rounds debate template
template_2 = """Question : {question}

Answer: Another agent thinks that {other_answer}. I think that {previous_answer}.
let's Use other agent answer to update mine, giving arguments and reasoning.
"""

prompt_2 = PromptTemplate(template=template_2, input_variables=["question", "other_answer", "previous_answer"])

# ---- Debate ---- #

chain11 = prompt_1 | agent1
chain12 = prompt_1 | agent2
chain21 = prompt_2 | agent1
chain22 = prompt_2 | agent2

question = "what is 10 divided by 6 plus 2 minus 1 ? "

nbRounds = -1
answers1 = []
answers2 = []
while nbRounds < 3:
  nbRounds += 1
  
  if nbRounds == 0:
    answers1.append(chain11.invoke({"question" : question}))
    answers2.append(chain12.invoke({"question" : question}))
    continue
  
  answers1.append(chain21.invoke({
      "question" : question,
      "other_answer" : answers2[-1],
      "previous_answer" : answers1[-1]
  }))

  answers2.append(chain22.invoke({
      "question" : question,
      "other_answer" : answers1[-1],
      "previous_answer" : answers2[-1]
  }))
  
  
  
 # visualization
for i, (text1, text2) in enumerate(zip(answers1, answers2)):
   print(f" ------------ Round {i + 1} --------- \n\n")
   print(f"Agent 1 : {text1}\n")
   print(f"Agent 2 : {text2}\n")
   
# save the answers : 
# ...