# pip install langchain huggingface_hub transformers --quiet

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAI
import os


# ------ API KEYS MANAGEMENTS ------- #
# don't forget to write your key bellow
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '...'
os.environ['OPENAI_API_KEY'] = '...'



# ------ AGENT IMPORTATION ------- #
agent2 = HuggingFaceHub(
    repo_id="google/gemma-7b-it",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 150,
        "top_k": 30,
        "temperature": 1e-10,
        "repetition_penalty": 1.03,
    },
)

agent1 = OpenAI()

#agent2 = HuggingFaceHub(
#    repo_id="lmsys/fastchat-t5-3b-v1.0",
#    task="text-generation",
#    model_kwargs={
#        "max_new_tokens": 150,
#        "top_k": 30,
#        "temperature": 1e-10,
#        "repetition_penalty": 1.03,
#    },
#)


# ------ TEMPLATES AND MEMORIES ------- #

# first round debate template
template_1 = """
Youre are in the first round of a debate and you have to give your point of view about the question.
Since you are a professional, you answers are consice and well argumented.
A professional does not give unneccessary points for the debate.
---------
question : {question}
---------

Answer : 

"""

prompt_1 = PromptTemplate(template=template_1, input_variables=["question"])

# middle rounds debate template
template_2 = """
You are in a debate with a human and you are trying to reach a consencus.
Answer the question based of your previous answer (delimited by <hs></hs>) and use opponent human answer (delimited by <ctx></ctx>) to update 
if necessary your previous answer to the question.
--------
<hs>
{chat_history}
</hs>
--------
<ctx>
{agent_answer}
</ctx>
--------

Answer : 
"""

prompt_2 = PromptTemplate(template=template_2, input_variables=["chat_history", "agent_answer"])

# last round template
# TODO

# memory 

# 3 types of memory : full, window and summary
memory1 = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
memory2 = ConversationBufferWindowMemory(memory_key="chat_history", k=3)

# ------  DEBATE ------- #

chain11 = LLMChain(
    llm=agent1,
    prompt=prompt_1,
    verbose=True,
    memory=memory1
)

chain12 = LLMChain(
    llm=agent2,
    prompt=prompt_1,
    verbose=True,
    memory=memory2
)

chain21 = LLMChain(
    llm=agent1,
    prompt=prompt_2,
    verbose=True,
    memory=memory1
)

chain22 = LLMChain(
    llm=agent2,
    prompt=prompt_2,
    verbose=True,
    memory=memory2
)

question = "Should artificial intelligence be granted legal personhood, and what implications would that have on society?"

counter = -1
answers1 = []
answers2 = []
while counter < 5:
    counter += 1
    
    if counter == 0:
        answers1.append(chain11.predict(question = question))
        answers2.append(chain12.predict(question = question))
        continue
    # for a reason that I don't know some models gives the enire template in the answer,
    # then some formating is necessary to have a reasonable context size
    txt2 = answers2[-1].split("Answer : ")[-1]
    answers1.append(chain21.predict(agent_answer=txt2))
    answers2.append(chain22.predict(agent_answer=answers1[-1]))
  
  
 # visualization
for i, (text1, text2) in enumerate(zip(answers1, answers2)):
    print(f" ------------ Round {i + 1} --------- \n\n")
    print(f"Agent 1 : {text1}\n")
    print(f"Agent 2 : {text2}\n")
   
# save the answers : 
# ...