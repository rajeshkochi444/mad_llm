from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAI
from utils import getOpponentResponses, printDebate
import os


# ------ API KEYS MANAGEMENTS ------- #
# don't forget to write your key bellow
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['OPENAI_API_KEY'] = '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'



# ------ AGENT IMPORTATION ------- #

AGENT_IDS = [
    #'google/gemma-7b-it',
    'openai',
    'openai',
    #'google/flan-t5-large',
    # continu here
]

agents = []
for id in AGENT_IDS:
    
    if id == 'openai':
        agents.append(OpenAI())
        continue

    agents.append(
        HuggingFaceHub(
            repo_id=id,
            task="text2text-generation",
            model_kwargs={
                "max_new_tokens": 150,
                "top_k": 30,
                "temperature": 0.3,
                "repetition_penalty": 1.03,
            },
        )
    )



# ------ TEMPLATES AND MEMORIES ------- #

# first round template
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

# memories 
memories = []
for agent in agents:
    memories.append(ConversationBufferWindowMemory(memory_key="chat_history", k=3))

# ------  DEBATE ------- #

FirstRoundChains = {}
for i, agent in enumerate(agents):
    FirstRoundChains[i] = LLMChain(
                    llm=agent,
                    prompt=prompt_1,
                    verbose=True,
                    memory=memories[i]
                )


DebateRoundChains = {}
for i, agent in enumerate(agents):
    DebateRoundChains[i] = LLMChain(
                    llm=agent,
                    prompt=prompt_2,
                    verbose=True,
                    memory=memories[i]
                )
    

# the question has to be automated accorifn to a specific dataset
question = "what is two plus two ?"
nbRounds = 2
responses = {} # key : index of the agents | values : list of answers
for c in range(nbRounds):
    if c == 0:
        for i, agent in enumerate(agents):
            responses[i] = []
            responses[i].append(FirstRoundChains[i].predict(question = question))
        continue
    
    # use the response of all other agent 
    # can be concatenated or summurized 
    for i, agent in enumerate(agents):
        responses[i].append(
            DebateRoundChains[i].predict(
                agent_answer=getOpponentResponses(querying_agent_index = i, responses = responses)
            )
        )

  
 # visualization
printDebate(responses)
   
# save the answers : 
# ...