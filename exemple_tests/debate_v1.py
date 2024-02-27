from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub, VLLM
from langchain_openai import OpenAI
from utils import getOpponentResponses, printDebate
import random
import os

random.seed(42)

# ------ API KEYS MANAGEMENTS ------- #
# don't forget to write your key bellow
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['OPENAI_API_KEY'] = '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'



# ------ AGENT IMPORTATION ------- #

AGENT_IDS = [
    #'google/gemma-7b-it',
    #'mosaicml/mpt-7b', too big
    'openai',
    'openai',
    'openai',
    #'google/flan-t5-large',
    # continu here
]

summurizer = OpenAI()
agents = []
for id in AGENT_IDS:
    
    if id == 'openai':
        agents.append(OpenAI(temperature=random.random()))
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
    
    #a lot of dependencie issues but would be much better
    # agents.append(
    #     VLLM(
    #         model=id,
    #         trust_remote_code=True,  # mandatory for hf models
    #         max_new_tokens=128,
    #         top_k=10,
    #         top_p=0.95,
    #         temperature=0.8,
    #     )
    # )



# ------ TEMPLATES AND MEMORIES ------- #

# first round template
template_1 = """
You are in a debate on a question. provide your structured answer.
---------
question : {question}
---------

Answer :

"""

prompt_1 = PromptTemplate(template=template_1, input_variables=["question"])

# middle rounds debate template
template_2 = """
You are in a debate with agents and the goal is to reach a consensus.
Using your previous answer (delimited by <hs></hs>) and opponents answers (delimited by <ctx></ctx>) as additionnal advices,
can you give an updated response.
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

# summurising template 
stemplate = """
You are a judge in a debate and you want to summurize the overall idea given by all the parties based on their final answers.
------------
final_answers : {final_answers}
------------
"""
spromt = PromptTemplate(template=stemplate, input_variables=["final_answers"])


# memories 
memories = []
for agent in agents:
    memories.append(ConversationBufferWindowMemory(memory_key="chat_history", k=5))

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

schain = LLMChain(
            llm=summurizer,
            prompt=spromt,
            verbose=True
        )
  
 # visualization
printDebate(responses)

print(schain.predict(final_answers = [responses[i][-1] for i in responses].join()))
# save the answers : 
# ...