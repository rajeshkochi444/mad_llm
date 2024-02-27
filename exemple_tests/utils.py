from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAI



def getOpponentResponses(querying_agent_index, responses):
    """
    return the processed answers from the opponent agents.
    
        Parameters:
                querying_agent_index (int): index of the agent that ask opponents informations.
                responses (dict) : dict that stores the agent indexes as keys and their corresponding answers in a list.
                
        Returns:
                data (String): processed information coming from opponnent agents.
    """
    data = ""
    for i in range(len(responses)):
        if i == querying_agent_index:
            continue
            
        data += f"Opponnent Agent number {i} : {responses[i][-1]}\n"
    
    # do somthing with those data here : summurize etc
    return data

def printDebate(responses):
    """print the entiere debate turn by turn.

    Args:
        responses (dict): dict that stores the agent indexes as keys and their corresponding answers in a list.
    """
    
    for c in range(len(responses[0])):
        print(f"\n ---- Round {c + 1} ---- ")
        for i in responses:
            print(f"Agent {i} : {responses[i][c]}\n")