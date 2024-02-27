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
            
            
def save_sample(responses, summarizerChain, name, agent_ids):
    """_summary_

    Args:
        responses (_type_): _description_
        summarizer (_type_): _description_
    """
    
    #TODO : number of round before conseus if so.
    text = ""
    text += "----- Models -----\n"
    for i, repo_id in enumerate(agent_ids):
        text += f"• Agent {i} : {repo_id}\n"
    
    text += " \n -------- FULL DEBATE -------- \n"
    for c in range(len(responses[0])):
        text += f"\n ---- Round {c + 1} ---- \n"
        for i in responses:
            text += f"Agent {i} : {responses[i][c]}\n"
            
    text += "\n ----- Output ------\n"
    text += summarizerChain.predict(final_answers = "\n".join([f"Agent {i} : {responses[i][-1]}" for i in responses]))
    
    with open(f"./samples/{name}.txt", "w") as file:
        file.write(text)