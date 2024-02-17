import os
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain


os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''


model_id_1 = 'google/flan-t5-large'
tokenizer_1 = AutoTokenizer.from_pretrained(model_id_1)
model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_id_1, load_in_8bit=False)

pipe_1 = pipeline(
    "text2text-generation",
    model=model_1, 
    tokenizer=tokenizer_1, 
    max_length=100
)

# model_id_2 = 'openai-community/gpt2'
# tokenizer_2 = AutoTokenizer.from_pretrained(model_id_2)
# model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_id_2, load_in_8bit=False)

# pipe_2 = pipeline(
#     "text2text-generation",
#     model=model_2, 
#     tokenizer=tokenizer_2, 
#     max_length=100
# )


# model hosted locall
local_llm_1 = HuggingFacePipeline(pipeline=pipe_1)
print(type(model_1))

# local_llm_2 = HuggingFacePipeline(pipeline=pipe_1)
# print(type(model_2))


# chain creation
template = """Question : {question}

Answer : Explain with arguments."""

promt = PromptTemplate(template=template, input_variables=["question"])

llm_chain_1 = LLMChain(
    prompt=promt,
    llm=local_llm_1
)

# llm_chain_2 = LLMChain(
#     prompt=promt,
#     llm=local_llm_2
# )


#  --- some qestions --- 
question = "Is Paris a good city for tourists?"

# without the chain
print(local_llm_1(question))
print("\n")

# with the chain
data = llm_chain_1(question)
print(data["text"].replace(". ", ".\n"))

# without the chain
# print(local_llm_2(question))
# print("\n")

# # with the chain
# data = llm_chain_2(question)
# print(data["text"].replace(". ", ".\n"))


