# %%
import os
import pandas as pd

from crewai import Agent, Task, Crew
from crewai.process import Process
from dotenv import load_dotenv
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.wolfram_alpha import WolframAlphaQueryRun
from langchain.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI

#from langchain_community.llms import VLLM, Ollama
from datasets import load_dataset

from textwrap import dedent
from utils_math import math_analysis


df = pd.read_csv('../cleaned_math_dataset.csv')

subset = df.iloc[:1000]

os.environ["OPENAI_API_KEY"] = "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"
#os.environ['hUGGINGFACEHUB_API_TOKEN'] = "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"
hugg_key = "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"
#lm = HuggingFaceHub(repo_id='google/gemma-2b-it', huggingfacehub_api_token = hugg_key, model_kwargs={"max_new_tokens": 1000})

llm = ChatOpenAI(model='gpt-3.5-turbo')
#llm = ChatOpenAI(model='gpt-4')

math_student = Agent(
    role = dedent(f"""\
                  Math Student at Georgia Tech University"""
                  ),
    goal = dedent(f"""\
                  Solve the given mathematic problems and summarize your apprach."""
                  ),
    backstory = dedent("""\
                       You are one of the best students in the Maths department of the university. 
                       You can solve simple to complex mathematics problems. 
                       You always seek feedback and advice from  your professor and update your approach to solve the problem incoporating the advice from your professor"""
                       ),
    llm = llm,
    verbose=True,
    allow_delegation=False,
    #tools=[math_tool],
    memory = True
    )

math_professor = Agent(
    role = dedent(f"""\
                  Math professor at Georgia Tech University"""
                  ), 
    goal = dedent("""\
                  You check the solutions, correct them, and provide feeback. You provide alternative paths to approach the problem. 
                  You also provide the final right answers of math problems submitted by your students"""
                  ),
    backstory = dedent(f"""\
                       You are a professor at the university and have a PH.D in Mathematics. 
                       You are expert in grading and correcting solutions to math problems. 
                       You always provide excellent feedback to students and think alternative easy ways to solve the math problems"""
                       ),
    llm = llm,
    verbose=True,
    allow_delegation=False, 
    memory = True
)

mediator = Agent(
    role = dedent(f"""\
                  Expert mediator who evaluates the arguments from student and professor and make  a final decision"""
                  ),
    goal = dedent(f"""\
                  Arrivate at final conclusion and final correct solution after evaluating the arguments, calculation steps and solutions provided by both students and professor."""
                  ),
    backstory = dedent(f"""\
                       You are an expert mediator who has many experience in solving complex math problems. 
                       You evalute the arguments and solutions and provided by the student and the professor and arrive at conclusion on correct solutions.
                       You are an expert in grading and correcting solutions to math problems."""
                       ),
    llm = llm,
    verbose=True,
    allow_delegation=False, 
    memory = True
)

def task_solving(math_question):
  return Task(
            description=  dedent(f"""\
                                  Solve the given math problem: {math_question}. 
                                  If feedback or suggestions or advice from professor is available debate on professor's advice or feedback.
                                  If necessary, update or improve your solutions or steps with the professor's feedback and advice. 
                                  Give detailed summary of approach"""),
          
            expected_output= dedent(f"""\
                                    Detailed summary of your steps and/or approach to solve the given problem. Do not forget to include your final answer.
                                    Debate on professor's feedback if available. Improve your answer by """
                                    ),
            #context = [debate],
            agent=math_student,
            output_file="solving_task.txt",
          )


def task_grading(math_question):
    return  Task(
        description= dedent(f"""\
                            For the solution provided for the math problem: {math_question}, debate on the student's answer.
                            You must review the solution, give feedback to student and provide correct solutions.
                            If the solution is wrong, give correct solution with explanation. 
                            Provide alternative pathways or easy methods to solve the math queston if any.
                            If you found the answer is wrong, ask the student to imporve or correct his answer using your comments/feedback 
                            including alternative pathways as suggested."""
                            ),
        expected_output = dedent(f""" \
                                A detailed feedback of student's solution. Give alternative pathways to solutions and provide correct solutions."""
                                ),
        #context = [solving_task],
        agent=math_professor,
        output_file = 'grading_task.txt'
        )


def task_mediator(solving_task, grading_task, math_question):
    return Task(
                  description= dedent(f"""\
                                      Evalute the arguments and solutions provided by the student and the professor for the math problem: {math_question}
                                      Based on the aruguments provided by both the student and professor, think logically and 
                                      arrive at a final conclusion and final correct solution."""
                                      ),
                  expected_output = dedent(f""" \
                                           A single numeric value.  Do not add any prefix or suffix or pharases like "The final answer". Just the final numeric value. 
                                           If your questions ask for interger, report integer values and for floating values, use one or two decimal places. 
                                           Fractions must be reported as a/b format. 
                                           Example output: 32.8
                                           """
                                           ),
                  context = [solving_task, grading_task],
                  agent=mediator,
                  output_file = 'mediator_task.txt'
                  )

def math_question_selection(i):
    question_with_details = subset.iloc[i]
    #print(question_with_details)
    math_question = subset.iloc[i]["Problem"]
    answer_options =  subset.iloc[i]["options"]
    correct_answer = subset.iloc[i]['correct_numeric']
    print("math_question:", math_question)
    print("answer_options:", answer_options, "correct_answer:", correct_answer )
    return math_question, answer_options, correct_answer


result_list = []
for i in range(100):
   print(i)
   math_question, answer_options, correct_answer = math_question_selection(i)

   solving_task = task_solving(math_question) 
   grading_task =  task_grading(math_question) 
   mediator_task = task_mediator(solving_task, grading_task, math_question) 
   crew = Crew(
      agents=[math_student, math_professor, mediator ],
      tasks=[solving_task, grading_task, solving_task, grading_task, solving_task, grading_task, mediator_task ],
      process=Process.sequential,
      full_output= True,
      verbose=True, # You can set it to True, 1 or 2 to different logging levels
  )
   # Get your crew to work!
   result = crew.kickoff()  
   
   print("#############################RESULT#################################################################")
   print(result)
   
   final_output = mediator_task.output.raw_output
   print(f""" Output: {final_output} """)
   result_list.append([final_output, correct_answer])

print(result_list)
df_before_analysis = pd.DataFrame(result_list, columns = ['AI_answer', 'correct_answer'])
df_before_analysis.to_csv('final_results_two_agents_before_math_analysis.csv', index=False)

df_result, bool_counts = math_analysis(result_list)
df_result.to_csv('final_result_two_agents.csv', index=False)

print(df_result)
print(bool_counts)
