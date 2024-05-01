from crewai import Task
from textwrap import dedent

def task_kickoff(science_question, options, agent):
  return Task(
            description=  dedent(f"""\
                                  Give the correct answer label from the options given: {options} for the  medical science question: {science_question}. 
                                  Give detailed reasonings and arguments for selecting your answer."""),
          
            expected_output= dedent(f"""\
                                    Detailed summary of your reasonings and arguments for selecting the answer for the given medical science question. 
                                    Do not forget to include your answer and the  correct answer label from the options given {options}.
                                    Example:
                                    Correct answer label: 
                                    Reason/argument:  """
                                    ),
            #context = [debate],
            agent=agent,
            output_file="debate_student1.txt",
          )

def task_debate_student1(science_question, options, agent, debate_student2):
  return Task(
            description=  dedent(f"""\
                                  Give the correct answer label from the options given: {options} for the medical science question: {science_question}. 
                                  Give detailed reasonings and arguments for selecting your answer.
                                  If your friend is debating with his arguments and reasonings, you must debate with your arguments and reasonings for your answer.
                                  His debate arguments are given as context.
                                  If he is wrong, correct him and give correct answer.
                                  If your friend's debate provide you new insights, update your reasoning about answering the medical science question.
                                  You may also use your previous conversations from memory or available contexts for your argumnets and reasonings for your answer."""),
          
            expected_output= dedent(f"""\
                                    Detailed summary of your reasonings and arguments for selecting the answer for the given medical science question. 
                                    If you have any debate with other student, give your debate arguments and reasonings.
                                    Do not forget to include your answer and the  correct answer label from the options given {options}.
                                    Example:
                                    Correct answer label: 
                                    Reason/argument:  """
                                    ),
            context = [debate_student2],
            agent=agent,
            output_file="debate_student1.txt",
          )

def task_debate_student2(science_question, options, agent, debate_student1):
    return  Task(
        description= dedent(f"""\
                            You are big critic of your friend and always debate with him with your counter arguments and reasonings on the given medical science question: {science_question} 
                            His debate arguments are given as context.
                            After debate with your friend, give the correct answer and select the correct answer label from  the options {options}  
                            You may also use your previous conversations from memory and available contexts for your argumnets and reasonings for your answer.                            
                            ."""
                            ),
        expected_output = dedent(f""" \
                                A detailed summary of your reasonings and arguments for selecting the answer for the given medical science question.
                                Give  your debate arguments and reasonings that supports your debate.
                                Do not forget to include your answer and the correct answer label from the given options list {options}.
                                 Example:
                                Correct answer label: 
                                Reason/argument:  """
                                ),
        context = [debate_student1],
        agent=agent,
        output_file = 'debate_student2.txt'
        )


def task_mediator(science_question, options, agent, debate_student1, debate_student2, output_file ):
    return Task(
                  description= dedent(f"""\
                                      Evaluate the arguments and reasonings provided by the two students for their answer  for science question: {science_question}
                                      You must arrive at a conclusion and final answer based on the aruguments and reasonings  provided by both these students.
                                      Provide a single label from options {options} for your answer"""
                                      ),
                  expected_output = dedent(f""" \
                                           Your answer MUST BE a single character label from options {options}.
                                           Do not add any prefix or suffix or pharases like "The final answer". Just the final character label for your answe
                                           Example output: d """),
                  context = [debate_student1, debate_student2],
                  agent=agent,
                  output_file = output_file
                  )

def task_single_agent_mediator(science_question, options, agent, kickoff_task):
    return Task(
                  description= dedent(f"""\
                                       Extract the final answer after evaluting the arguments and reasoning provided by the medical student for the medical science question{science_question}
                                      Provide a single label from options {options} for your answer."""
                                      ),
                  expected_output = dedent(f""" \
                                           Your answer MUST BE a single character label from options {options}.
                                           Do not add any prefix or suffix or pharases like "The final answer". Just the final character label for your answe
                                           Example output: d """
                                           ),
                  context = [kickoff_task],
                  agent=agent,
                  output_file = 'kickoff_task.txt'
                  )
