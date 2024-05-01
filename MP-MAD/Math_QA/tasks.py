from crewai import Task
from textwrap import dedent

def task_kickoff(math_question, agent):
  return Task(
            description=  dedent(f"""\
                                  Solve the given math problem: {math_question}. Think step by step 
                                  Give detailed summary of approach and final answer"""),
          
            expected_output= dedent(f"""\
                                    Detailed summary of your steps and/or approach to solve the given problem. Do not forget to include your final answer."""
                                    ),
            #context = [debate],
            agent=agent,
            output_file="kickoff_task.txt",
          )

def task_solving(math_question, agent, task_context_grading):
  return Task(
            description=  dedent(f"""\
                                  Solve the given math problem: {math_question} by using your professor's feedback.  Think step by step 
                                  your respone must start with  "I agree with you. Here is my reasons and arguments" or "I disgree with you. here is my reasons/arguments"
                                  If feedback or suggestions or advice from professor is available debate on professor's advice or feedback.
                                  Update or improve your solutions or steps with the professor's feedback and advice. 
                                  Use previous conversions and contexts available for debating. 
                                  Give detailed summary of approach"""),
          
            expected_output= dedent(f"""\
                                    your respone must start with  "I agree with you. Here is my reason" or "I disgree with you. here is my reasons/arguments"
                                    Detailed summary of your steps and/or approach to solve the given problem. Do not forget to include your final answer.
                                    Debate on professor's feedback if available. """
                                    ),
            context = [task_context_grading],
            agent=agent,
            output_file="solving_task.txt",
          )


def task_grading(math_question, agent, task_context_solving):
    return  Task(
        description= dedent(f"""\
                            For the solution provided for the math problem: {math_question}, debate on the student's answer. Think step by step 
                            Your respone must start with "I disgree with you. here is my reasons/arguments" as your are a great critic of students
                            Use previous conversions and contexts available for debating. 
                            You must review the solution, give feedback to student and provide correct solutions.
                            If the solution is wrong, give correct solution with explanation. 
                            Provide alternative pathways or easy methods to solve the math queston if any.
                            If you found the answer is wrong, ask the student to imporve or correct his answer using your comments/feedback 
                            including alternative pathways as suggested."""
                            ),
        expected_output = dedent(f""" \
                                your respone must start with "I disgree with you. here is my reasons/arguments:"
                                A detailed feedback of student's solution. Give alternative pathways to solutions and provide correct solutions."""
                                ),
        context = [task_context_solving],
        agent=agent,
        output_file = 'grading_task.txt'
        )


def task_mediator(math_question, agent, solving_task, grading_task, output_file):
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
                  agent=agent,
                  output_file = output_file
                  )

def task_single_agent_mediator(math_question, agent, kickoff_task):
    return Task(
                  description= dedent(f"""\
                                      Evalute the reasongs and solutions provided by the student for the math problem: {math_question}
                                      Based on the this, extract the final correct solution."""
                                      ),
                  expected_output = dedent(f""" \
                                           A single numeric value.  Do not add any prefix or suffix or pharases like "The final answer". Just the final numeric value. 
                                           If your questions ask for interger, report integer values and for floating values, use one or two decimal places. 
                                           Fractions must be reported as a/b format. 
                                           Example output: 32.8
                                           """
                                           ),
                  context = [kickoff_task],
                  agent=agent,
                  output_file = 'kickoff_task.txt'
                  )