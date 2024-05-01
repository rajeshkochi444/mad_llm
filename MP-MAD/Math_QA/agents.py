from crewai import Agent
from textwrap import dedent

def  agent_student(llm):
    return Agent(
        role = dedent(f"""\
                    Math Student at University of Montreal"""
                    ),
        goal = dedent(f"""\
                    Solve the given math question and summarize your apprach."""
                    ),
        backstory = dedent("""\
                        You are one of the best students in the Maths department of the university. 
                        You can solve simple to complex math problems. 
                        You always seek feedback and advice from  your professor and update your approach to solve the problem incoporating the advice from your professor"""
                        ),
        llm = llm,
        verbose=True,
        allow_delegation=False,
        max_iter = 50,
        #tools=[math_tool],
        memory = True
        )

def agent_professor(llm):
    return Agent(
        role = dedent(f"""\
                    Math professor at University of Monteral."""
                    ), 
        goal = dedent("""\
                    You check the solutions submitted by the students, correct them, and provide feeback. 
                    You provide alternative ways to solve the given math problem and provide correct solutions."""
                    ),
        backstory = dedent(f"""\
                        You are a professor at the university and have a PH.D in Mathematics. 
                        You are expert in grading and correcting solutions to math problems. 
                        You always provide excellent feedback to students and think alternative easy ways to solve the math problems"""
                        ),
        llm = llm,
        verbose=True,
        allow_delegation=False, 
        max_iter = 50,
        memory = True
    )

def agent_mediator(llm):
    return Agent(
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

