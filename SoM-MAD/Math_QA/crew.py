from agents import agent_student, agent_professor, agent_mediator
from tasks import task_solving, task_grading, task_mediator, task_kickoff, task_single_agent_mediator
from crewai import Crew
from crewai.process import Process

def Crew4task(llm, math_question, n_rounds):
    student = agent_student(llm)
    professor = agent_professor(llm)
    mediator = agent_mediator(llm)

    # Round 1 
    kickoff_task1 =  task_kickoff(math_question, student)
    mediator_task0 = task_single_agent_mediator(math_question, mediator, kickoff_task1)  # Single angent answer extraction
    grading_task1 =  task_grading(math_question, professor, kickoff_task1) 
    mediator_task1 = task_mediator(math_question, mediator, kickoff_task1, grading_task1, 'mediator_task1.txt')

    # Round 2
    solving_task2 = task_solving(math_question, student, grading_task1)
    grading_task2 =  task_grading(math_question, professor, solving_task2) 
    mediator_task2 = task_mediator(math_question, mediator, solving_task2, grading_task2, 'mediator_task2.txt') 

    # Round 3
    solving_task3 = task_solving(math_question, student, grading_task2)
    grading_task3 =  task_grading(math_question, professor, solving_task3) 
    mediator_task3 = task_mediator(math_question, mediator, solving_task3, grading_task3, 'mediator_task3.txt') 
    

    agents4task =[student, professor, mediator ]
    task_list = [kickoff_task1, mediator_task0, grading_task1, mediator_task1, solving_task2, grading_task2, mediator_task2, solving_task3, grading_task3, mediator_task3]
        
    crew =  Crew(
      agents = agents4task,
      tasks= task_list,
      process=Process.sequential,
      full_output= True,
      verbose=True, 
      memory = True
      )
    
    # Get your crew to work!
    result = crew.kickoff()  
    output_round0 = mediator_task0.output.raw_output #  Single agent output
    output_round1 = mediator_task1.output.raw_output
    output_round2 = mediator_task2.output.raw_output
    output_round3 = mediator_task3.output.raw_output
    
    return result, output_round0, output_round1, output_round2, output_round3

