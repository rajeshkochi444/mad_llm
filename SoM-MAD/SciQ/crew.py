from agents import agent_student1, agent_student2, agent_mediator
from tasks import task_debate_student1, task_debate_student2, task_mediator, task_kickoff, task_single_agent_mediator
from crewai import Crew
from crewai.process import Process

def Crew4task(llm, science_question, options):
    student1 = agent_student1(llm)
    student2 = agent_student2(llm)
    mediator = agent_mediator(llm)

    # Round 1 
    student1_kickoff =  task_kickoff(science_question, options, student1)
    mediator_round0 = task_single_agent_mediator(science_question, options, mediator, student1_kickoff)  # Single angent answer extraction
    student2_debate1 =  task_debate_student2(science_question, options, student2, student1_kickoff) 
    mediator_round1 = task_mediator(science_question, options, mediator, student1_kickoff, student2_debate1, 'mediator_round1.txt')

    # Round 2
    student1_debate2 = task_debate_student1(science_question, options, student1, student2_debate1)
    student2_debate2 =  task_debate_student2(science_question, options, student2, student1_debate2) 
    mediator_round2 = task_mediator(science_question, options, mediator, student1_debate2, student2_debate2, 'mediator_round2.txt') 

    # Round 3
    student1_debate3 = task_debate_student1(science_question, options, student1, student2_debate2)
    student2_debate3 =  task_debate_student2(science_question, options, student2, student1_debate3) 
    mediator_round3 = task_mediator(science_question, options, mediator, student1_debate3, student2_debate3, 'mediator_round3.txt') 
    

    agents4task =[student1, student2, mediator ]
    task_list = [student1_kickoff, mediator_round0, student2_debate1, mediator_round1, student1_debate2, student2_debate2, mediator_round2, student1_debate3, student2_debate3, mediator_round3]
        
    crew =  Crew(
      agents = agents4task,
      tasks= task_list,
      process=Process.sequential,
      #full_output= True,
      verbose=True, # You can set it to True, 1 or 2 to different logging levels
      )
    
    # Get your crew to work!
    result = crew.kickoff()  
    output_round0 = mediator_round0.output.raw_output #  Single agent output
    output_round1 = mediator_round1.output.raw_output
    output_round2 = mediator_round2.output.raw_output
    output_round3 = mediator_round3.output.raw_output
    
    return result, output_round0, output_round1, output_round2, output_round3

