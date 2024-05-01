# mad_llm

This repository contains research on the Multi-agent Debate (MAD) approach, designed to enhance AI capabilities through structured arguments among multiple AI agents. The study focuses on the effectiveness of MAD strategies across diverse question-answering scenarios.

The research explores two distinct strategies within the MAD framework:

Society of Minds (SoM-MAD): This strategy leverages the collective intelligence of multiple AI agents, each contributing a unique perspective to the debate.

Multi-Persona: This approach employs multiple AI personas, each designed to argue from a different stance or viewpoint, enriching the discourse and diversity of thought.

The research assesses the adaptability and applicability of MAD in various domains, including:Mathematics, Medicine, Science.

The goal is to explore how MAD, through strategies like SoM-MAD and Multi-Persona, can improve reasoning tasks, providing nuanced, balanced, and comprehensive outputs.

This project offers insights into the potential enhancements that structured AI debates can bring to question-answering systems, aiming to foster more sophisticated and reliable AI decision-making processes.

![image](https://github.com/rajeshkochi444/mad_llm/assets/40799655/4747bfbd-7a06-4aad-b6d2-98fb0f740973)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed the latest version of Python. You can download Python [here](https://www.python.org/downloads/).

## Setting Up Your Development Environment

Follow these steps to set up your development environment:

1. **Clone the Repository**

   Clone this repository to your local machine using the following command:
   
   git clone https://github.com/rajeshkochi444/mad_llm.git
   
   cd mad_llm
   
2. Create and Activate a Virtual Environment
   
   python -m venv myenv
   
   source myenv/bin/activate

3. Install Required Packages
   
   pip install -r requirements.txt

4. Navigate to the SoM-MAD or MP-MAD project directory

   cd SoM-MAD

5. SoM-MAD and MP-MAD supports three different datasets. Choose one to work with by navigating into the corresponding directory

   cd Math_QA (or MedMCQA/SciQ)

6. Configuring the Language Models
   
You need to configure your chosen language model and API keys in the main.py

This project allows the use of various large language models (LLMs) and virtual large language models (VLLMs). 

Specify your LLM and API_KEY or opt for VLLM and HuggingFaceEndpoint configurations for deploying the models.

7. Run the code 

   python main.py

   This command will start the multi-agent debate using the configurations you have set. Ensure that all parameters are correctly configured in your main.py script before executing.

   Adjust any paths or parameters as needed based on your specific setup or HPC requirements. This structured approach helps users navigate the setup process easily and effectively.

8. Completion Notification
   
Upon successful execution, main.py will generate files named result_round_{i}_before_science_analysis.csv for i ranging from 0 to 3

9. Final Answer Extraction

To extract the final answers for the Math_QA datasets, or to determine the correct labels for SciQ and MedMCQA datasets, you will need to use the final answer extractor powered by GPT-3.5-Turbo.

Update the base_fname variable in final_gpt_answer_extraction_original.py to point to the appropriate file generated by main.py.

Execute the script with the following command: python final_gpt_answer_extraction_original.py


   
