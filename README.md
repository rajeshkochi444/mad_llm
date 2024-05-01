# mad_llm
Multi-Agent Debate LLM
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


   
