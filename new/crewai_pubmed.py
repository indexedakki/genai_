# Install required packages
# !pip install crewai langchain langchain-community chromadb sentence-transformers

from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
from langchain.llms import HuggingFaceHub
# from langchain.embeddings import HuggingFaceEmbeddings
import os
# Set up Hugging Face embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# os.environ["HUGGINGFACE_ACCESS_TOKEN"]

# # Set up Hugging Face LLM
# hf_llm = HuggingFaceHub(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     huggingfacehub_api_token
# )

from litellm import completion
hf_llm = completion(
    model="huggingface/mistralai/Mistral-7B-Instruct-v0.2"
)

pdf_rag_tool = PDFSearchTool(
    pdf='KID.0002002021.pdf',
    config=dict(
        llm=dict(
            provider="huggingface", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                base_url= ""
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

# Create agents
medical_expert = Agent(
    role='Senior Medical Document Analyst',
    goal='Analyze and extract key information from complex medical documents',
    backstory="""An expert in parsing complex medical terminology and research documents,
    with years of experience in clinical research and medical content analysis.""",
    tools=[pdf_rag_tool],
    llm=hf_llm,
    verbose=True
)

summarizer = Agent(
    role='Medical Text Summarization Specialist',
    goal='Create clear and concise summaries of medical information',
    backstory="""A professional medical writer with expertise in distilling complex
    medical information into easy-to-understand summaries for various audiences.""",
    llm=hf_llm,
    verbose=True
)

# Create tasks
analysis_task = Task(
    description="""Analyze the provided medical document and extract key findings,
    treatment protocols, and significant statistical data. Pay special attention to:
    - Drug efficacy data
    - Clinical trial results
    - Patient outcome statistics
    - Notable side effects or warnings""",
    expected_output="A comprehensive report of extracted medical information in bullet points",
    agent=medical_expert,
    tools=[pdf_rag_tool]
)

summary_task = Task(
    description="""Create a patient-friendly summary of the medical document analysis.
    Include:
    - Key findings in simple terms
    - Important statistics simplified
    - Main conclusions
    - Any critical warnings or notes""",
    expected_output="A concise 3-paragraph summary in plain language with key points highlighted",
    agent=summarizer
)

# Form crew
medical_crew = Crew(
    agents=[medical_expert, summarizer],
    tasks=[analysis_task, summary_task],
    process=Process.sequential,
    verbose=True
)

# Execute the crew
result = medical_crew.kickoff()

print("\n\nFinal Summary:")
print(result)
