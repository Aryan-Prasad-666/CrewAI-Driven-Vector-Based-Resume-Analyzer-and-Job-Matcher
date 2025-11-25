from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool, SerperDevTool
from crewai import LLM
from dotenv import load_dotenv
import os

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()

gemini_key = os.getenv('GEMINI_API_KEY_SHAKTHI')
serper_key = os.getenv('SERPER_API_KEY')

file_path = r"Sample_resume1.pdf" 

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.5
)

reader = PDFSearchTool(
    pdf=file_path,
    config=dict(
        llm=dict(
            provider="google",
            config=dict(
                model="gemini-2.5-flash", 
                api_key=gemini_key,
                temperature=0.5
            ),
        ),
        
        embedding_model=dict(
            provider="sentence-transformer", 
            config=dict(
                model=EMBEDDING_MODEL_NAME, 
            ),
        ),
        
        vectordb=dict(
            provider="chromadb",
            config=dict(
                collection_name="resume_project", 
            )
        ),
    )
)

web_search_tool = SerperDevTool(api_key=serper_key)

resume_analyzer = Agent(
    role='Resume Analyzer',
    goal='Extract detailed information such as skills, work experience, and education from a candidate’s resume text.',
    backstory='You are a senior HR professional with years of experience in resume analysis and candidate profiling.',
    verbose=True,
    llm=gemini_llm,
    tools=[reader],
)

job_finder = Agent(
    role='Job Finder',
    goal='Identify relevant job opportunities that match the candidate’s skills and background extracted from the resume.',
    backstory='You are a career consultant with deep knowledge of various industries and job markets. You specialize in matching candidate profiles to ideal job listings.',
    verbose=True,
    llm=gemini_llm,
    tools=[web_search_tool]
)

analyze_resume_task = Task(
    description=(
        "Analyze the given resume and extract structured information including:\n"
        "- Candidate's name\n"
        "- Contact details\n"
        "- Skills\n"
        "- Work experience\n"
        "- Education history\n"
        "- Certifications (if any)\n\n"
        "Output MUST be in JSON format."
    ),
    expected_output=(
        "A JSON summary of the candidate's resume including skills, experiences, "
        "education, certifications, and personal info."
    ),
    agent=resume_analyzer,
    output_file='resume_summary.json'  # Changed to .json
)

find_jobs_task = Task(
    description=(
        "Using the candidate profile from the Resume Analyzer (resume_summary.json), "
        "search for relevant job listings from Indeed, LinkedIn, Glassdoor, etc.\n"
        "Match jobs to candidate's background and give output in JSON."
    ),
    expected_output=(
        "A JSON list of at least 5 jobs with:\n"
        "- Title\n- Company\n- Location\n- Match reason\n- Apply URL"
    ),
    agent=job_finder,
    output_file='jobs.json',  # Changed to .json
    context=[analyze_resume_task]
)

crew = Crew(
    agents=[
        resume_analyzer,
        job_finder,
    ],
    tasks=[
        analyze_resume_task,
        find_jobs_task,
    ],
    process=Process.sequential,
    verbose=True
)

# ... (rest of the original agents.py code) ...

try:
    result = crew.kickoff()
    print("Crew execution completed successfully.")
    print(result)
    
    # --- POST-PROCESSING STEP ---
    output_file_path = 'jobs.json'
    print(f"Cleaning {output_file_path}...")
    
    try:
        with open(output_file_path, 'r') as f:
            content = f.read()
        
        # Remove common markdown fence
        if content.strip().startswith('```json'):
            content = content.replace('```json', '', 1).strip()
        if content.strip().endswith('```'):
            content = content.rsplit('```', 1)[0].strip()
            
        # Optional: Attempt to load/save to validate and re-format JSON
        import json
        clean_data = json.loads(content)
        
        with open(output_file_path, 'w') as f:
            json.dump(clean_data, f, indent=4)
        
        print(f"Successfully cleaned and re-saved {output_file_path}.")

    except FileNotFoundError:
        print(f"Warning: Output file {output_file_path} not found for cleaning.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON during cleaning: {e}. File content might be malformed.")
    except Exception as e:
        print(f"An unexpected error occurred during file cleaning: {e}")
        
    # --- END POST-PROCESSING STEP ---

except Exception as e:
    print(f"Error during crew execution: {str(e)}")