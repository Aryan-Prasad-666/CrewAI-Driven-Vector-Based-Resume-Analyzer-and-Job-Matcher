from crewai import Agent
from crewai import Task, Crew
from crewai_tools import PDFSearchTool, SerperDevTool
from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()

gemini_key = os.getenv('GOOGLE_API_KEY')
serper_key = os.getenv('SERPER_API_KEY')

file_path = r"Sample_resume1.pdf"

gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_key,
    temperature=0.5
)

reader = tool = PDFSearchTool(
    pdf = file_path,
    config=dict(
        llm=dict(
            provider="google",
            config=dict(
                model="gemini-2.0-flash", 
                api_key=gemini_key,
                temperature=0.5
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document"
            ),
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
        "Ensure the information is presented clearly and can be used for job-matching purposes."
        "Give in pure json format"
    ),
    expected_output=(
        "A detailed summary of the candidate's resume including skills, experiences, "
        "education, certifications, and personal info in JSON format."
        "Give in pure json format"
    ),
    agent=resume_analyzer,
    output_file='resume_summary.txt'
)

find_jobs_task = Task(
    description=(
        "Using the candidate profile from the Resume Analyzer (stored in resume_summary.json), including skills, work experience, education, and certifications, "
        "search for relevant job listings from real-time sources such as Indeed, Naukri, LinkedIn, and Glassdoor.\n\n"
        "Use these platforms to find jobs that best match the candidate’s profile. Ensure that roles are current, relevant, and match the candidate’s background.\n"
        "You are allowed to search using web queries (simulate or suggest URLs if direct API access is not available)."
        "Give in pure json format"
    ),
    expected_output=(
        "A list of at least 5 job roles in JSON format with the following info:\n"
        "- Job Title\n"
        "- Company\n"
        "- Location\n"
        "- Matching reason (based on candidate profile)\n"
        "- URL to apply (if available)"
        "Give in pure json format"
    ),
    agent=job_finder,
    output_file='jobs.txt',
    context=[analyze_resume_task]
)

crew = Crew(
    agents=[resume_analyzer, job_finder],
    tasks=[analyze_resume_task, find_jobs_task],
    verbose=True
)

try:
    result = crew.kickoff()
    print("Crew execution completed successfully.")
    print(result)
except Exception as e:
    print(f"Error during crew execution: {str(e)}")






