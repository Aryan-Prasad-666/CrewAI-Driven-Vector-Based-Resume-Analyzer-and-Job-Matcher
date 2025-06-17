from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool, SerperDevTool
from crewai import LLM
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

gemini_key = os.getenv('GOOGLE_API_KEY')
serper_key = os.getenv('SERPER_API_KEY')

file_path = r"Sample_resume1.pdf"

# Initialize Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_key,
    temperature=0.5
)

# PDF Reader for resume input
reader = PDFSearchTool(
    pdf=file_path,
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

# Web search tool for job finding
web_search_tool = SerperDevTool(api_key=serper_key)

# === Agents ===

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

ats_scorer = Agent(
    role='ATS Matching Specialist',
    goal='Evaluate the ATS (Applicant Tracking System) score of the candidate’s resume based on the job listings found in the previous task.',
    backstory='You are an ATS expert who simulates how companies filter candidates using keyword matches and experience relevance.',
    verbose=True,
    llm=gemini_llm,
    tools=[web_search_tool]
)

resume_optimizer = Agent(
    role='Resume Coach',
    goal='Help improve resumes based on ATS feedback',
    backstory='You are a professional resume consultant who rewrites resumes for better ATS compatibility.',
    verbose=True,
    llm=gemini_llm
)

cover_letter_writer = Agent(
    role='Professional Writer',
    goal='Craft tailored cover letters based on resume and job description',
    backstory='You specialize in writing compelling, personalized cover letters for applicants across industries.',
    verbose=True,
    llm=gemini_llm
)

interview_trainer = Agent(
    role='Recruiter Trainer',
    goal='Generate personalized mock interview questions for practice',
    backstory='You are a hiring manager and behavioral interview expert who knows how to probe candidate strengths.',
    verbose=True,
    llm=gemini_llm
)

# === Tasks ===

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
    output_file='resume_summary.txt'
)

find_jobs_task = Task(
    description=(
        "Using the candidate profile from the Resume Analyzer (resume_summary.txt), "
        "search for relevant job listings from Indeed, LinkedIn, Glassdoor, etc.\n"
        "Match jobs to candidate's background and give output in JSON."
    ),
    expected_output=(
        "A JSON list of at least 5 jobs with:\n"
        "- Title\n- Company\n- Location\n- Match reason\n- Apply URL"
    ),
    agent=job_finder,
    output_file='jobs.txt',
    context=[analyze_resume_task]
)

ats_score_evaluation_task = Task(
    description=(
        "Evaluate the ATS score of the resume based on the job listings. "
        "Provide a score (0–100), explain the match, and suggest improvements."
    ),
    expected_output=(
        "A JSON object with:\n"
        "- ATS Score\n- 3–4 bullet points on match quality\n- Suggestions for improvement"
    ),
    agent=ats_scorer,
    output_file='ats_score_evaluation.txt',
    context=[find_jobs_task, analyze_resume_task]
)

optimize_resume_task = Task(
    description=(
        "Improve the resume based on ATS feedback. Add missing keywords, tweak phrasing, "
        "and restructure if needed to increase ATS compatibility."
    ),
    expected_output=(
        "A JSON with:\n- revised_resume (text)\n- changes (list of improvements)"
    ),
    agent=resume_optimizer,
    context=[ats_score_evaluation_task],
    output_file='optimized_resume.json'
)

write_cover_letter_task = Task(
    description=(
        "Using the resume and job listings, write a tailored cover letter that highlights the candidate’s strengths and fit for the job."
    ),
    expected_output=(
        "A cover letter formatted as markdown (3 paragraphs), personalized to one of the matched jobs."
    ),
    agent=cover_letter_writer,
    context=[analyze_resume_task, find_jobs_task],
    output_file='cover_letter.md'
)

interview_question_task = Task(
    description=(
        "Generate 7 personalized mock interview questions based on the resume and job roles. "
        "Include behavioral, technical, and situational types."
    ),
    expected_output=(
        "A JSON list with:\n- question\n- category\n- difficulty"
    ),
    agent=interview_trainer,
    context=[analyze_resume_task, find_jobs_task],
    output_file='mock_interview_questions.json'
)

# === CREW ===

crew = Crew(
    agents=[
        resume_analyzer,
        job_finder,
        ats_scorer,
        resume_optimizer,
        cover_letter_writer,
        interview_trainer
    ],
    tasks=[
        analyze_resume_task,
        find_jobs_task,
        ats_score_evaluation_task,
        optimize_resume_task,
        write_cover_letter_task,
        interview_question_task
    ],
    process=Process.sequential,
    verbose=True
)

# === RUN CREW ===

try:
    result = crew.kickoff()
    print("Crew execution completed successfully.")
    print(result)
except Exception as e:
    print(f"Error during crew execution: {str(e)}")
