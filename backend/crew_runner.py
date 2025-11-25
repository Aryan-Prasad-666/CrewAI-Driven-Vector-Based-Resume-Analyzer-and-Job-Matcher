import os
import shutil
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import PDFSearchTool, SerperDevTool

load_dotenv()

def run_crew():
    print(" Starting CrewAI pipeline")

    result_dir = "backend/results"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    print(" Cleaned results directory")

    resume_path = "backend/Sample_resume1.pdf"
    if not os.path.exists(resume_path):
        raise FileNotFoundError(" Resume not found. Please upload it first.")

    gemini_key = os.getenv("GOOGLE_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")

    if not gemini_key or not serper_key:
        raise ValueError(" API keys missing in .env")

    llm_general = LLM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=gemini_key,
        temperature=0.5
    )

    llm_deterministic = LLM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=gemini_key,
        temperature=0.0
    )

    pdf_tool = PDFSearchTool(
        pdf=resume_path,
        config=dict(
            llm=dict(
                provider="google",
                config=dict(
                    model="gemini-2.0-flash",
                    api_key=gemini_key,
                    temperature=0.5
                )
            ),
            embedder=dict(
                provider="google",
                config=dict(
                    model="models/embedding-001",
                    task_type="retrieval_document"
                )
            )
        )
    )

    web_tool = SerperDevTool()

    # === AGENTS ===
    resume_analyzer = Agent(
        role="Resume Analyzer",
        goal="Extract structured candidate information from resume PDF",
        backstory="Expert HR with 10+ years of resume parsing experience.",
        verbose=True,
        llm=llm_general,
        tools=[pdf_tool]
    )

    job_finder = Agent(
        role="Job Finder",
        goal="Find jobs matching resume data across platforms",
        backstory="Career consultant with expertise in LinkedIn and job boards.",
        verbose=True,
        llm=llm_general,
        tools=[web_tool]
    )

    ats_scorer = Agent(
        role="ATS Score Evaluator",
        goal="Assess resume against job listings and simulate ATS ranking",
        backstory="Ex-recruiter now focused on resume compatibility scoring.",
        verbose=True,
        llm=llm_deterministic,
        tools=[web_tool]
    )

    resume_optimizer = Agent(
        role="Resume Optimizer",
        goal="Improve resume content and formatting for ATS compatibility",
        backstory="Professional resume editor specializing in keyword optimization.",
        verbose=True,
        llm=llm_general
    )

    cover_letter_writer = Agent(
        role="Cover Letter Writer",
        goal="Craft customized cover letters using resume and job context",
        backstory="Writer skilled in converting resumes into persuasive letters.",
        verbose=True,
        llm=llm_general
    )

    interview_trainer = Agent(
        role="Interview Coach",
        goal="Generate personalized mock questions based on role and resume",
        backstory="Behavioral interview trainer who prepares candidates with real-world scenarios.",
        verbose=True,
        llm=llm_general
    )

    # === TASKS ===

    t1 = Task(
        description=(
            "Extract structured information from the resume PDF. "
            "Output MUST be a JSON object with the following keys:\n"
            "- name\n- contact\n- skills\n- experience\n- education\n"
            "- certifications\n- languages\n- interests\n\n"
            "Ensure clean formatting and no repetition."
        ),
        expected_output="A JSON object with top-level keys: name, contact, skills, experience, education, certifications, languages, interests.",
        agent=resume_analyzer,
        output_file="backend/results/resume_summary.txt"
    )

    t2 = Task(
        description=(
            "Using the extracted resume summary, search for 5–7 relevant job listings "
            "from major platforms like LinkedIn, Naukri, Indeed, and Glassdoor.\n"
            "Match jobs by skills and experience. Include:\n"
            "- title\n- company\n- location\n- match_reason\n- apply_url"
        ),
        expected_output="A JSON list of matching jobs with title, company, location, reason, and URL.",
        agent=job_finder,
        context=[t1],
        output_file="backend/results/jobs.txt"
    )

    t3 = Task(
        description=(
            "Simulate an ATS scan.\n"
            "Use the resume summary and job listings.\n"
            "Return a score from 0–100 based on keyword overlap, formatting, and relevance.\n"
            "Also include 3 strengths and 3 improvement suggestions."
        ),
        expected_output="JSON with: ATS Score, strengths (list), suggestions (list).",
        agent=ats_scorer,
        context=[t1, t2],
        output_file="backend/results/ats_score_evaluation.txt"
    )

    t4 = Task(
        description=(
            "Rewrite and improve the resume based on ATS suggestions. Add missing keywords, clean formatting, fix structure.\n"
            "Return:\n"
            "- revised_resume (plain text)\n- changes (list of bullet points explaining improvements)"
        ),
        expected_output="JSON with revised_resume and list of changes made.",
        agent=resume_optimizer,
        context=[t3],
        output_file="backend/results/optimized_resume.json"
    )

    t5 = Task(
        description="Write a short (3-paragraph) cover letter personalized to one of the top-matching jobs using the resume.",
        expected_output="Markdown-formatted cover letter written in formal tone.",
        agent=cover_letter_writer,
        context=[t1, t2],
        output_file="backend/results/cover_letter.md"
    )

    t6 = Task(
        description="Generate 7 mock interview questions (technical, behavioral, situational) based on resume and job context.",
        expected_output="JSON array with question, category, difficulty.",
        agent=interview_trainer,
        context=[t1, t2],
        output_file="backend/results/mock_interview_questions.json"
    )

    crew = Crew(
        agents=[
            resume_analyzer, job_finder, ats_scorer,
            resume_optimizer, cover_letter_writer, interview_trainer
        ],
        tasks=[t1, t2, t3, t4, t5, t6],
        process=Process.sequential,
        verbose=True
    )

    print(" Launching CrewAI agents...")
    try:
        result = crew.kickoff()
        print("Crew finished running")
    except Exception as e:
        print(" Crew failed with error:", e)
        raise

    # Validate generated files
    for f in [
        "resume_summary.txt", "jobs.txt", "ats_score_evaluation.txt",
        "optimized_resume.json", "cover_letter.md", "mock_interview_questions.json"
    ]:
        full_path = os.path.join("backend/results", f)
        if os.path.exists(full_path):
            print(f" Output ready: {f}")
        else:
            print(f" Missing output: {f}")
