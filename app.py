from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import re # Make sure this is imported

from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool, SerperDevTool
from crewai import LLM
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_change_me'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

load_dotenv()
# Ensure these environment variables are set in your .env file
gemini_key = os.getenv('ARYAN_GEMINI_KEY') 
serper_key = os.getenv('SERPER_API_KEY')

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def initialize_llm():
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=gemini_key,
        temperature=0.5
    )

def create_pdf_tool(pdf_path):
    return PDFSearchTool(
        pdf=pdf_path,
        config=dict(
            llm=dict(
                provider="google",
                config=dict(
                    model="gemini/gemini-2.5-flash", 
                    api_key=gemini_key,
                    temperature=0.5
                ),
            ),
            embedding_model=dict(
                provider="sentence-transformer", 
                config=dict(model=EMBEDDING_MODEL_NAME),
            ),
            vectordb=dict(
                provider="chromadb",
                config=dict(collection_name="resume_project")
            ),
        ),
    )

def create_web_search_tool():
    return SerperDevTool(api_key=serper_key)

def create_agents_and_tasks(pdf_tool, web_tool, llm):
    resume_analyzer = Agent(
        role='Resume Analyzer',
        goal='Extract detailed information such as skills, work experience, and education from a candidate’s resume text.',
        backstory='You are a senior HR professional with years of experience in resume analysis and candidate profiling.',
        verbose=False,
        llm=llm,
        tools=[pdf_tool],
    )

    job_finder = Agent(
        role='Job Finder',
        goal='Identify relevant job opportunities that match the candidate’s skills and background extracted from the resume.',
        backstory='You are a career consultant with deep knowledge of various industries and job markets. You specialize in matching candidate profiles to ideal job listings.',
        verbose=False,
        llm=llm,
        tools=[web_tool]
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
            "Output MUST be in JSON format. STRICT REQUIREMENT: The final output file MUST contain ONLY the JSON object. Do not include any preamble, explanation, or markdown wrappers like ```json."
        ),
        expected_output=(
            "A JSON summary of the candidate's resume including skills, experiences, "
            "education, certifications, and personal info."
        ),
        agent=resume_analyzer,
        output_file='temp_resume_summary.txt'
    )

    find_jobs_task = Task(
        description=(
            "Using the candidate profile from the Resume Analyzer (temp_resume_summary.txt), "
            "search for relevant job listings from Indeed, LinkedIn, Glassdoor, etc.\n"
            "Match jobs to candidate's background and give output in JSON. STRICT REQUIREMENT: The final output file MUST contain ONLY the JSON array of jobs. Do not include any preamble, explanation, or markdown wrappers like ```json."
        ),
        expected_output=(
            "A JSON list of at least 5 jobs with:\n"
            "- Title\n- Company\n- Location\n- Match reason\n- Apply URL"
        ),
        agent=job_finder,
        output_file='temp_jobs.txt',
        context=[analyze_resume_task]
    )

    return [resume_analyzer, job_finder], [analyze_resume_task, find_jobs_task]

# 💡 UPDATED FUNCTION FOR ROBUST JSON EXTRACTION
def clean_json_output(content):
    content = content.strip()
    
    # 1. Remove markdown wrapper (if present)
    if content.startswith('```json'):
        content = content[7:].strip()
    if content.startswith('```'):
        content = content[3:].strip()
    if content.endswith('```'):
        content = content[:-3].strip()
        
    # 2. Use regex to find and extract the outermost JSON object/array
    # This searches for the structure starting with { or [ and ending with the corresponding } or ].
    match = re.search(r'(\{|\[).*(\]|\})', content, re.DOTALL)
    if match:
        return match.group(0).strip()
    
    # Fallback to original content if no clear JSON structure is found
    return content

def run_crew(pdf_path):
    llm = initialize_llm()
    pdf_tool = create_pdf_tool(pdf_path)
    web_tool = create_web_search_tool()
    agents, tasks = create_agents_and_tasks(pdf_tool, web_tool, llm)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    try:
        # The kickoff might be what's causing the underlying issue if it fails silently
        # to produce the output files. We proceed assuming the CrewAI part works.
        result = crew.kickoff()
        
        # --- Resume Summary Parsing ---
        with open('temp_resume_summary.txt', 'r') as f:
            raw_content = f.read()
            cleaned_resume = clean_json_output(raw_content)
            resume_summary = json.loads(cleaned_resume)
        
        # --- Jobs List Parsing ---
        with open('temp_jobs.txt', 'r') as f:
            raw_content = f.read()
            cleaned_jobs = clean_json_output(raw_content)
            jobs = json.loads(cleaned_jobs)
        
        # Clean up temporary files
        os.remove('temp_resume_summary.txt')
        os.remove('temp_jobs.txt')
        
        # Ensure 'jobs' is a list (if the LLM outputted an object instead of an array)
        if isinstance(jobs, dict):
            # Attempt to find the list of jobs if wrapped in an object
            jobs_list = []
            for value in jobs.values():
                if isinstance(value, list):
                    jobs_list = value
                    break
            jobs = jobs_list
            
        return {
            'success': True,
            'resume': resume_summary,
            'jobs': jobs,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        # Include the content that failed to parse for better debugging
        error_details = {
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Attempt to read the raw outputs for debugging visibility
        try:
            with open('temp_resume_summary.txt', 'r') as f:
                error_details['raw_resume_output'] = f.read()
        except FileNotFoundError:
            error_details['raw_resume_output'] = 'File not created or accessible.'
            
        try:
            with open('temp_jobs.txt', 'r') as f:
                error_details['raw_jobs_output'] = f.read()
        except FileNotFoundError:
            error_details['raw_jobs_output'] = 'File not created or accessible.'
            
        return {
            'success': False,
            'error': f"JSON Error: {error_details['error']}. Check temp outputs for content. Resume Output: '{error_details['raw_resume_output'][:100]}...', Jobs Output: '{error_details['raw_jobs_output'][:100]}...'",
            'timestamp': error_details['timestamp']
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        flash('No file selected.')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = run_crew(filepath)
        os.remove(filepath)
        
        if results['success']:
            return render_template('results.html', 
                                 resume=results['resume'], 
                                 jobs=results['jobs'],
                                 timestamp=results['timestamp'])
        else:
            flash(f'Processing Error: {results["error"]}')
            return redirect(url_for('index'))
    else:
        flash('Please upload a valid PDF file.')
        return redirect(url_for('index'))

@app.route('/progress')
def progress():
    return jsonify({'progress': 50, 'status': 'Processing...'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)