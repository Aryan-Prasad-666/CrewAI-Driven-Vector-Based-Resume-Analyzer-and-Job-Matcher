import os
import tempfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import PDFSearchTool, SerperDevTool
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
# Allow CORS for development if the frontend is served separately
CORS(app) 

# --- Configuration (using environment variables from .env) ---
GEMINI_KEY = os.getenv('ARYAN_GEMINI_KEY')
SERPER_KEY = os.getenv('SERPER_API_KEY')
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if not GEMINI_KEY or not SERPER_KEY:
    print("Error: ARYAN_GEMINI_KEY or SERPER_API_KEY not found in environment variables.")
    # In a real-world scenario, you might want to stop the app or handle this more gracefully.

# --- CrewAI Setup Function ---
def kickoff_crew_analysis(file_path: str):
    """Initializes and runs the CrewAI process with the uploaded PDF."""
    try:
        # 1. Initialize LLM
        gemini_llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=GEMINI_KEY,
            temperature=0.5
        )

        # 2. Initialize Tools
        # PDFSearchTool is initialized with the dynamic file_path
        reader = PDFSearchTool(
            pdf=file_path,
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-2.5-flash", 
                        api_key=GEMINI_KEY,
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

        web_search_tool = SerperDevTool(api_key=SERPER_KEY)

        # 3. Define Agents
        resume_analyzer = Agent(
            role='Resume Analyzer',
            goal='Extract detailed information such as skills, work experience, and education from a candidate’s resume text.',
            backstory='You are a senior HR professional with years of experience in resume analysis and candidate profiling.',
            verbose=False, # Set to False for production endpoint output
            llm=gemini_llm,
            tools=[reader],
            allow_delegation=False
        )

        job_finder = Agent(
            role='Job Finder',
            goal='Identify relevant job opportunities that match the candidate’s skills and background extracted from the resume.',
            backstory='You are a career consultant with deep knowledge of various industries and job markets. You specialize in matching candidate profiles to ideal job listings.',
            verbose=False, # Set to False for production endpoint output
            llm=gemini_llm,
            tools=[web_search_tool],
            allow_delegation=False
        )

        # 4. Define Tasks
        analyze_resume_task = Task(
            description=(
                "Analyze the given resume and extract structured information including:\n"
                "- Candidate's name\n"
                "- Contact details\n"
                "- Skills\n"
                "- Work experience\n"
                "- Education history\n"
                "- Certifications (if any)\n\n"
                "Output MUST be in a single JSON object. DO NOT include any explanatory text outside the JSON."
            ),
            expected_output=(
                "A JSON summary of the candidate's resume including skills, experiences, "
                "education, certifications, and personal info."
            ),
            agent=resume_analyzer
            # Removed output_file since we capture output directly
        )

        find_jobs_task = Task(
            description=(
                "Using the candidate profile from the Resume Analyzer, "
                "search for relevant job listings from Indeed, LinkedIn, Glassdoor, etc.\n"
                "Match jobs to candidate's background and give output as a JSON list of objects. "
                "DO NOT include any explanatory text outside the JSON."
            ),
            expected_output=(
                "A JSON list of at least 5 jobs with:\n"
                "- Title\n- Company\n- Location\n- Match reason\n- Apply URL"
            ),
            agent=job_finder,
            context=[analyze_resume_task]
            # Removed output_file since we capture output directly
        )

        # 5. Create and Kickoff Crew
        crew = Crew(
            agents=[resume_analyzer, job_finder],
            tasks=[analyze_resume_task, find_jobs_task],
            process=Process.sequential,
            verbose=True # Keep verbose for internal debugging logs
        )

        # The result from crew.kickoff() will be the output of the last task (find_jobs_task).
        # We need to capture the output of *both* tasks.
        final_result = crew.kickoff()
        
        # We can reconstruct the results from the tasks internal memory for better structure
        resume_json = analyze_resume_task.output.raw_output
        jobs_json = find_jobs_task.output.raw_output
        
        # Try to parse the raw outputs to ensure they are valid JSON before sending
        try:
            resume_data = json.loads(resume_json)
        except json.JSONDecodeError:
            resume_data = {"error": "Could not parse resume JSON output.", "raw_output": resume_json}

        try:
            jobs_data = json.loads(jobs_json)
        except json.JSONDecodeError:
            jobs_data = {"error": "Could not parse jobs JSON output.", "raw_output": jobs_json}
        
        return {
            "resume_summary": resume_data,
            "job_listings": jobs_data
        }

    except Exception as e:
        # General error handling for CrewAI execution
        return {"error": f"Crew execution failed: {str(e)}"}, 500


# --- Flask Route for File Upload and Analysis ---
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    """Endpoint to handle resume upload and start the CrewAI analysis."""
    if 'resume' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Use tempfile to securely save the uploaded file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)

        try:
            # Run the CrewAI process
            results = kickoff_crew_analysis(temp_file_path)
            
            # Check if the result is an error message
            if 'error' in results:
                return jsonify(results), results.get('status', 500)
                
            return jsonify(results), 200
        
        finally:
            # IMPORTANT: Clean up the temporary file after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

# --- Run Application ---
if __name__ == '__main__':
    # You might need to change the port or host for deployment
    app.run(debug=True, port=5000)