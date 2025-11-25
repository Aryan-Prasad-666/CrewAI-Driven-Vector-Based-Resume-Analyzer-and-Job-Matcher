from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
from crew_runner import run_crew

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
def upload_resume(file: UploadFile = File(...)):
    resume_path = Path("backend/Sample_resume1.pdf")
    with open(resume_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "✅ Resume uploaded successfully"}

def safe_read(path: str):
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return "❌ File not generated. Check pipeline or logs."

@app.post("/run/")
def run_pipeline():
    try:
        run_crew()
    except Exception as e:
        return {"error": f"❌ Pipeline crashed: {str(e)}"}

    return {
        "resume_summary": safe_read("backend/results/resume_summary.txt"),
        "jobs": safe_read("backend/results/jobs.txt"),
        "ats": safe_read("backend/results/ats_score_evaluation.txt"),
        "optimized_resume": safe_read("backend/results/optimized_resume.json"),
        "cover_letter": safe_read("backend/results/cover_letter.md"),
        "interview_questions": safe_read("backend/results/mock_interview_questions.json")
    }
