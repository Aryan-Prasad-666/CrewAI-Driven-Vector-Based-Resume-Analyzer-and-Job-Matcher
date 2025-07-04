import shutil
from pathlib import Path

def save_uploaded_pdf(upload_file, filename='Sample_resume1.pdf'):
    path = Path("backend") / filename
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return str(path)
