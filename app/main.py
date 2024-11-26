from fastapi import FastAPI, UploadFile, Request
from app.models.file_processing import extract_text, parse_sections, extract_text_llama
from app.models.summarization import generate_summary
from app.models.compliance_check import check_compliance
from app.models.technical_content import generate_technical_content
from jinja2 import Template
from fastapi.templating import Jinja2Templates

from datetime import datetime

import shutil
import os

app = FastAPI(title="Augier AI", description="A simple AI-powered proposal generator")

Templates = Jinja2Templates(directory="app/templates")

# Directory for temporary file storage
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def home(request: Request):
    return "The server is running healthy!"

@app.post("/generate_proposal")
async def generate_proposal(file: UploadFile, request: Request):

    # This is an example of how to use the file_processing module

    # # Extract text and sections using LlamaParser
    # sections = extract_text_llama(file.file)

    # # Summarization
    # summary = generate_summary(sections["Opportunity Details"])

    # # Compliance Check
    # compliance = check_compliance(" ".join(sections.values()), ["FAR", "HIPAA", "GDPR"])

    # # Technical Content Generation
    # technical_content = generate_technical_content(
    #     sections["Requirements"], "Cloud Migration Expertise"
    # )

    # Save the uploaded file temporarily using FastAPI's UploadFile
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(file_location)

    text = extract_text(file_location)
    sections = parse_sections(text)

    os.remove(file_location)
    

    summary = generate_summary(text)
    compliance = check_compliance(text, ["FAR", "HIPAA", "GDPR"])
    technical_content = generate_technical_content(
        sections["Requirements"], "Cloud Migration Expertise"
    )

    template = Template(open("app/templates/proposal_template.jinja2").read())
    # proposal = template.render(
    #     summary=summary, compliance=compliance, technical_content=technical_content
    # )
    # return {"proposal": proposal}
    return Templates.TemplateResponse(
    "proposal.html",
    {
        "request": request,
        "summary": summary,
        "compliance": compliance,  # Ensure this is a list
        "technical_content": technical_content,
        "year": datetime.now().year,
    },
)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
    print("Server started on http://localhost:8080")
