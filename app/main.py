import os
import time
from datetime import datetime
from dotenv import load_dotenv

import json

import boto3
from botocore.exceptions import NoCredentialsError
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Request
from jinja2 import Template
from fastapi.templating import Jinja2Templates

from app.formatting import format_proposal
from app.ocr_processing import extract_text
from app.models.compliance_model import check_compliance
from app.models.summarization_model import summarize_rfp
from app.models.technical_content_model import generate_technical_content

from botocore.exceptions import ClientError

Templates = Jinja2Templates(directory="templates")

# Initialize FastAPI
app = FastAPI()

root_router = APIRouter()

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

# AWS S3 Configuration
S3_BUCKET = "raw-rfp-documents"
s3_client = boto3.client("s3", region_name="us-east-2")


def get_openai_api_key():

    secret_name = "ApiKeys"
    region_name = "us-east-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response["SecretString"]

    secret_dict = json.loads(secret)

    openai_api_key = secret_dict["OPENAI_API_KEY"]

    print(openai_api_key)

    return openai_api_key


# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = get_openai_api_key()


@root_router.get("/", status_code=200)
async def home(request: Request):
    return "The server is running healthy!"


app.include_router(root_router)


# Model integration: Summarization, Compliance, and Technical Approach
@app.post("/upload/")
async def upload_rfp(file: UploadFile = File(...)):
    """
    Upload an RFP file to S3.
    """
    try:
        file_contents = await file.read()
        file_key = f"rfps/{file.filename}"
        s3_client.put_object(Bucket=S3_BUCKET, Key=file_key, Body=file_contents)
        return {"message": f"File uploaded successfully as {file_key}"}
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 credentials not configured")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/summarize/")
async def summarize_endpoint(file_key: str):
    """
    Generate an executive summary for an RFP stored in S3.
    """
    try:
        start_time = time.time()
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        file_contents = response["Body"].read()
        rfp_text = extract_text(file_contents)
        time_after_ocr = time.time()
        print(f"Time taken for OCR: {time_after_ocr - start_time} seconds")
        summary = summarize_rfp(rfp_text)
        time_after_summarization = time.time()
        print(
            f"Time taken for summarization: {time_after_summarization - time_after_ocr} seconds"
        )
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-compliance/")
async def compliance_endpoint(file_key: str):
    """
    Check compliance for an RFP stored in S3.
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        file_contents = response["Body"].read()
        rfp_text = extract_text(file_contents)
        compliance = check_compliance(rfp_text, ["HIPAA", "GDPR", "FAR"])
        return {"compliance": compliance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/tech/")
async def technical_content_endpoint(requirements, expertise):
    """
    Generate a technical approach based on RFP requirements.
    """
    try:
        technical_approach = generate_technical_content(requirements, expertise)
        return {"technical_approach": technical_approach}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/rfp/")
async def generate_proposal_endpoint(file_key: str, request: Request):
    """
    Generate a full proposal for an RFP stored in S3.
    """
    try:
        # Get the file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        file_contents = response["Body"].read()
        rfp_text = extract_text(file_contents)

        # Generate each section
        summary = summarize_rfp(rfp_text)
        compliance = check_compliance(rfp_text, ["HIPAA", "GDPR", "FAR"])
        technical_approach = generate_technical_content(
            "Technical requirements extracted from RFP",
            "Java, Python, AWS, and Apache, etc",
        )
        price = "Pricing details extracted from RFP"

        # Format the proposal
        proposal = format_proposal(summary, compliance, technical_approach, price)

        # Save proposal back to S3
        proposal_key = file_key.replace("rfps/", "proposals/").replace(
            ".pdf", "_proposal.html"
        )
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=proposal_key, Body=proposal.encode("utf-8")
        )

        # return {
        #     "message": "Proposal generated successfully",
        #     "proposal_key": proposal_key,
        # }
        return Templates.TemplateResponse(
            "proposal.html",
            {
                "request": request,
                "summary": summary,
                "compliance": compliance,  # Ensure this is a list
                "technical_content": technical_approach,
                "year": datetime.now().year,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("Starting webserver...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        debug=os.getenv("DEBUG", False),
        log_level=os.getenv("LOG_LEVEL", "info"),
        proxy_headers=True,
    )
