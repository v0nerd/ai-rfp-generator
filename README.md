# RFP Proposal Generator

## Overview

An end-to-end system for generating proposals using fine-tuned models.

## Features

- Text summarization using BART
- Compliance classification using BERT
- Technical content generation using GPT-4

## Usage

- Run the app: `uvicorn app.main:app --reload --port 8080`
- Test endpoints via `/docs`

##  Setting Up AWS Environment

- Install AWS CLI:

```
pip install awscli
aws configure
```

# Setting Up Local Docker Environment

```
docker build -t rfp-generator .
docker run -p 8080:8080 rfp-generator
```

# Setting Up AWS Elastic Container Registry

```
aws configure
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker tag my-app:latest public.ecr.aws/namespace/my-app:v1.0
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/rfp-generator:latest
docker push public.ecr.aws/namespace/my-app:v1.0
```