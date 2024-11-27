import boto3
import json

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

# Load the fine-tuned model
model_path = "models/fine_tuned_bart"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)


sagemaker_runtime = boto3.client(
    "runtime.sagemaker", region_name="us-east-2"
)  # Update region if needed


def summarize_rfp(rfp_text: str) -> str:

    prompt = f"""
    Summarize the RFP content in the following structure.
    
    <STRUCTURE BEGIN>
    1. Overview
    2. Estimated cost
    3. Timeline
    <STRUCTURE END>
    
    <RFP Content BEGIN>
    {rfp_text}
    <RFP Content END>
    """

    # # Define your IAM role
    # role = "arn:aws:iam::194722413586:role/service-role/AmazonSageMaker-ExecutionRole-20241126T074243"

    # # Define your Hugging Face model
    # huggingface_model = HuggingFaceModel(
    #     model_data="s3://rfp-fine-tuned-models/fine_tuned_bart.tar.gz",  # S3 path to your model
    #     role=role,
    #     transformers_version="4.17",  # Match your transformers version
    #     pytorch_version="1.10",  # Match your PyTorch version
    #     py_version="py38",  # Python version
    #     env={
    #     "HF_TASK": "summarization"  # Specify the task
    #     },
    # )

    # # Deploy the model
    # predictor = huggingface_model.deploy(
    #     initial_instance_count=1,
    #     instance_type="ml.g4dn.xlarge",  # Adjust based on expected traffic
    # )

    # print(f"Model deployed successfully. Endpoint Name: {predictor.endpoint_name}")

    # endpoint_name = "huggingface-pytorch-inference-2024-11-27-02-37-55-914"

    # payload = {
    #     "inputs": prompt,
    #     "parameters": {
    #         "max_length": 200,
    #         "min_length": 50,
    #         "length_penalty": 2.0,
    #         "num_beams": 4,
    #         "early_stopping": True,
    #     },
    # }

    # response = sagemaker_runtime.invoke_endpoint(
    #     EndpointName=endpoint_name,
    #     ContentType="application/json",
    #     Body=json.dumps(payload),
    # )
    # print(response)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=500,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
