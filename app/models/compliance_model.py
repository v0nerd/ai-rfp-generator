from transformers import BertTokenizer, BertForSequenceClassification
import torch

from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role


def load_compliance_model(model_path="models/fine_tuned_bert"):
    """
    Load the fine-tuned multi-label compliance classification model.
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


def check_compliance(text, compliance_rules):
    """
    Check compliance of a given text against specified compliance rules.
    Parameters:
        text (str): The input text to evaluate.
        compliance_rules (list): A list of compliance rules to evaluate (e.g., ["FAR", "HIPAA", "GDPR"]).
    Returns:
        dict: A dictionary containing compliance scores and issue status for each rule.
    """
    prompt = f"""
    Given the text: {text}
    Choose the compliance rule for each issue:
    {', '.join(compliance_rules)}
    """

    # Define your IAM role
    # role = "arn:aws:iam::194722413586:role/service-role/AmazonSageMaker-ExecutionRole-20241126T074243"

    # # Define your Hugging Face model
    # huggingface_model = HuggingFaceModel(
    #     model_data="s3://rfp-fine-tuned-models/fine_tuned_bert.tar.gz",  # S3 path to your model
    #     role=role,
    #     transformers_version="4.17",  # Match your transformers version
    #     pytorch_version="1.10",  # Match your PyTorch version
    #     py_version="py38",  # Python version
    #     env={"HF_TASK": "compliance"},  # Specify the task
    # )

    # # Deploy the model
    # predictor = huggingface_model.deploy(
    #     initial_instance_count=1,
    #     instance_type="ml.t3.medium",  # Adjust based on expected traffic
    # )

    # print(f"Model deployed successfully. Endpoint Name: {predictor.endpoint_name}")

    # Load the model and tokenizer
    tokenizer, model = load_compliance_model()

    # Define dummy mapping for compliance rules to labels
    rule_to_label_map = {"FAR": "budget", "HIPAA": "privacy", "GDPR": "privacy"}

    # Map compliance rules to model output labels
    labels = ["timeline", "budget", "privacy"]

    # Tokenize the input text
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Predict compliance
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities

    # Interpret results for requested rules
    results = {}

    result_text = ""

    for rule in compliance_rules:
        # Match rule to label
        label = rule_to_label_map.get(rule, None)
        if label and label in labels:
            index = labels.index(label)
            score = probabilities[0][index].item() * 100
            issue = "Compliant" if score >= 50 else "Non-Compliant"
            results[rule] = {"score": f"{score:.2f}%", "issue": issue}
            result_text += f"{rule}: {issue} (Score: {score:.2f}%)\n"
        else:
            results[rule] = {"score": "N/A", "issue": "Rule not supported"}

    print("Compliance Results:", result_text)

    return result_text


if __name__ == "__main__":
    rfp_text = "This is a sample RFP text."
    check_compliance(rfp_text, ["FAR", "HIPAA", "GDPR"])
