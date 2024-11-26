import os
import uuid
from datetime import datetime


def generate_unique_filename(extension="txt"):
    """
    Generate a unique filename using UUID and timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex
    return f"{timestamp}_{unique_id}.{extension}"


def save_temp_file(file, directory="temp_files"):
    """
    Save an uploaded file to a temporary directory and return its path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = generate_unique_filename(file.filename.split(".")[-1])
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def cleanup_temp_files(directory="temp_files"):
    """
    Remove all files in the temporary directory.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


def format_compliance_result(compliance_result):
    """
    Format compliance check results for better readability.
    """
    if isinstance(compliance_result, list):
        return "\n".join(compliance_result)
    return compliance_result


def format_summary(summary):
    """
    Format summary text with proper capitalization and line breaks.
    """
    return summary.strip().capitalize()


def log_message(message, log_file="logs/app.log"):
    """
    Log a message to a file for debugging or auditing.
    """
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()}: {message}\n")
