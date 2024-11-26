import os

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 10


def validate_file_extension(filename):
    """
    Validate that the file extension is allowed.
    """
    if "." not in filename:
        raise ValueError("File does not have a valid extension.")
    extension = filename.rsplit(".", 1)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Invalid file extension '{extension}'. Allowed: {ALLOWED_EXTENSIONS}"
        )


def validate_file_size(file):
    """
    Validate that the file size is within the allowed limit.
    """
    file.seek(0, os.SEEK_END)  # Seek to end to get file size
    file_size = file.tell() / (1024 * 1024)  # Convert bytes to MB
    file.seek(0)  # Reset file pointer
    if file_size > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB} MB limit.")


def validate_required_sections(sections, required_fields):
    """
    Validate that all required sections are present in the extracted text.
    """
    missing_fields = [field for field in required_fields if not sections.get(field)]
    if missing_fields:
        raise ValueError(f"Missing required sections: {', '.join(missing_fields)}")


def validate_compliance_rules(rules):
    """
    Validate compliance rules configuration.
    """
    if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
        raise ValueError("Compliance rules must be a list of strings.")
