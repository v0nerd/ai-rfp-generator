from jinja2 import Environment, FileSystemLoader
import os

# Define the directory where your Jinja templates are located
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


def format_proposal(
    executive_summary: str,
    compliance_matrix: str,
    technical_approach: str,
    pricing_strategy: str,
) -> str:
    """
    Format the RFP proposal into a structured document using Jinja2 templates.
    """
    # Load the proposal template
    template = env.get_template("proposal_template.jinja2")

    # Data to fill in the template
    proposal_data = {
        "executive_summary": executive_summary,
        "compliance_matrix": compliance_matrix,
        "technical_approach": technical_approach,
        "pricing_strategy": pricing_strategy,
    }

    # Render the template with the proposal data
    formatted_proposal = template.render(proposal_data)

    return formatted_proposal
