import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")


def generate_technical_content(requirements, expertise):
    """
    Generate a technical approach based on given requirements and expertise.

    Parameters:
        requirements (str): A description of the requirements.
        expertise (str): A description of the expertise or domain knowledge.

    Returns:
        str: The generated technical approach content.
    """
    try:
        # Initialize ChatOpenAI with the API key and model configuration
        chat_gpt = ChatOpenAI(
            api_key=api_key,  # Replace with your actual API key or use environment variables for security
            model="gpt-4",
            temperature=0,  # Control creativity (lower = more deterministic)
        )

        # Prepare the message for the AI model
        messages = [
            HumanMessage(
                content=(
                    f"Generate a technical approach based on the following requirements:\n"
                    f"Requirements: {requirements}\n"
                    f"Expertise: {expertise}\n"
                    f"Output a detailed and structured explanation phase to phase."
                )
            ),
        ]

        # Call the AI model to get a response
        ai_message = chat_gpt.invoke(messages)

        # Display the generated content
        print("Generated Technical Content:", ai_message.content)

        return ai_message.content

    except Exception as e:
        # Handle any errors that occur during the process
        print(f"Error generating technical content: {e}")
        return "An error occurred while generating the technical content."


if __name__ == "__main__":
    generate_technical_content("Cloud Platform", "Financial Information Technology")
