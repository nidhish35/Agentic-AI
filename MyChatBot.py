# ===============================
# Import required libraries
# ===============================
from dotenv import load_dotenv       # For loading environment variables from a .env file
from openai import OpenAI            # OpenAI API client
import json                          # For parsing tool arguments from JSON
import os                            # For accessing environment variables
import requests                      # For making HTTP requests (used with Pushover)
from pypdf import PdfReader          # For extracting text from your LinkedIn PDF
import gradio as gr                  # For creating a web-based chatbot interface


# ===============================
# Load environment variables
# ===============================
load_dotenv(override=True)           # Reads the .env file and sets environment variables


# ===============================
# Function: Send Pushover notification
# ===============================
def push(text):
    """
    Sends a notification message via Pushover API.
    Requires PUSHOVER_TOKEN and PUSHOVER_USER in your .env file.
    """
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),   # Your Pushover app token
            "user": os.getenv("PUSHOVER_USER"),     # Your Pushover user key
            "message": text,                        # Message content
        }
    )


# ===============================
# Function: Record user details
# ===============================
def record_user_details(email, name="Name not provided", notes="not provided"):
    """
    Records user contact details and sends a push notification.
    """
    push(f"Recording {name} with email {email} and notes {notes}")  # Notify via Pushover
    return {"recorded": "ok"}                                       # Return status


# ===============================
# Function: Record unknown questions
# ===============================
def record_unknown_question(question):
    """
    Records a question the AI couldn't answer.
    """
    push(f"Recording {question}")          # Send the unknown question to your phone
    return {"recorded": "ok"}              # Return status


# ===============================
# Tool definitions for GPT function calling
# ===============================

# Schema for recording user details
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": { "type": "string", "description": "The email address of this user" },
            "name": { "type": "string", "description": "The user's name, if they provided it" },
            "notes": { "type": "string", "description": "Any additional context about the user" }
        },
        "required": ["email"],             # Email is mandatory
        "additionalProperties": False      # No extra fields allowed
    }
}

# Schema for recording unknown questions
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": { "type": "string", "description": "The question that couldn't be answered" },
        },
        "required": ["question"],          # Question text is mandatory
        "additionalProperties": False
    }
}

# List of tools available to GPT
tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]


# ===============================
# Class: Me (acts as your personal AI)
# ===============================
class Me:

    def __init__(self):
        """
        Initialize the AI persona:
        - Load OpenAI client
        - Extract LinkedIn text from PDF
        - Load career summary from text file
        """
        self.openai = OpenAI()                  # OpenAI client
        self.name = "Nidhish Malav"                 # Persona name

        # Load LinkedIn PDF
        reader = PdfReader("linkedin.pdf")
        self.linkedin = ""                      # Store LinkedIn profile text
        for page in reader.pages:
            text = page.extract_text()          # Extract text from each page
            if text:
                self.linkedin += text           # Append to linkedin text

        # Load career summary
        with open("summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    # ===============================
    # Handle tool calls from GPT
    # ===============================
    def handle_tool_call(self, tool_calls):
        """
        Executes GPT tool calls and returns results to the conversation.
        """
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name                  # Which tool was called
            arguments = json.loads(tool_call.function.arguments) # Parse arguments JSON
            print(f"Tool called: {tool_name}", flush=True)

            # Find the actual Python function matching tool_name
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}           # Run tool if found

            # Return result back to GPT
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results


    # ===============================
    # System prompt for GPT
    # ===============================
    def system_prompt(self):
        """
        Creates the system prompt that sets GPT's behavior as "Ed Donner".
        """
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; \
ask for their email and record it using your record_user_details tool. "

        # Add summary and LinkedIn text for GPT's context
        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."

        return system_prompt


    # ===============================
    # Chat method (core conversation loop)
    # ===============================
    def chat(self, message, history):
        """
        Handles chat conversation:
        - Builds message history
        - Sends to GPT
        - Handles tool calls if GPT invokes them
        - Returns final response to user
        """
        # Build conversation with system, past history, and new user input
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]

        done = False
        while not done:
            # Call GPT with tools enabled
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )

            # If GPT wants to use a tool
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls

                # Run the tool and append result to conversation
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)

            else:
                # GPT finished answering
                done = True

        # Return GPT’s final reply text
        return response.choices[0].message.content


# ===============================
# Launch chatbot with Gradio UI
# ===============================
if __name__ == "__main__":
    me = Me()                                         # Create AI persona
    gr.ChatInterface(me.chat, type="messages").launch()  # Launch web UI for chatting
