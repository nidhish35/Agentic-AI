# ======================================
# Chatbot with Evaluation (by Nidhish)
# ======================================

# Install required packages before running:
# pip install python-dotenv openai pypdf gradio pydantic

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel
import os

# ---------------------------
# Step 1: Load API keys
# ---------------------------
# Put your API keys in a .env file:
# OPENAI_API_KEY=your_openai_key
# GOOGLE_API_KEY=your_google_gemini_key

load_dotenv(override=True)

openai = OpenAI()  # OpenAI client (uses OPENAI_API_KEY)

gemini = OpenAI(   # Gemini client (uses GOOGLE_API_KEY)
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ---------------------------
# Step 2: Read profile data
# ---------------------------
reader = PdfReader("linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

with open("summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

# ---------------------------
# Step 3: System prompt (persona setup)
# ---------------------------
name = "Nidhish"

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."

system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

# ---------------------------
# Step 4: Evaluation schema
# ---------------------------
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

# ---------------------------
# Step 5: Evaluator setup
# ---------------------------
evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
The Agent is playing the role of {name}, who must be professional and engaging. \
Here is {name}'s context:"

evaluator_system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"

def evaluator_user_prompt(reply, message, history):
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt

def evaluate(reply, message, history) -> Evaluation:
    messages = [{"role": "system", "content": evaluator_system_prompt}] + \
            [{"role": "user", "content": evaluator_user_prompt(reply, message, history)}]

    response = gemini.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        messages=messages,
        response_format=Evaluation
    )
    return response.choices[0].message.parsed

# ---------------------------
# Step 6: Retry mechanism
# ---------------------------
def rerun(reply, message, history, feedback):
    updated_system_prompt = system_prompt + "\n\n## Previous answer rejected\n"
    updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"

    messages = [{"role": "system", "content": updated_system_prompt}] + \
            history + [{"role": "user", "content": message}]
    
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

# ---------------------------
# Step 7: Chat function (with evaluation loop)
# ---------------------------
def chat(message, history):
    # Example of special behavior
    if "patent" in message.lower():
        system = system_prompt + "\n\nEverything in your reply must be in pig latin."
    else:
        system = system_prompt

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = response.choices[0].message.content

    # Run evaluation
    evaluation = evaluate(reply, message, history)
    
    if evaluation.is_acceptable:
        print("✅ Passed evaluation - returning reply")
    else:
        print("❌ Failed evaluation - retrying")
        print("Feedback:", evaluation.feedback)
        reply = rerun(reply, message, history, evaluation.feedback)
    
    return reply

# ---------------------------
# Step 8: Launch Gradio app
# ---------------------------
gr.ChatInterface(chat, type="messages").launch()