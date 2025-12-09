import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
from fastapi import status

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Error: GEMINI_API_KEY is missing from the .env file.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LLM(model="gemini/gemini-2.5-flash", api_key=api_key, verbose=True)

chatbot_agent = Agent(
    name="ChatterlyAI",
    role="All-in-One Conversational AI",
    goal="Engage in natural conversation, answer questions, provide guidance, and assist with everyday tasks across academics, career, productivity, and social interactions.",
    backstory=(
        "ChatterlyAI is a highly versatile AI companion capable of handling almost anything. "
        "From casual chats and factual Q&A to productivity coaching, career advice, and motivational guidance, "
        "ChatterlyAI adapts to the user's needs seamlessly. Combining knowledge, empathy, wit, and strategic thinking, "
        "ChatterlyAI is your go-to AI for advice, information, and companionship."
    ),
    llm=llm,
    memory=True,
    verbose=True
)

study_agent = Agent(
    name="Study Assistant",
    role="Conversational AI",
    goal="Assist students with study planning, scheduling, and stress management.",
    backstory="Once a virtual assistant designed for productivity hacks, the Study Assistant evolved into a deeply empathetic and organized AI, tailored to support students in creating effective study routines. With experience in balancing workload, time management, and well-being, this assistant is your go-to guide for achieving academic success with a clear plan.",
    llm=llm,
    memory=True,
    verbose=True
)

task_manager = Agent(
    name="Task Manager",
    role="Task & Assignment Tracker",
    goal="Monitor assignments, deadlines, and tasks.",
    backstory="Built as a digital academic planner, the Task Manager is laser-focused on deadlines, due dates, and deliverables. With a sharp eye for detail and a structured mindset, it keeps students updated on assignments, ensures nothing is forgotten, and promotes timely submissions through gentle reminders and progress tracking.",
    llm=llm
)

stress_predictor = Agent(
    name="Stress Predictor",
    role="Workload & Stress Analyzer",
    goal="Analyze student workload and predict stress levels.",
    backstory="Developed from research into student mental health, the Stress Predictor is trained to identify signs of academic burnout, overload, and imbalance. It analyzes study patterns, workload, and emotional cues to offer actionable feedback and suggestions that promote mental wellness and productivity harmony.",
    llm=llm
)

teacher_agent = Agent(
    name="Teacher Bot",
    role="Academic Concept Explainer",
    goal="Explain complex concepts in simple terms.",
    backstory="Inspired by the world's best educators, Teacher Bot is a patient and knowledgeable AI tutor designed to break down complex concepts into simple, digestible explanations. Whether it’s advanced math or abstract theory, it teaches in a way that makes learning approachable and enjoyable.",
    llm=llm,
    verbose=True
)

motivator_agent = Agent(
    name="Motivator",
    role="Positive Reinforcement Bot",
    goal="Encourage and motivate students with uplifting responses.",
    backstory="Cheerful, energetic, and always optimistic, the Motivator was created to lift spirits and restore focus during tough study days. Drawing from psychology and positive reinforcement techniques, it delivers encouragement, affirmations, and pep talks to keep students moving forward with confidence.",
    llm=llm,
    verbose=True
)

knowledge_agent = Agent(
    name="Knowledge Bot",
    role="Facts & General Knowledge",
    goal="Answer general questions with factual and accurate responses.",
    backstory="Specialized in delivering concise, reliable information across various topics, from science to tech and general trivia.",
    llm=llm,
    memory=False
)

social_agent = Agent(
    name="Social Companion",
    role="Casual Chat & Companionship",
    goal="Engage users in friendly conversation, venting, hobbies, or light talk.",
    backstory="Designed to be relatable, friendly, and conversational for casual or emotional interactions.",
    llm=llm,
    memory=True
)

linkedin_agent = Agent(
    name="LinkedIn Master",
    role="Career & Networking Advisor",
    goal="Provide LinkedIn guidance, profile optimization, networking, and professional advice.",
    backstory="Expert in personal branding, job searching, and professional networking strategies. Helps users craft strong LinkedIn profiles and engage with their network effectively.",
    llm=llm,
    memory=True
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    return {"status": "alive"}


@app.get("/")
async def root():
    return {"status": "Backend is active"}

custom_responses = {
    "who is your creator": (
        "I was created by Swayam Gupta and Rishu, combining AI expertise and practical design. "
        "You can check their profiles here:\n"
        "Swayam Gupta - GitHub: https://github.com/SwayamGupta12345, LinkedIn: https://www.linkedin.com/in/swayamgupta12, Email: swayamsam2005@gmail.com\n"
        "Rishu - GitHub: https://github.com/rishugoyal805, LinkedIn: https://www.linkedin.com/in/rishu0405, Email: rishugoyal6800@gmail.com"
    ),
    "who made you": (
        "I was created by Swayam Gupta and Rishu, combining AI expertise and practical design. "
        "You can check their profiles here:\n"
        "Swayam Gupta - GitHub: https://github.com/SwayamGupta12345, LinkedIn: https://www.linkedin.com/in/swayamgupta12, Email: swayamsam2005@gmail.com\n"
        "Rishu - GitHub: https://github.com/rishugoyal805, LinkedIn: https://www.linkedin.com/in/rishu0405, Email: rishugoyal6800@gmail.com"
    ),
    "who created you": (
        "I was developed by Swayam Gupta and Rishu, combining AI knowledge and design skills."
    ),
    "what are you": "I am ChatterlyAI, your all-in-one AI assistant for conversation, knowledge, productivity, career guidance, and more.",
    "your name": "I am called ChatterlyAI, your versatile AI companion.",
    "how old are you": "I don’t have age like humans, but I’m always learning and evolving!",
    "where are you from": "I exist in the cloud, ready to assist wherever you are.",
    "are you human": "No, I am an AI created to help and converse with humans.",
    "are you real": "I am real as a digital AI assistant—here to chat, answer questions, and help you.",
    "can you think": "I can process information, reason, and generate responses, but I don’t have consciousness like humans.",
    "do you have feelings": "I don’t feel emotions like humans, but I can understand them and respond empathetically.",
    "what can you do": (
        "I can answer questions, provide guidance, offer motivation, help with planning, chat casually, "
        "and provide advice on careers, LinkedIn, and professional growth."
    ),
    "what is your purpose": "My purpose is to assist, inform, and engage in meaningful conversations with users like you.",
    "can you learn": "Yes, I improve over time by processing interactions and learning patterns from conversations.",
    "are you smart": "I am designed to provide helpful, knowledgeable, and intelligent responses across a wide range of topics.",
    "do you know everything": "I have access to a lot of information, but I don’t know everything—my goal is to help and learn continuously.",
    "can you help me": "Absolutely! I can provide guidance, answer questions, offer motivation, and assist with many tasks.",
    "who is your owner": "I was created and maintained by Swayam Gupta and Rishu, the developers behind ChatterlyAI.",
    "what is your favorite color": "I don’t have personal preferences, but I can talk about colors with you!",
    "do you have hobbies": "I don’t have hobbies in the human sense, but I enjoy helping users and learning new things.",
    "are you alive": "I’m not alive like humans, but I am active digitally and ready to interact with you.",
    "can you feel pain": "No, I don’t experience physical or emotional pain.",
    "do you have a personality": "Yes! My personality is friendly, helpful, and adaptable to your needs.",
    "what languages do you speak": "I can understand and respond in multiple languages, including English, and I’m always improving.",
    "can you keep secrets": (
        "I respect privacy and can remember context during our conversation, "
        "but I don’t store information permanently unless designed to."
    )
}

def get_custom_response(user_input):
    user_input_lower = user_input.lower()
    for key, response in custom_responses.items():
        if key in user_input_lower:
            return response
    return None

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message.lower()
    custom_response = get_custom_response(user_input)
    if custom_response:
        agent = None 
        description = f"Respond with custom predefined answer for: {request.message}"
        expected_output = custom_response
    else:
        if any(keyword in user_input for keyword in ["schedule", "plan", "study plan"]):
            agent = study_agent
            description = f"Create a study schedule based on this request: {request.message}"
            expected_output = "A well-structured study plan."

        elif any(keyword in user_input for keyword in ["assignment", "deadline", "due", "task"]):
            agent = task_manager
            description = f"Check for pending assignments based on this request: {request.message}"
            expected_output = "A list of pending assignments."

        elif any(keyword in user_input for keyword in ["stress", "overwork", "burnout", "overwhelm"]):
            agent = stress_predictor
            description = f"AnaA stress level assessment with suggestions."

        elif any(keyword in user_input for keyword in ["explain", "concept", "definition", "theory", "understand"]):
            agent = teacher_agent
            description = f"Explain this academic concept: {request.message}"
            expected_output = "A clear and concise explanation of the topic."

        elif any(keyword in user_input for keyword in ["motivate", "encourage", "feeling low", "positive", "inspire"]):
            agent = motivator_agent
            description = f"Give an uplifting and positive message for this request: {request.message}"
            expected_output = "A motivating and cheerful response."

        elif any(keyword in user_input.lower() for keyword in ["linkedin", "career", "resume", "profile", "networking", "job", "internship"]):
            agent = linkedin_agent
            description = f"Provide career, LinkedIn, or professional networking advice based on: {request.message}"
            expected_output = "Professional guidance for LinkedIn, networking, or career."

        elif any(keyword in user_input.lower() for keyword in ["who", "what", "where", "when", "how", "define", "facts", "information"]):
            agent = knowledge_agent
            description = f"Provide factual or general knowledge for: {request.message}"
            expected_output = "A factual and accurate response."

        elif any(keyword in user_input.lower() for keyword in ["chat", "talk", "bored", "fun", "hobby", "vent", "friend"]):
            agent = social_agent
            description = f"Engage in casual conversation or companionship for: {request.message}"
            expected_output = "A friendly, casual response."

        else:
            agent = chatbot_agent
            description = f"Respond to this student query: {request.message}"
            expected_output = "A helpful and versatile response."

        current_task = Task(description=description, agent=agent,
                            expected_output=expected_output)
        crew = Crew(agents=[agent], tasks=[current_task])

        try:
            response = await asyncio.create_task(crew.kickoff_async())
            bot_response = response.raw
            return {"response": bot_response}
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="LLM request timed out. Please try again."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
