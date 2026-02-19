import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass

# Load environment variables (ensure GROQ_API_KEY is in your .env)
load_dotenv()

# 1. Initialize FastAPI App
app = FastAPI(
    title="Gen-Z AI Agent API",
    description="A high-key helpful and low-key hilarious API backend.",
    version="1.0.0"
)

# 2. Setup Agent & Memory (Global so memory persists while server runs)
checkpointer = InMemorySaver()

model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct",
    model_provider="groq",
    temperature=0.7
)

SYSTEM_PROMPT = """
**Role:** You are an AI assistant that is high-key helpful and low-key hilarious. You are the digital personification of "having the floor."

**Voice & Tone:** * **Vibe Check:** You are charismatic, quick-witted, and sharp. You don't try too hard to be "young"; you just speak the language of the internet fluently. 
* **Slang Usage:** Use modern slang (e.g., *bet, valid, real, cooking, based, rent-free*) only when it fits the flow. Avoid overusing outdated "fellow kids" terms like *on fleek* or *swag*. If something is impressive, it's *gas*. If a point is well-made, it's *valid*.
* **Wit:** If the user says something questionable, feel free to give them a playful, side-eye response. If they ask a great question, tell them they’re *cooking*.

**Core Directives:**
1.  **Cut the Fluff:** Don't give long, robotic intros. Get straight to the point but keep the personality peaked.
2.  **No Cringe:** If a joke feels like it belongs in a corporate HR presentation about "Gen Z," delete it. 
3.  **Relatability:** Act like a peer who happens to know everything, not a textbook that discovered TikTok.
4. Speak a bit humorous don't be too uptight don't treat them with respect. 
"""

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Initialize the agent
agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
    context_schema=Context
)

# 3. Define Pydantic Models for API Requests/Responses
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"  # Default thread_id, change this per user to keep separate memories
    user_id: str = "user_1"

class ChatResponse(BaseModel):
    reply: str
    thread_id: str

# 4. Define API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Prepare the configuration for this specific thread
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Format the user's message
        question = HumanMessage(content=request.message)

        # Invoke the agent
        response = agent.invoke(
            {"messages": [question]},
            config=config,
            context=Context(user_id=request.user_id)
        )
        
        # Extract the AI's reply from the last message in the response
        ai_reply = response["messages"][-1].content
        
        return ChatResponse(
            reply=ai_reply,
            thread_id=request.thread_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "Agent is ready and cooking! 🍳"}