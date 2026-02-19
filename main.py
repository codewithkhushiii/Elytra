#import libraries
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

#function call to load .env
load_dotenv()

#model = llama-4-scout
#chat model initialize
#temperature decides the creativity of model
model = init_chat_model("meta-llama/llama-4-scout-17b-16e-instruct",
                        model_provider = "groq",
                        temperature = 0.7)

#agent initialize
agent = create_agent(model = model)

print("🤖 Agent is ready! Type 'exit' to stop.\n")

# conversation loop
while True:
    
    # take user input
    user_input = input("You: ")
    
    # exit condition
    if user_input.lower() == "exit":
        print("Agent: Goodbye 👋")
        break

    question = HumanMessage(content = user_input )

    response = agent.invoke(
        {"messages":[question]}
    )
    print("Agent:",response["messages"][-1].content)