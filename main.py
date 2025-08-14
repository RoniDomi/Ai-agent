from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

def main():
    model = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0 # Temperature determines randomness of LLM, 0 - no randomness
    )

    tools = []
    agent_executor = create_react_agent(model, tools)

    print("Hello, I am your AI agent! To stop chatting with me type 'quit'.")
    print("You can ask me to perform calculations, or simply chat with me!")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break

        print("\nAI Agent: ", end="")
        for chunk in agent_executor.stream(
                {'messages': [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                # Make sure if current response is from agent and if there is any message
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()