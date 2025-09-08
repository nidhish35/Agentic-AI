# # Example usage of the OpenAI Agent SDK
# from agents import Agent, Runner, trace
# import asyncio
# # 1. Define your agent in one line
# agent = Agent(
#     name="Jokester", 
#     instructions="You are a joke teller", 
#     model="gpt-4o-mini"
# )
# # 2. Run it with context and tracing
# async def main():
#     with trace("Telling a joke"):
#         result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
#         print(result.final_output)

# The imports

from dotenv import load_dotenv
from agents import Agent, Runner, trace

# The usual starting point

load_dotenv(override=True)


# Make an agent with name, instructions, model

agent = Agent(name="Jokester", instructions="You are a joke teller", model="gpt-4o-mini")

# Run the joke with Runner.run(agent, prompt) then print final_output

import asyncio

async def main():
    with trace("Telling a joke"):
        result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
        print(result.final_output)

asyncio.run(main())