import os
from typing import Any, Dict, List

import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


@tool
def tavily_search(query: str) -> List[Dict[str, Any]]:
    """Search the web for current information and return relevant results."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("Missing TAVILY_API_KEY in environment variables.")

    response = requests.post(
        f"{os.getenv('TAVILY_API_URL', 'https://api.tavily.com')}/search",
        json={
            "query": query,
            "max_results": 5,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "topic": "general",
        },
        headers={
            "Authorization": f"Bearer {tavily_api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    return [
        {
            "title": result["title"],
            "url": result["url"],
            "content": result["content"],
        }
        for result in payload.get("results", [])
    ]


llm = ChatOpenAI(model="gpt-5")
tools = [tavily_search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Search for 3 job postings for an AI engineer using LangChain "
                        "in the Bay Area on LinkedIn and list their details."
                    ),
                }
            ]
        }
    )
    structured = result.get("structured_response")
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
