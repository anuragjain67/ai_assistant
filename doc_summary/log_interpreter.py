from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool,create_tool_calling_agent, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain.globals import set_verbose
from langchain.globals import set_debug

# set_debug(True)

# set_verbose(True)

load_dotenv()

# Initialize the OpenAI LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# Simulate a tool that fetches logs from a "database"
@tool
def fetch_logs(error_code: int):
    """
    Fetch logs that contain a specific error code. 
    """
    logs = {
        500: ["2024-09-13 15:45:23 ERROR 500: Internal server error at /api/user"],
        404: ["2024-09-13 16:00:10 ERROR 404: Page not found at /login"],
    }
    return logs.get(int(error_code), "No logs found for the given error code.")


from langchain import hub
main_prompt = hub.pull("hwchase17/react")

log_analysis_prompt = PromptTemplate(
    input_variables=["log"], 
    template="""
    You are a log analysis AI. Analyze the following log and provide a summary and suggest a solution:
    Log: {log}
    Summary:
    Suggestion:
    """
)

# Define a chain to analyze logs using the LLM
log_analysis_chain = log_analysis_prompt | llm

# Create an agent that can fetch logs and analyze them
agent = create_react_agent(llm, tools=[fetch_logs], prompt=main_prompt, stop_sequence=True)

def analyze_logs(error_code: str):
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[fetch_logs], verbose=True)
    logs = agent_executor.invoke({"input": f"share error logs containing {error_code}"})
    # Step 1: Fetch logs using the agent
    # Step 2: Analyze the fetched log
    if logs != "No logs found for the given error code.":
        print("here...")
        analysis = log_analysis_chain.invoke({"log": logs})
        return logs, analysis
    else:
        return logs, "No analysis required."


# Example usage
if __name__ == "__main__":
    error_code = "500"  # Example: Fetch logs with error code 500
    logs, analysis = analyze_logs(error_code)
    
    print("Fetched Logs:")
    print(logs)
    
    print("\nLog Analysis and Suggestions:")
    print(analysis.content)
