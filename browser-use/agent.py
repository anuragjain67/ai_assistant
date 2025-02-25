import asyncio
from browser_use import Agent, Browser, BrowserConfig


from dotenv import load_dotenv
load_dotenv()

from llm import llm

config = BrowserConfig(
    chrome_instance_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)

browser = Browser(config=config)


async def main(task):
    task = open(task).read()
    print(task)
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    # get filename from input and run main
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="task file")
    args = parser.parse_args()
    if args.task:
        asyncio.run(main(task=args.task))
    else:
        print("Please provide a task file")
