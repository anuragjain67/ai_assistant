# Setup Instructions

Follow the steps below to set up the environment and run the agent.

## Step 1: Set Up the Environment
Follow the quickstart guide from the official documentation:  
[Quickstart Guide](https://docs.browser-use.com/quickstart)

## Step 2: Activate the Virtual Environment
Run the following command to activate your virtual environment:

```bash
source .venv/bin/activate
```

## Step 3: (Optional) Run Chrome in debugging mode
To enable remote debugging in Chrome, execute the following command:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --remote-allow-origins=*
```

## Step 4: Configure LLM (Language Model)
Grab your Gemini API key or modify the llm.py file to use a different LLM (Language Model).

## Step 5: Create a .env file
Create a .env file in the project directory, then add your Gemini API Key and any other necessary environment variables:

```bash
GOOGLE_API_KEY=<your-gemini-api-key>
```

## Step 6: Run the agent
Execute the following command to run the agent with a specific task:

```bash
python agent.py --task=example_tasks/task1.txt
```