# AI Assistant 
It's an attempt to learn how to take advantage of LLM and create a better product. 

1. Doc summarizer (done)
User upload the document or preprocessed his docs which will be stored in vector database. Now whenever user is quering, first passing this query to vector db which is returning context using  using similarity search. Pass question and context to LLM with prompt which will return answer only from the context provided. Also have support for chat history so particular session becomes relevant for the user.

![Architecture Diagram](docs/assets/architecture_diagram.png "Architecture Diagram")

2. Voice Assistant (todo)
3. Search Engine (todo)


## Installation
To set up the AI Assistant, follow these steps:

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Obtain an API key from Google and copy to .env file
   ```bash
   cp env_sample .env
   ```
3. You can also convert existing docs locally without uploading using
   ```bash
   python ai/db.py
   ```
6. Launch the application:
   ```bash
   streamlit run app.py
   ```

## TechStack
1. LangChain
2. ChromaDB
2. StreamLit
3. LLM - Google Gemini


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=anuragjain67/ai_assistant&type=Date)](https://star-history.com/#anuragjain67/ai_assistant&Date)
