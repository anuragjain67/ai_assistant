import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta
import plotly.express as px
from collections import defaultdict
import json
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser


def parse_whatsapp_chat(text):
    """Parse WhatsApp chat text into a structured format."""

    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s+(?:AM|PM))\s+-\s+([^:]+):\s*(.*)'
    messages = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        match = re.match(pattern, line)
        if match:
            date_str, sender, message = match.groups()
            try:
                # Handle 2-digit year format
                date_str = date_str.replace("\u202f", " ")
                format_str = "%m/%d/%y, %I:%M %p"
                date = datetime.strptime(date_str, format_str)
                
                messages.append({
                    'datetime': date,
                    'date': date.date(),
                    'time': date.time(),
                    'sender': sender.strip(),
                    'message': message.strip() if message else ""  # Handle empty messages
                })
            except ValueError as e:
                print(f"Error parsing date: {date_str} - {e}")
                continue
    
    return pd.DataFrame(messages) if messages else pd.DataFrame()


class ChunkResponse(BaseModel):
    topic: str
    summary: str
    decisions: list[str]
    actions: list[str]
    urgency: str

class MonthlySummaryResponse(BaseModel):
    major_issues: list[str]
    decisions: list[str]
    ongoing_concerns: list[str]
    completed_actions: list[str]

def setup_llm_chains(api_key):
    """Setup Langchain with Gemini for different analysis tasks."""
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.3
    )
    
    # Chain for analyzing conversation chunks
    chunk_analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze this conversation chunk from a society group chat.
        Extract and format the following information as JSON:
        {{
            "topic": "main topic discussed",
            "summary": "brief summary",
            "decisions": ["list of decisions made"],
            "actions": ["list of action items"],
            "urgency": "high/medium/low"
        }}

        Ensure only in Json format is returned.
        Conversation:
        {text}
        """
    )
    
    chunk_chain = LLMChain(
        llm=llm,
        prompt=chunk_analysis_prompt,
        output_key="analysis"
    )
    
    # Chain for monthly summaries
    monthly_summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Create a monthly summary of these society group chat conversations.
        Focus on:
        1. Major issues discussed
        2. Decisions made
        3. Ongoing concerns
        4. Completed actions

        Conversations:
        {text}
        
        Format the response as JSON:
        {{
            "major_issues": ["list of major issues"],
            "decisions": ["list of decisions"],
            "ongoing_concerns": ["list of concerns"],
            "completed_actions": ["list of completed items"]
        }}
        Ensure only JSON format is returned.
        """
    )
    
    monthly_chain = LLMChain(
        llm=llm,
        prompt=monthly_summary_prompt,
        output_key="summary"
    )
    
    # Setup map-reduce chain for large text summarization
    map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=True
    )
    
    return chunk_chain, monthly_chain, map_reduce_chain

def chunk_messages_by_topic(messages_df, chunk_size=20):
    """Group messages into conversation chunks based on time gaps and potential topic changes."""
    chunks = []
    current_chunk = []
    
    for i in range(len(messages_df)):
        current_msg = messages_df.iloc[i]
        
        new_topic_keywords = ['regarding', 'new issue', 'attention', 'notice', 
                            'update', 'announcement', 'urgent', 'important']
        
        should_start_new_chunk = (
            len(current_chunk) == 0 or
            len(current_chunk) >= chunk_size or
            (len(current_chunk) > 0 and 
             (current_msg['datetime'] - current_chunk[-1]['datetime']).total_seconds() > 10800) or
            any(keyword in current_msg['message'].lower() for keyword in new_topic_keywords)
        )
        
        if should_start_new_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
        
        current_chunk.append(current_msg.to_dict())
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def analyze_time_period(df, start_date, end_date, chunk_chain, map_reduce_chain):
    """Analyze messages within a specific time period using Langchain."""
    period_msgs = df[
        (df['datetime'] >= start_date) & 
        (df['datetime'] <= end_date)
    ].copy()
    
    if len(period_msgs) == 0:
        return None
    
    # Split into chunks
    chunks = chunk_messages_by_topic(period_msgs)
    
    # Analyze chunks
    summaries = []
    with st.spinner(f'Analyzing {len(chunks)} conversation chunks...'):
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            # Convert chunk to text
            chunk_text = "\n".join([
                f"{msg['datetime'].strftime('%Y-%m-%d %H:%M')} - {msg['sender']}: {msg['message']}"
                for msg in chunk
            ])
            
            # Analyze chunk
            try:
                result = chunk_chain.run(chunk_text)
                chunk_parser = JsonOutputParser(pydantic_object=ChunkResponse)
                summaries.append(chunk_parser.parse(result))
            except Exception as e:
                # show warning and disappear after 5 seconds
                st.warning(f"Error analyzing chunk {i+1}: {str(e)}")

            
            progress_bar.progress((i + 1) / len(chunks))
    
    return summaries

def main():
    st.title("Society WhatsApp Chat Analyzer (Langchain + Gemini)")
    st.write("Analyze large society group chats using Google's Gemini LLM.")
    
    # Google API Key input
    api_key = st.text_input("Enter your Google API Key", type="password")
    
    if api_key:
        try:
            # Setup Langchain
            chunk_chain, monthly_chain, map_reduce_chain = setup_llm_chains(api_key)
            
            uploaded_file = st.file_uploader("Choose a WhatsApp chat export file", type=['txt'])
            
            if uploaded_file:
                # Read and parse chat
                chat_text = uploaded_file.read().decode('utf-8')
                st.write(f"Parsing messages")
                df = parse_whatsapp_chat(chat_text)
                
                if len(df) == 0:
                    st.warning("No messages found in the chat file")
                if len(df) > 0:
                    st.write(f"Successfully parsed {len(df)} messages")
                    
                    # Basic statistics
                    st.subheader("Chat Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Messages", len(df))
                    with col2:
                        st.metric("Unique Participants", df['sender'].nunique())
                    with col3:
                        st.metric("Time Span", f"{df['date'].min()} to {df['date'].max()}")
                    
                    # Analysis Options
                    st.subheader("Analysis Options")
                    analysis_type = st.radio(
                        "Choose analysis period:",
                        ["Recent (Last 7 days)", "Custom Period", "Monthly Summary"]
                    )
                    
                    if analysis_type == "Recent (Last 7 days)":
                        end_date = df['datetime'].max()
                        start_date = end_date - timedelta(days=7)
                        summaries = analyze_time_period(df, start_date, end_date, chunk_chain, map_reduce_chain)
                        
                        if summaries:
                            st.subheader("Recent Issues and Discussions")
                            
                            # Group by urgency
                            urgent_issues = defaultdict(list)
                            for summary in summaries:
                                urgent_issues[summary['urgency']].append(summary)
                            
                            # Display urgent issues first
                            if urgent_issues['high']:
                                st.write("🚨 Urgent Issues:")
                                for summary in urgent_issues['high']:
                                    with st.expander(f"🔴 {summary['topic']}"):
                                        st.write(summary['summary'])
                                        if summary['actions']:
                                            st.write("Action Items:")
                                            for action in summary['actions']:
                                                st.write(f"- {action}")
                            
                            if urgent_issues['medium']:
                                st.write("⚠️ Important Discussions:")
                                for summary in urgent_issues['medium']:
                                    with st.expander(f"🟡 {summary['topic']}"):
                                        st.write(summary['summary'])
                            
                            if urgent_issues['low']:
                                st.write("📝 General Updates:")
                                for summary in urgent_issues['low']:
                                    with st.expander(f"⚪ {summary['topic']}"):
                                        st.write(summary['summary'])
                    
                    elif analysis_type == "Custom Period":
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.date_input("Start date", df['date'].min())
                        with col2:
                            end_date = st.date_input("End date", df['date'].max())
                        
                        if st.button("Analyze Period"):
                            start_datetime = datetime.combine(start_date, datetime.min.time())
                            end_datetime = datetime.combine(end_date, datetime.max.time())
                            summaries = analyze_time_period(df, start_datetime, end_datetime, chunk_chain, map_reduce_chain)
                            
                            if summaries:
                                st.subheader(f"Analysis: {start_date} to {end_date}")
                                for summary in summaries:
                                    with st.expander(f"{summary['topic']}"):
                                        st.write(summary['summary'])
                                        if summary['decisions']:
                                            st.write("Decisions Made:")
                                            for decision in summary['decisions']:
                                                st.write(f"- {decision}")
                    
                    elif analysis_type == "Monthly Summary":
                        df['month'] = df['datetime'].dt.to_period('M')
                        months = sorted(df['month'].unique())
                        selected_month = st.selectbox("Select month", months)
                        
                        if selected_month:
                            start_date = selected_month.start_time
                            end_date = selected_month.end_time
                            
                            # Get monthly messages
                            month_msgs = df[
                                (df['datetime'] >= start_date) & 
                                (df['datetime'] <= end_date)
                            ]
                            
                            # Create monthly summary using map-reduce chain
                            texts = [
                                Document(page_content=msg['message'])
                                for _, msg in month_msgs.iterrows()
                            ]
                            
                            try:
                                with st.spinner("Generating monthly summary..."):
                                    summary = monthly_chain.run("\n".join([doc.page_content for doc in texts]))
                                    monthly_parser = JsonOutputParser(pydantic_object=MonthlySummaryResponse)
                                    summary_data = monthly_parser.parse(summary)
                                    
                                    st.subheader(f"Monthly Summary: {selected_month}")
                                    
                                    st.write("🎯 Major Issues:")
                                    for issue in summary_data['major_issues']:
                                        st.write(f"- {issue}")
                                    
                                    st.write("✅ Decisions Made:")
                                    for decision in summary_data['decisions']:
                                        st.write(f"- {decision}")
                                    
                                    st.write("⚠️ Ongoing Concerns:")
                                    for concern in summary_data['ongoing_concerns']:
                                        st.write(f"- {concern}")
                                    
                                    st.write("✔️ Completed Actions:")
                                    for action in summary_data['completed_actions']:
                                        st.write(f"- {action}")
                            
                            except Exception as e:
                                st.error(f"Error generating monthly summary: {str(e)}")
                    
                    # Activity Visualization
                    st.subheader("Message Activity")
                    daily_messages = df.groupby('date').size().reset_index(name='count')
                    fig = px.line(daily_messages, x='date', y='count', 
                                title='Daily Message Count')
                    st.plotly_chart(fig)
                    
                    # Most Active Hours
                    st.subheader("Most Active Hours")
                    df['hour'] = df['datetime'].dt.hour
                    hourly_messages = df.groupby('hour').size().reset_index(name='count')
                    fig = px.bar(hourly_messages, x='hour', y='count',
                                title='Message Distribution by Hour')
                    st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Error setting up Langchain: {str(e)}")

if __name__ == "__main__":
    main()
