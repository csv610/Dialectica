import streamlit as st
import sys
import logging
import time
import datetime
import random
from philosophyqa import CLASSES, TONES, PhilosophyQuestions
from llm import build_prompt, LLMClient


@st.cache_resource
def get_llm_client(model_name, temperature, max_tokens):
    return LLMClient(model=model_name)


def generate_response(client, user_input, temperature=0.8, max_tokens=4096):
    try:
        start_time = time.time()
        response = client.completion(user_input, temperature=temperature, max_tokens=max_tokens)
        end_time = time.time()
        time_taken = end_time - start_time

        return {
            "response": response,
            "input": user_input,
            "time": time_taken,
        }
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        st.error("Sorry, I couldn't generate a response.")
        return {
            "response": "Sorry, I couldn't generate a response.",
            "input": user_input,
            "time": 0,
        }


logging.basicConfig(filename='llamachat.log', filemode='w', level=logging.INFO)


@st.cache_resource
def load_philosophy_questions(file_path="questions.json"):
    try:
        return PhilosophyQuestions(file_path)
    except FileNotFoundError:
        st.error(f"Questions file not found: {file_path}")
        return None


def display_chat_history():
    for entry in reversed(st.session_state.chat_history):
        st.write(f"**User**: {entry['input']}")
        st.write(f"**Assistant**: {entry['response']}")
        st.write(f"**Word Count**: {len(entry['response'])}")
        st.write(f"**Time Taken**: {entry['time']:.2f}s")
        st.divider()


def config_panel():
    st.sidebar.header("Dialectica")

    model_options = [
        "ollama/llama3.2",
        "ollama/llama3.1",
        "ollama/gemma3",
        "ollama/gemma3:12b",
        "gemini-2.5-flash",
        "gpt-4",
        "gpt-4-turbo",
        "claude-3-sonnet",
    ]
    model_name = st.sidebar.selectbox("Select Model", model_options, index=0)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.8, 0.1)
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=8192, value=4096, step=100)

    selected_tone = st.sidebar.selectbox("Select Tone", TONES.keys())

    if st.sidebar.button("Explain Tone"):
        st.sidebar.write(TONES[selected_tone])

    if st.sidebar.button("Clear History"):
        st.session_state.chat_history = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    client = get_llm_client(model_name, temperature, max_tokens)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.write(f"Time: {current_time}")

    return client, selected_tone, temperature, max_tokens


def get_new_question(questions_obj):
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if st.button("Get Random Question", key="random_question_button"):
        if questions_obj:
            all_questions = questions_obj.get_all()
            random_question = random.choice(all_questions)["question"]
            st.session_state.user_input = random_question

    question = st.text_input("Enter your input:", value="", key="user_input")
    return question


def main():
    st.set_page_config(layout="wide")

    client, tone, temperature, max_tokens = config_panel()

    questions_obj = load_philosophy_questions()

    question = get_new_question(questions_obj)

    class_question = f"Given the question: {question}, what is the most accurate classification? Choose from the following categories: {', '.join(CLASSES)}."

    if question:
        with st.spinner("Generating response..."):
            result = generate_response(client, class_question, temperature, max_tokens)
            class_result = result['response']
            prompt = build_prompt(tone, question)
            result = generate_response(client, prompt, temperature, max_tokens)
            result['class'] = class_result
        st.session_state.chat_history.append(result)

    display_chat_history()


if __name__ == "__main__":
    main()
