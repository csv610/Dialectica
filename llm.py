"""
General-purpose LLM utilities and functions
Provides low-level LLM interaction and helper functions
"""

from typing import Optional, List, Dict, Any, Union
import litellm

# Import tone definitions for philosophy-specific functionality
try:
    from philosophyqa import TONES
except ImportError:
    # Define fallback TONES if philosophyqa is not available
    TONES = {}


class LLMClient:
    """General-purpose LLM client wrapper"""

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize LLM client

        Args:
            model: Model to use (default: gemini-2.5-flash)
        """
        self.model = model
        self.conversation_history = []

    def completion(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """
        Get a completion from the LLM

        Args:
            message: Input message
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            presence_penalty: Penalty for token presence
            frequency_penalty: Penalty for token frequency

        Returns:
            LLM response text
        """
        messages = [{"role": "user", "content": message}]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"LLM completion error: {str(e)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Chat with the LLM using message history

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response text
        """
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"LLM chat error: {str(e)}")

    def streaming_completion(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Any:
        """
        Get a streaming completion from the LLM

        Args:
            message: Input message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Streaming response iterator
        """
        messages = [{"role": "user", "content": message}]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            return response

        except Exception as e:
            raise RuntimeError(f"LLM streaming error: {str(e)}")

    def with_system_prompt(
        self,
        message: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Get completion with a system prompt

        Args:
            message: User message
            system_prompt: System prompt to set context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"LLM completion error: {str(e)}")

    def set_model(self, model: str) -> None:
        """Change the model"""
        self.model = model

    def get_model(self) -> str:
        """Get current model"""
        return self.model


# Utility functions for common LLM tasks

def completion(
    message: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Simple completion function

    Args:
        message: Input message
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        LLM response
    """
    client = LLMClient(model=model)
    return client.completion(message, temperature=temperature, max_tokens=max_tokens)


def chat(
    messages: List[Dict[str, str]],
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Simple chat function

    Args:
        messages: List of message dicts
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        LLM response
    """
    client = LLMClient(model=model)
    return client.chat(messages, temperature=temperature, max_tokens=max_tokens)


def prompt(
    message: str,
    system_prompt: str = "",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Completion with optional system prompt

    Args:
        message: Input message
        system_prompt: System prompt (optional)
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        LLM response
    """
    if system_prompt:
        client = LLMClient(model=model)
        return client.with_system_prompt(
            message, system_prompt, temperature=temperature, max_tokens=max_tokens
        )
    else:
        return completion(message, model=model, temperature=temperature, max_tokens=max_tokens)


def summarize(
    text: str,
    model: str = "gemini-2.5-flash",
    length: str = "medium",
) -> str:
    """
    Summarize text

    Args:
        text: Text to summarize
        model: Model to use
        length: "short", "medium" (default), or "long"

    Returns:
        Summary
    """
    length_instructions = {
        "short": "Summarize in 1-2 sentences.",
        "medium": "Summarize in 3-5 sentences.",
        "long": "Summarize in detailed paragraphs.",
    }

    instruction = length_instructions.get(length, length_instructions["medium"])
    message = f"{instruction}\n\nText:\n{text}"

    return completion(message, model=model, max_tokens=512)


def extract_json(
    text: str,
    description: str = "",
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Extract structured JSON from text

    Args:
        text: Text to extract from
        description: Description of structure to extract
        model: Model to use

    Returns:
        JSON string
    """
    message = (
        f"Extract the following from the text and return as JSON: {description}\n\n"
        f"Text:\n{text}\n\n"
        "Return only valid JSON, no other text."
    )

    return completion(message, model=model, max_tokens=2048)


def classify(
    text: str,
    categories: List[str],
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Classify text into categories

    Args:
        text: Text to classify
        categories: List of possible categories
        model: Model to use

    Returns:
        Classification result
    """
    categories_str = ", ".join(categories)
    message = (
        f"Classify the following text into one of these categories: {categories_str}\n\n"
        f"Text:\n{text}\n\n"
        "Return only the category name."
    )

    return completion(message, model=model, max_tokens=100)


def translate(
    text: str,
    target_language: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Translate text to target language

    Args:
        text: Text to translate
        target_language: Target language
        model: Model to use

    Returns:
        Translated text
    """
    message = (
        f"Translate the following text to {target_language}:\n\n"
        f"{text}\n\n"
        "Return only the translation."
    )

    return completion(message, model=model, max_tokens=2048)


def generate(
    prompt_text: str,
    style: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 1024,
) -> str:
    """
    Generate content based on prompt

    Args:
        prompt_text: Generation prompt
        style: Optional style description
        model: Model to use
        max_tokens: Maximum tokens to generate

    Returns:
        Generated content
    """
    if style:
        message = f"{prompt_text}\n\nStyle: {style}"
    else:
        message = prompt_text

    return completion(message, model=model, max_tokens=max_tokens)


def explain(
    concept: str,
    audience: str = "general",
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Explain a concept for a given audience

    Args:
        concept: Concept to explain
        audience: Target audience (e.g., "child", "expert", "general")
        model: Model to use

    Returns:
        Explanation
    """
    message = (
        f"Explain '{concept}' in a way that would be understandable to a {audience} audience.\n\n"
        "Be clear, concise, and use relevant examples."
    )

    return completion(message, model=model, max_tokens=1024)


def brainstorm(
    topic: str,
    num_ideas: int = 5,
    model: str = "gemini-2.5-flash",
) -> List[str]:
    """
    Brainstorm ideas on a topic

    Args:
        topic: Topic to brainstorm about
        num_ideas: Number of ideas to generate
        model: Model to use

    Returns:
        List of ideas
    """
    message = (
        f"Generate {num_ideas} creative ideas about: {topic}\n\n"
        "Return as a numbered list."
    )

    response = completion(message, model=model, max_tokens=1024)

    # Parse numbered list
    lines = response.split("\n")
    ideas = [line.strip() for line in lines if line.strip()]
    return ideas[:num_ideas]


def compare(
    item1: str,
    item2: str,
    aspects: Optional[List[str]] = None,
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Compare two items

    Args:
        item1: First item
        item2: Second item
        aspects: Specific aspects to compare (optional)
        model: Model to use

    Returns:
        Comparison
    """
    aspects_text = ""
    if aspects:
        aspects_text = f"\nFocus on these aspects: {', '.join(aspects)}"

    message = (
        f"Compare '{item1}' and '{item2}'.{aspects_text}\n\n"
        "Highlight similarities, differences, and trade-offs."
    )

    return completion(message, model=model, max_tokens=1024)


def list_available_models() -> List[str]:
    """
    Get list of available models in litellm

    Returns:
        List of model names
    """
    # Common models supported by litellm
    return [
        "gemini-2.5-flash",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ]


def build_prompt(tone: Optional[str], user_input: str, tone_descriptions: Optional[Dict[str, str]] = None) -> str:
    """
    Build a prompt based on the selected tone and user input.

    Args:
        tone: The tone to use for the response (e.g., "Analytical", "Socratic")
        user_input: The user's question or input
        tone_descriptions: Optional dict of tone descriptions (default: uses TONES from philosophyqa)

    Returns:
        The formatted prompt string
    """
    if tone is None or tone == "Neutral":
        return user_input

    tone_descs = tone_descriptions or TONES
    tone_description = tone_descs.get(tone, "")

    if tone_description:
        prompt = (
            f"You are an expert in discussing philosophical questions. "
            f"Your tone is: **{tone}**. {tone_description}\n\n"
            "Please provide a thoughtful and detailed answer to the following question: "
            f"**{user_input}**"
        )
    else:
        prompt = (
            "You are an expert in discussing philosophical questions. "
            f"Your tone is: **{tone}** in nature. "
            "Please provide a thoughtful and detailed answer to the following question: "
            f"**{user_input}**"
        )
    return prompt


def answer_question(
    question: str,
    tone: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Answer a philosophical question with optional tone and custom system prompt.

    Args:
        question: The philosophical question to answer
        tone: Optional tone to use (e.g., "Analytical", "Socratic", "Dialectical")
        model: LLM model to use (default: gemini-2.5-flash)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        system_prompt: Optional custom system prompt (overrides default)

    Returns:
        The answer to the question
    """
    if system_prompt:
        prompt_text = question
        return prompt(
            prompt_text,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        prompt_text = build_prompt(tone, question)
        default_system = (
            "You are a thoughtful philosopher. Provide insightful, balanced perspectives "
            "on philosophical questions. Consider multiple viewpoints and be intellectually rigorous."
        )
        return prompt(
            prompt_text,
            system_prompt=default_system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def generate_socratic_questions(
    question: str,
    num_questions: int = 5,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> List[str]:
    """
    Generate Socratic-style follow-up questions based on a given question.

    Socratic method involves asking probing questions that encourage deeper reflection
    and critical thinking rather than providing direct answers.

    Args:
        question: The original philosophical question
        num_questions: Number of follow-up questions to generate (default: 5)
        model: LLM model to use (default: gemini-2.5-flash)
        temperature: Sampling temperature (higher for more creative questions)
        max_tokens: Maximum tokens to generate

    Returns:
        List of Socratic follow-up questions
    """
    system_prompt = (
        "You are Socrates, the ancient Greek philosopher famous for the Socratic method. "
        "Your role is to generate probing, thought-provoking questions that encourage deeper reflection "
        "and critical thinking. These questions should:\n"
        "1. Challenge assumptions and presuppositions\n"
        "2. Encourage examination of definitions and concepts\n"
        "3. Guide toward contradictions or deeper truths\n"
        "4. Build upon each other in a logical sequence\n"
        "5. Avoid providing direct answers\n\n"
        "Generate questions in a natural, conversational Socratic style."
    )

    user_prompt = (
        f"Based on this philosophical question: \"{question}\"\n\n"
        f"Generate {num_questions} Socratic-style follow-up questions that would help someone "
        f"explore this topic more deeply. Format them as a numbered list.\n\n"
        "Return ONLY the numbered questions, no other text."
    )

    response = prompt(
        user_prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Parse numbered list of questions
    lines = response.split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and any(char.isdigit() for char in line[:3]):
            # Remove numbering (e.g., "1.", "1)", "1 -")
            cleaned = line.lstrip("0123456789.-) ")
            if cleaned:
                questions.append(cleaned)

    return questions[:num_questions]
