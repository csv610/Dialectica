"""
Common code for LLM-based philosopher implementations
Provides shared utilities and base classes for philosophical Q&A
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import litellm


CLASSES = [
    "Metaphysics",
    "Epistemology",
    "Ethics",
    "Aesthetics",
    "Logic and Reasoning",
    "Political Philosophy",
    "Philosophy of Science",
    "Philosophy of Religion",
    "Philosophy of Mind",
    "Philosophy of Language",
    "Philosophy of Time",
]

TONES = {
    "Neutral": "Neutral",
    "Analytical": "This tone focuses on breaking down arguments into smaller parts, evaluating their logic, and ensuring clarity. It's often precise, critical, and detailed.",
    "Speculative": "A more exploratory and imaginative tone that considers possibilities, hypotheses, or abstract ideas that go beyond concrete facts.",
    "Socratic": "Based on Socrates' method, this tone is questioning and inquisitive, often encouraging the other person to reflect on their beliefs and assumptions.",
    "Didactic": "A teaching or instructive tone, where the speaker aims to impart knowledge or explain complex concepts in a clear, authoritative manner.",
    "Dialectical": "This tone is characterized by an exchange of ideas between opposing viewpoints, with the aim of arriving at a higher truth through reasoned dialogue.",
    "Cynical": "A more skeptical and sometimes dismissive tone, often critical of established ideas or institutions, questioning motives, and highlighting flaws.",
    "Optimistic": "A hopeful and constructive tone that focuses on positive possibilities, growth, or ideal outcomes in philosophical exploration.",
    "Pessimistic": "This tone reflects a more doubtful or negative outlook on human nature, existence, or philosophical concepts, often focusing on limitations and problems.",
    "Empirical": "A tone that emphasizes experience, observation, and evidence, often associated with philosophers who stress the importance of real-world data and facts in their reasoning.",
    "Existential": "A deeply personal and reflective tone that deals with individual experience, meaning, and the human condition, often touching on themes like freedom, isolation, and choice.",
    "Normative": "This tone deals with values, ethics, and how things should be. It often involves moral judgments or considerations of right and wrong.",
    "Absurdist": "A tone that reflects on the inherent contradictions or lack of meaning in life, often humorously or paradoxically, following in the tradition of philosophers like Camus.",
}


class PhilosophyQuestions:
    """Manages philosophical questions from JSON file"""

    def __init__(self, questions_file: str = "questions.json"):
        """
        Initialize questions loader

        Args:
            questions_file: Path to questions.json file
        """
        self.questions = self._load_questions(questions_file)

    def _load_questions(self, questions_file: str) -> List[Dict[str, Any]]:
        """Load questions from JSON file"""
        file_path = Path(questions_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")

        with open(file_path, "r") as f:
            return json.load(f)

    def get_by_id(self, question_id: int) -> Optional[Dict[str, Any]]:
        """Get a question by its ID"""
        for q in self.questions:
            if q["id"] == question_id:
                return q
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all questions"""
        return self.questions

    def get_range(self, start: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
        """Get a range of questions"""
        return self.questions[start - 1 : start - 1 + limit]

    def count(self) -> int:
        """Get total number of questions"""
        return len(self.questions)


class LLMPhilosopher:
    """Base class for LLM-based philosopher"""

    SYSTEM_PROMPT = (
        "You are a thoughtful philosopher. Provide insightful, balanced perspectives "
        "on philosophical questions. Consider multiple viewpoints and be intellectually rigorous."
    )

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize LLM Philosopher

        Args:
            model: LLM model to use (default: gemini-2.5-flash)
        """
        self.model = model

    def query(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Query the LLM

        Args:
            message: User message
            system_prompt: System prompt (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response
        """
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        else:
            messages = [{"role": "user", "content": message}]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error querying model: {str(e)}"

    def answer_question(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Answer a philosophical question

        Args:
            question: The philosophical question
            system_prompt: Custom system prompt (uses default if None)
            temperature: Temperature for generation

        Returns:
            Philosophical answer
        """
        system = system_prompt or self.SYSTEM_PROMPT
        return self.query(f"Question: {question}", system_prompt=system, temperature=temperature)

    def explore_question(self, question: str, depth: str = "balanced") -> str:
        """
        Explore a question with different depth levels

        Args:
            question: The philosophical question
            depth: "brief", "balanced" (default), or "deep"

        Returns:
            Philosophical exploration
        """
        depth_prompts = {
            "brief": "Provide a concise answer (2-3 sentences)",
            "balanced": "Provide a balanced, thoughtful perspective",
            "deep": "Provide a deep, comprehensive analysis with multiple perspectives",
        }

        depth_instruction = depth_prompts.get(depth, depth_prompts["balanced"])
        custom_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n{depth_instruction}"
        )

        return self.answer_question(question, system_prompt=custom_prompt)

    def compare_perspectives(self, question: str, perspectives: List[str]) -> str:
        """
        Compare different philosophical perspectives on a question

        Args:
            question: The philosophical question
            perspectives: List of perspective names (e.g., ["Stoicism", "Existentialism"])

        Returns:
            Comparison of perspectives
        """
        perspectives_text = ", ".join(perspectives)
        prompt = (
            f"Compare the following perspectives on this question: {perspectives_text}\n\n"
            f"Question: {question}"
        )

        system_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            "Compare and contrast the perspectives, highlighting key differences and similarities."
        )

        return self.query(prompt, system_prompt=system_prompt)

    def dialogue(self, question: str, speaker1: str = "Socrates", speaker2: str = "Aristotle") -> str:
        """
        Generate a philosophical dialogue between two thinkers

        Args:
            question: The philosophical question
            speaker1: First philosopher name
            speaker2: Second philosopher name

        Returns:
            Philosophical dialogue
        """
        prompt = (
            f"Create a dialogue between {speaker1} and {speaker2} discussing this question:\n\n"
            f"Question: {question}\n\n"
            f"Format the dialogue with clear labels for each speaker."
        )

        system_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Write in the distinctive styles of {speaker1} and {speaker2}. "
            "Make the dialogue engaging and historically accurate to their philosophies."
        )

        return self.query(prompt, system_prompt=system_prompt)

    def analyze_counterargument(self, question: str, position: str) -> str:
        """
        Analyze counterarguments to a philosophical position

        Args:
            question: The philosophical question
            position: A philosophical position on the question

        Returns:
            Analysis of counterarguments
        """
        prompt = (
            f"Question: {question}\n\n"
            f"Position: {position}\n\n"
            "What are the strongest counterarguments to this position? "
            "Analyze each one critically."
        )

        system_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            "Provide rigorous logical analysis of counterarguments. "
            "Be fair to opposing views while maintaining intellectual rigor."
        )

        return self.query(prompt, system_prompt=system_prompt)

    def set_model(self, model: str) -> None:
        """Change the LLM model"""
        self.model = model
