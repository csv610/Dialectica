#!/usr/bin/env python3
"""
Philosophy Q&A CLI - Philosophical question answering tool
Uses litellm for LLM provider abstraction and Gemini 2.5 Flash as default model
"""

import argparse
import sys
import random
from typing import Optional
from philosophyqa import PhilosophyQuestions
from llm import answer_question as answer_question_llm


class PhilosophyQACLI:
    """Command-line interface for philosophy question answering"""

    def __init__(self, questions_file: str, model: str = "gemini-2.5-flash"):
        """
        Initialize the Philosophy Q&A CLI

        Args:
            questions_file: Path to the questions.json file
            model: LLM model to use (default: gemini-2.5-flash)
        """
        self.model = model
        self.questions_obj = PhilosophyQuestions(questions_file)
        self.questions = self.questions_obj.get_all()

    def answer_question(self, question_text: str) -> str:
        """Get an answer to a philosophical question"""
        return answer_question_llm(
            question=question_text,
            model=self.model,
            temperature=0.7,
            max_tokens=1024,
        )

    def explore_question(self, question_id: int) -> str:
        """Explore a specific question by ID"""
        question = self.questions_obj.get_by_id(question_id)
        if not question:
            return f"Question with ID {question_id} not found."

        return self.answer_question(question["question"])

    def list_questions(self, start: int = 1, limit: int = 10) -> None:
        """List questions"""
        for q in self.questions_obj.get_range(start, limit):
            print(f"#{q['id']}: {q['question']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Philosophy Q&A CLI using LiteLLM and Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer specific question by ID
  python philosophyqa_cli.py -q 1

  # Answer a random question
  python philosophyqa_cli.py --random

  # List available questions
  python philosophyqa_cli.py --list 20

  # Use different model
  python philosophyqa_cli.py -q 5 --model gpt-4

  # Custom questions file
  python philosophyqa_cli.py -q 1 -f /path/to/questions.json
        """,
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="questions.json",
        help="Path to questions.json file (default: questions.json)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model to use (default: gemini-2.5-flash)",
    )

    parser.add_argument(
        "-q",
        "--question",
        type=int,
        metavar="ID",
        help="Answer specific question by ID",
    )

    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Answer a random question",
    )

    parser.add_argument(
        "-l",
        "--list",
        type=int,
        const=10,
        nargs="?",
        metavar="N",
        help="List first N questions (default: 10)",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="Philosophy Q&A CLI 1.0",
    )

    args = parser.parse_args()

    try:
        # Initialize CLI
        cli = PhilosophyQACLI(args.file, model=args.model)

        # Handle command-line options
        if args.question is not None:
            response = cli.explore_question(args.question)
            print(response)

        elif args.random:
            question = random.choice(cli.questions)
            print(f"Question #{question['id']}: {question['question']}\n")
            response = cli.answer_question(question["question"])
            print(response)

        elif args.list is not None:
            cli.list_questions(limit=args.list)

        else:
            parser.print_help()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
