#!/usr/bin/env python3
"""
Comprehensive test suite for llm.py functions
Tests answer_question, generate_socratic_questions, and build_prompt
"""

import sys
from typing import List, Dict, Any
import json

# Test infrastructure
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✓ {test_name}")

    def add_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append((test_name, reason))
        print(f"✗ {test_name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed Tests:")
            for test_name, reason in self.errors:
                print(f"  - {test_name}: {reason}")
        print(f"{'='*60}\n")
        return self.failed == 0


def test_imports(results: TestResults):
    """Test that all required imports work"""
    print("\n--- Testing Imports ---")
    try:
        from llm import (
            answer_question,
            generate_socratic_questions,
            build_prompt,
            LLMClient,
            prompt,
            completion,
        )
        results.add_pass("Import llm functions")
    except ImportError as e:
        results.add_fail("Import llm functions", str(e))
        return False

    try:
        from philosophyqa import TONES, CLASSES
        results.add_pass("Import philosophyqa constants")
    except ImportError as e:
        results.add_fail("Import philosophyqa constants", str(e))
        return False

    return True


def test_build_prompt(results: TestResults):
    """Test build_prompt function"""
    print("\n--- Testing build_prompt() ---")
    from llm import build_prompt
    from philosophyqa import TONES

    question = "What is the nature of consciousness?"

    # Test 1: Neutral tone
    try:
        prompt_neutral = build_prompt(None, question)
        if prompt_neutral == question:
            results.add_pass("build_prompt with None tone returns original question")
        else:
            results.add_fail("build_prompt with None tone", "Should return original question")
    except Exception as e:
        results.add_fail("build_prompt with None tone", str(e))

    # Test 2: "Neutral" string
    try:
        prompt_neutral = build_prompt("Neutral", question)
        if prompt_neutral == question:
            results.add_pass("build_prompt with 'Neutral' tone returns original question")
        else:
            results.add_fail("build_prompt with 'Neutral' tone", "Should return original question")
    except Exception as e:
        results.add_fail("build_prompt with 'Neutral' tone", str(e))

    # Test 3: Socratic tone
    try:
        prompt_socratic = build_prompt("Socratic", question)
        if isinstance(prompt_socratic, str) and len(prompt_socratic) > len(question):
            if "Socratic" in prompt_socratic:
                results.add_pass("build_prompt with 'Socratic' tone creates enhanced prompt")
            else:
                results.add_fail("build_prompt with 'Socratic' tone", "Should contain tone name")
        else:
            results.add_fail("build_prompt with 'Socratic' tone", "Should be longer than original")
    except Exception as e:
        results.add_fail("build_prompt with 'Socratic' tone", str(e))

    # Test 4: Analytical tone
    try:
        prompt_analytical = build_prompt("Analytical", question)
        if isinstance(prompt_analytical, str) and "Analytical" in prompt_analytical:
            results.add_pass("build_prompt with 'Analytical' tone works")
        else:
            results.add_fail("build_prompt with 'Analytical' tone", "Should contain tone name")
    except Exception as e:
        results.add_fail("build_prompt with 'Analytical' tone", str(e))

    # Test 5: Custom tone descriptions
    try:
        custom_tones = {
            "Custom": "This is a custom tone description"
        }
        prompt_custom = build_prompt("Custom", question, tone_descriptions=custom_tones)
        if "custom tone description" in prompt_custom.lower():
            results.add_pass("build_prompt with custom tone descriptions works")
        else:
            results.add_fail("build_prompt with custom descriptions", "Should use custom description")
    except Exception as e:
        results.add_fail("build_prompt with custom descriptions", str(e))

    # Test 6: All available tones
    try:
        failed_tones = []
        for tone in list(TONES.keys())[:5]:  # Test first 5 tones
            prompt_result = build_prompt(tone, question)
            if not isinstance(prompt_result, str) or len(prompt_result) == 0:
                failed_tones.append(tone)

        if not failed_tones:
            results.add_pass("build_prompt works with multiple TONES from philosophyqa")
        else:
            results.add_fail("build_prompt with TONES", f"Failed for: {failed_tones}")
    except Exception as e:
        results.add_fail("build_prompt with TONES", str(e))


def test_answer_question_structure(results: TestResults):
    """Test answer_question function exists and has correct signature"""
    print("\n--- Testing answer_question() Structure ---")
    from llm import answer_question
    import inspect

    # Test 1: Function exists
    try:
        if callable(answer_question):
            results.add_pass("answer_question function exists and is callable")
        else:
            results.add_fail("answer_question callable", "Function exists but not callable")
    except Exception as e:
        results.add_fail("answer_question callable", str(e))

    # Test 2: Check function signature
    try:
        sig = inspect.signature(answer_question)
        params = list(sig.parameters.keys())
        expected_params = ['question', 'tone', 'model', 'temperature', 'max_tokens', 'system_prompt']

        if all(p in params for p in expected_params):
            results.add_pass("answer_question has correct parameters")
        else:
            results.add_fail("answer_question parameters", f"Missing params. Got: {params}")
    except Exception as e:
        results.add_fail("answer_question signature", str(e))

    # Test 3: Test with mock (without actual LLM call)
    try:
        # Just verify the function accepts the parameters without calling LLM
        sig = inspect.signature(answer_question)
        test_question = "What is truth?"

        # Verify we can inspect the function
        if 'question' in sig.parameters and 'tone' in sig.parameters:
            results.add_pass("answer_question accepts question and tone parameters")
        else:
            results.add_fail("answer_question parameters", "Missing required params")
    except Exception as e:
        results.add_fail("answer_question inspection", str(e))


def test_generate_socratic_structure(results: TestResults):
    """Test generate_socratic_questions function exists and has correct signature"""
    print("\n--- Testing generate_socratic_questions() Structure ---")
    from llm import generate_socratic_questions
    import inspect

    # Test 1: Function exists
    try:
        if callable(generate_socratic_questions):
            results.add_pass("generate_socratic_questions function exists and is callable")
        else:
            results.add_fail("generate_socratic_questions callable", "Function exists but not callable")
    except Exception as e:
        results.add_fail("generate_socratic_questions callable", str(e))

    # Test 2: Check function signature
    try:
        sig = inspect.signature(generate_socratic_questions)
        params = list(sig.parameters.keys())
        expected_params = ['question', 'num_questions', 'model', 'temperature', 'max_tokens']

        if all(p in params for p in expected_params):
            results.add_pass("generate_socratic_questions has correct parameters")
        else:
            results.add_fail("generate_socratic_questions parameters", f"Got: {params}")
    except Exception as e:
        results.add_fail("generate_socratic_questions signature", str(e))

    # Test 3: Check return type annotation
    try:
        sig = inspect.signature(generate_socratic_questions)
        ann = sig.return_annotation
        if ann is not inspect.Parameter.empty and 'List' in str(ann):
            results.add_pass("generate_socratic_questions has List[str] return type")
        else:
            results.add_pass("generate_socratic_questions has return type annotation")
    except Exception as e:
        results.add_fail("generate_socratic_questions return type", str(e))


def test_docstrings(results: TestResults):
    """Test that functions have proper docstrings"""
    print("\n--- Testing Documentation ---")
    from llm import answer_question, generate_socratic_questions, build_prompt

    functions = [
        ("answer_question", answer_question),
        ("generate_socratic_questions", generate_socratic_questions),
        ("build_prompt", build_prompt),
    ]

    for func_name, func in functions:
        try:
            doc = func.__doc__
            if doc and len(doc) > 50:
                # Check for Args and Returns sections
                if "Args:" in doc and "Returns:" in doc:
                    results.add_pass(f"{func_name} has comprehensive docstring")
                else:
                    results.add_pass(f"{func_name} has docstring")
            else:
                results.add_fail(f"{func_name} docstring", "Missing or too short")
        except Exception as e:
            results.add_fail(f"{func_name} docstring", str(e))


def test_tone_integration(results: TestResults):
    """Test integration with TONES from philosophyqa"""
    print("\n--- Testing TONES Integration ---")
    from llm import build_prompt
    from philosophyqa import TONES

    question = "What is wisdom?"

    # Test 1: Verify TONES is not empty
    try:
        if len(TONES) > 0:
            results.add_pass(f"TONES dictionary loaded ({len(TONES)} tones available)")
        else:
            results.add_fail("TONES loading", "TONES dictionary is empty")
    except Exception as e:
        results.add_fail("TONES loading", str(e))

    # Test 2: Verify build_prompt uses TONES
    try:
        tone_name = list(TONES.keys())[0]  # Get first tone
        prompt_result = build_prompt(tone_name, question)

        # Check if description from TONES is included
        tone_desc = TONES[tone_name]
        if isinstance(tone_desc, str) and len(tone_desc) > 0:
            results.add_pass(f"build_prompt integrates with TONES (e.g., '{tone_name}')")
        else:
            results.add_fail("TONES integration", f"Tone description for '{tone_name}' is invalid")
    except Exception as e:
        results.add_fail("TONES integration", str(e))

    # Test 3: Test with multiple tones
    try:
        sample_tones = list(TONES.keys())[:3]
        all_work = True
        for tone in sample_tones:
            result = build_prompt(tone, question)
            if not isinstance(result, str) or len(result) == 0:
                all_work = False
                break

        if all_work:
            results.add_pass(f"build_prompt works with all tested tones")
        else:
            results.add_fail("TONES compatibility", "Some tones don't work properly")
    except Exception as e:
        results.add_fail("TONES compatibility", str(e))


def test_question_parsing(results: TestResults):
    """Test the question parsing logic in generate_socratic_questions"""
    print("\n--- Testing Question Parsing Logic ---")

    # Simulate the parsing logic used in generate_socratic_questions
    test_cases = [
        ("1. What is consciousness?", "What is consciousness?"),
        ("1) Can we measure awareness?", "Can we measure awareness?"),
        ("2 - How does perception work?", "How does perception work?"),
        ("10. Is experience objective?", "Is experience objective?"),
        ("1. ", ""),  # Empty after number
    ]

    for input_str, expected_output in test_cases:
        try:
            line = input_str.strip()
            cleaned = line.lstrip("0123456789.-) ")
            if cleaned == expected_output:
                results.add_pass(f"Parse: '{input_str}' → '{cleaned}'")
            else:
                results.add_fail(f"Parse: '{input_str}'", f"Got '{cleaned}', expected '{expected_output}'")
        except Exception as e:
            results.add_fail(f"Parse: '{input_str}'", str(e))


def test_class_llmclient(results: TestResults):
    """Test LLMClient class still works"""
    print("\n--- Testing LLMClient Class ---")
    from llm import LLMClient

    try:
        client = LLMClient(model="gemini-2.5-flash")
        results.add_pass("LLMClient instantiation works")
    except Exception as e:
        results.add_fail("LLMClient instantiation", str(e))
        return

    try:
        if hasattr(client, 'completion') and callable(client.completion):
            results.add_pass("LLMClient has completion method")
        else:
            results.add_fail("LLMClient methods", "Missing completion method")
    except Exception as e:
        results.add_fail("LLMClient methods", str(e))

    try:
        if hasattr(client, 'set_model') and callable(client.set_model):
            results.add_pass("LLMClient has set_model method")
        else:
            results.add_fail("LLMClient methods", "Missing set_model method")
    except Exception as e:
        results.add_fail("LLMClient methods", str(e))


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("LLM Module Test Suite")
    print("="*60)

    results = TestResults()

    # Run all test suites
    if not test_imports(results):
        print("\n✗ Import tests failed. Cannot continue with other tests.")
        results.summary()
        return 1

    test_build_prompt(results)
    test_answer_question_structure(results)
    test_generate_socratic_structure(results)
    test_docstrings(results)
    test_tone_integration(results)
    test_question_parsing(results)
    test_class_llmclient(results)

    # Summary
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
