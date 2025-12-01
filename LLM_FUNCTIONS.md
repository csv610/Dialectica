# LLM Module - Philosophy Question Handling

The `llm.py` module now includes specialized functions for handling philosophical questions and generating Socratic-style inquiries.

## New Functions

### 1. `answer_question()`
**Purpose**: Get answers to philosophical questions with optional tone and custom system prompts.

**Signature**:
```python
def answer_question(
    question: str,
    tone: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
) -> str
```

**Parameters**:
- `question` (str): The philosophical question to answer
- `tone` (Optional[str]): Tone to use (e.g., "Analytical", "Socratic", "Dialectical", "Cynical")
- `model` (str): LLM model to use
- `temperature` (float): Sampling temperature (0.7 default)
- `max_tokens` (int): Maximum tokens in response
- `system_prompt` (Optional[str]): Custom system prompt (overrides default)

**Example**:
```python
from llm import answer_question

# Simple question
answer = answer_question("What is justice?")

# With specific tone
analytical_answer = answer_question(
    "What is justice?",
    tone="Analytical"
)

# With custom system prompt
answer = answer_question(
    "What is justice?",
    system_prompt="You are an expert in ancient Greek philosophy..."
)
```

### 2. `generate_socratic_questions()`
**Purpose**: Generate Socratic-style follow-up questions to encourage deeper reflection.

**Signature**:
```python
def generate_socratic_questions(
    question: str,
    num_questions: int = 5,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> List[str]
```

**Parameters**:
- `question` (str): The original philosophical question
- `num_questions` (int): Number of follow-up questions to generate (default: 5)
- `model` (str): LLM model to use
- `temperature` (float): Sampling temperature (0.8 default for more creativity)
- `max_tokens` (int): Maximum tokens to generate

**Returns**: List of Socratic follow-up questions

**Example**:
```python
from llm import generate_socratic_questions

questions = generate_socratic_questions(
    "What is consciousness?",
    num_questions=3
)

for q in questions:
    print(f"- {q}")
```

**Generated Questions Example**:
```
- Can we objectively measure consciousness, or is it inherently subjective?
- If consciousness is tied to physical brain states, what makes our subjective experience unique?
- How do we distinguish between simulated consciousness and genuine awareness?
```

### 3. `build_prompt()`
**Purpose**: Build formatted prompts with specific philosophical tones.

**Signature**:
```python
def build_prompt(
    tone: Optional[str],
    user_input: str,
    tone_descriptions: Optional[Dict[str, str]] = None
) -> str
```

**Parameters**:
- `tone` (Optional[str]): The tone to apply (e.g., "Analytical", "Socratic")
- `user_input` (str): The question or input to apply the tone to
- `tone_descriptions` (Optional[Dict]): Custom tone descriptions (uses TONES from philosophyqa by default)

**Returns**: Formatted prompt string

**Example**:
```python
from llm import build_prompt

prompt = build_prompt("Socratic", "What is the meaning of life?")
# Returns: "You are an expert in discussing philosophical questions. Your tone is: **Socratic**. Based on Socrates' method... [full description]"
```

## Available Tones

The following tones are available from `philosophyqa.TONES`:

- **Neutral**: Plain response without specific tone
- **Analytical**: Breaking down arguments, logic evaluation
- **Speculative**: Exploratory and imaginative thinking
- **Socratic**: Questioning and inquisitive method
- **Didactic**: Teaching and instructive approach
- **Dialectical**: Exchange of opposing viewpoints
- **Cynical**: Skeptical and critical perspective
- **Optimistic**: Hopeful and constructive outlook
- **Pessimistic**: Doubtful and negative outlook
- **Empirical**: Emphasis on experience and evidence
- **Existential**: Personal and reflective on human condition
- **Normative**: Focus on values and ethics
- **Absurdist**: Emphasis on life's contradictions and lack of meaning

## Usage Examples

### Complete Example: Exploring a Question

```python
from llm import answer_question, generate_socratic_questions, build_prompt

# Original question
question = "Is free will an illusion?"

# Get an analytical answer
analysis = answer_question(question, tone="Analytical")
print("ANALYSIS:")
print(analysis)

# Generate Socratic follow-ups
socratic_qs = generate_socratic_questions(question, num_questions=4)
print("\nSOCRATIC QUESTIONS:")
for q in socratic_qs:
    print(f"- {q}")

# Get a dialectical perspective
dialectical = answer_question(question, tone="Dialectical")
print("\nDIALECTICAL VIEW:")
print(dialectical)
```

### Integration with Streamlit App

```python
from llm import answer_question, generate_socratic_questions

# In philosopher_sl.py
def main():
    question = st.text_input("Enter a philosophical question:")
    tone = st.selectbox("Select tone:", TONES.keys())
    
    if question:
        # Get answer
        answer = answer_question(question, tone=tone)
        st.write("**Answer:**")
        st.write(answer)
        
        # Get Socratic questions
        socratic_qs = generate_socratic_questions(question, num_questions=3)
        st.write("**Explore Further:**")
        for q in socratic_qs:
            st.write(f"- {q}")
```

## Architecture Notes

- **Low-level utilities** in `llm.py`: General-purpose LLM functions
- **Philosophy-specific** functions: `answer_question()` and `generate_socratic_questions()`
- **Tone handling**: Integrated with `philosophyqa.TONES` for rich tone descriptions
- **Flexible system prompts**: All functions support custom system prompts for advanced use cases
