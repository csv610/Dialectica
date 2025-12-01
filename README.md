# Dialectica - Philosophy Q&A Platform

A comprehensive philosophical question-answering platform powered by LLM models, featuring Gemini 2.5 Flash as the default model via LiteLLM.

## Features

- **500+ Philosophical Questions**: Curated collection of thought-provoking philosophical questions
- **Multiple Interfaces**: CLI tools for different use cases
- **Flexible LLM Support**: Uses LiteLLM for easy model switching (Gemini, GPT-4, Claude, etc.)
- **Rich Philosophical Analysis**:
  - Answer questions with philosophical perspectives
  - Explore questions at different depths
  - Compare philosophical perspectives
  - Generate dialogues between philosophers
  - Analyze counterarguments

## Project Structure

```
dialectica/
├── philosophyqa_cli.py      # Main CLI tool for answering questions
├── philosopher_sl.py        # Streamlit web interface
├── llm_philosopher.py       # Core philosopher implementation
├── llm.py                   # Low-level LLM utilities
├── questions.json           # 500 philosophical questions
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dialectica
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
- For Gemini: Set `GOOGLE_API_KEY` environment variable
- For OpenAI: Set `OPENAI_API_KEY` environment variable
- For Anthropic: Set `ANTHROPIC_API_KEY` environment variable

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

### Command-Line Interface (philosophyqa_cli.py)

Answer philosophical questions from the command line:

```bash
# Answer a specific question by ID
python philosophyqa_cli.py -q 1

# Answer a random question
python philosophyqa_cli.py --random

# List available questions
python philosophyqa_cli.py --list 20

# Use a different model
python philosophyqa_cli.py -q 5 --model gpt-4

# Custom questions file
python philosophyqa_cli.py -q 1 -f /path/to/questions.json

# Show help
python philosophyqa_cli.py -h
```

### Streamlit Web Interface (philosopher_sl.py)

Run the interactive web interface:

```bash
streamlit run philosopher_sl.py
```

### Python API

Use the philosophy Q&A in your own code:

```python
from llm_philosopher import LLMPhilosopher, PhilosophyQuestions

# Load questions
questions = PhilosophyQuestions("questions.json")

# Create philosopher
philosopher = LLMPhilosopher(model="gemini-2.5-flash")

# Answer a question
question = questions.get_by_id(1)
answer = philosopher.answer_question(question["question"])
print(answer)

# Explore a question at different depths
brief = philosopher.explore_question(question["question"], depth="brief")
deep = philosopher.explore_question(question["question"], depth="deep")

# Compare perspectives
comparison = philosopher.compare_perspectives(
    question["question"],
    ["Stoicism", "Existentialism", "Pragmatism"]
)

# Generate dialogue
dialogue = philosopher.dialogue(
    question["question"],
    speaker1="Plato",
    speaker2="Aristotle"
)
```

### Low-Level LLM API

For general LLM tasks:

```python
from llm import completion, summarize, classify, brainstorm

# Simple completion
response = completion("What is the meaning of life?")

# Summarize text
summary = summarize("Long text here...", length="short")

# Classify text
category = classify("This text is about...", categories=["Science", "Philosophy", "Arts"])

# Brainstorm ideas
ideas = brainstorm("Ways to improve education", num_ideas=5)
```

## Question Database

The `questions.json` file contains 500 philosophical questions structured as:

```json
[
  {
    "id": 1,
    "question": "Is happiness just chemicals flowing through your brain or something more?"
  },
  ...
]
```

Questions cover topics including:
- Metaphysics and existence
- Ethics and morality
- Epistemology and knowledge
- Consciousness and mind
- Love and relationships
- Death and mortality
- Technology and society
- And many more...

## Models Supported

The platform supports any model via LiteLLM:

- **Google**: `gemini-2.5-flash` (default), `gemini-pro`, `gemini-1.5-pro`
- **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Mistral**: `mistral-large`, `mistral-medium`
- And many more...

Change model with `-m` flag or `model` parameter.

## Configuration

### Environment Variables

```bash
# API Keys
export GOOGLE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Optional: LiteLLM settings
export LITELLM_LOG="DEBUG"  # Enable logging
```

### Temperature & Sampling

Adjust generation behavior:

```python
philosopher = LLMPhilosopher()
answer = philosopher.answer_question(
    question,
    temperature=0.8  # Higher = more creative
)
```

## Development

### Project Structure

- **llm.py**: Low-level LLM client and utility functions
- **llm_philosopher.py**: Philosopher-specific implementations
- **philosophyqa_cli.py**: CLI interface
- **philosopher_sl.py**: Streamlit web interface

### Adding New Questions

Edit `questions.json` and add new question objects:

```json
{
  "id": 501,
  "question": "Your question here?"
}
```

## Requirements

- Python 3.8+
- litellm >= 1.0.0
- streamlit >= 1.0.0 (for web interface)
- python-dotenv (for environment management)

See `requirements.txt` for full list.

## Performance

- Average response time: 2-10 seconds (varies by model)
- Max tokens per response: 1024 (configurable)
- Questions supported: 500+

## Error Handling

The platform gracefully handles:
- Missing API keys
- Model unavailability
- Network errors
- Invalid question IDs

## Examples

### Example 1: Answer a Question

```bash
$ python philosophyqa_cli.py -q 1
Question #1: Is happiness just chemicals flowing through your brain or something more?

[LLM response...]
```

### Example 2: Random Question

```bash
$ python philosophyqa_cli.py --random
Question #347: Should animals have rights similar to human rights?

[LLM response...]
```

### Example 3: List Questions

```bash
$ python philosophyqa_cli.py --list 5
#1: Is happiness just chemicals flowing through your brain or something more?
#2: Can we really know everything?
#3: What is the meaning of a good life?
#4: Is there a God?
#5: What in life is truly objective and not subjective?
```

## Troubleshooting

### "Questions file not found"
Ensure `questions.json` is in the current directory or use `-f` flag:
```bash
python philosophyqa_cli.py -q 1 -f /path/to/questions.json
```

### "Error querying model"
Check that:
1. API key is set correctly
2. Model name is valid
3. You have internet connectivity
4. API quotas are not exceeded

### Slow responses
Try:
1. Using a faster model (e.g., `gemini-2.5-flash` instead of `gpt-4`)
2. Reducing `max_tokens`
3. Checking internet connection

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review error messages carefully

---

Built with ❤️ for philosophical exploration