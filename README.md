# RehabOS

Multi-agent clinical reasoning system for Physical Therapy (PT), Occupational Therapy (OT), and Speech-Language Pathology (SLP).

## Overview

RehabOS is an AI-powered clinical decision support system that uses specialized agents to assist rehabilitation professionals with:

- **Safety Screening**: Red flag identification and triage
- **Diagnosis Classification**: ICD-10 coding and clinical reasoning
- **Evidence Retrieval**: Clinical practice guideline lookup and literature search
- **Treatment Planning**: SMART goals, interventions, and home exercise programs
- **Outcome Measures**: Standardized assessment recommendations
- **Documentation**: Clinical note generation
- **Quality Assurance**: Plan review and critique

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Orchestrator                          │
│  Coordinates agent pipeline and manages clinical workflow    │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
┌─────────┐  ┌─────────────────────────────┐  ┌─────────────┐
│Red Flag │  │     Parallel Agents         │  │     QA      │
│ Agent   │  │  ┌─────────┐ ┌──────────┐  │  │   Agent     │
│(Safety) │  │  │Diagnosis│ │ Evidence │  │  │  (Review)   │
└────┬────┘  │  └────┬────┘ └────┬─────┘  │  └─────────────┘
     │       │       └─────┬─────┘        │
     │       │             ▼              │
     │       │  ┌─────────────────────┐   │
     │       │  │    Plan Agent       │   │
     │       │  │ (Treatment Plan)    │   │
     │       │  └──────────┬──────────┘   │
     │       │             ▼              │
     │       │  ┌─────────────────────┐   │
     │       │  │   Outcome Agent     │   │
     │       │  │ (Measures)          │   │
     │       └──┴─────────────────────┴───┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Router                              │
│         Local LLM (DGX Spark) ──► Claude Fallback           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rehab-os.git
cd rehab-os

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Edit `.env` file with your settings:

```bash
# Local LLM (DGX Spark) - OpenAI-compatible endpoint
LOCAL_LLM_BASE_URL=http://your-dgx-server:8000/v1
LOCAL_LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct

# Anthropic Claude (fallback)
ANTHROPIC_API_KEY=your-api-key

# Optional: PubMed API for enhanced evidence retrieval
PUBMED_API_KEY=your-ncbi-key
PUBMED_EMAIL=your-email@example.com
```

## Usage

### CLI

```bash
# Run a consultation
rehab-os consult "68yo male s/p L TKA POD 2, evaluate for PT" --discipline PT --setting inpatient

# With patient context file
rehab-os consult "Evaluate for PT" --patient patient.json

# Search for evidence
rehab-os evidence "rotator cuff conservative management" --condition "rotator cuff tear"

# Initialize knowledge base with sample guidelines
rehab-os init-kb --samples

# Start the API server
rehab-os serve --port 8080

# Check system health
rehab-os health
```

### REST API

```bash
# Start server
rehab-os serve

# Full consultation
curl -X POST http://localhost:8080/api/v1/consult \
  -H "Content-Type: application/json" \
  -d '{
    "query": "68yo male s/p L TKA POD 2, evaluate for PT",
    "discipline": "PT",
    "setting": "inpatient"
  }'

# Quick consultation
curl -X POST http://localhost:8080/api/v1/consult/quick \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Low back pain, acute onset, no red flags",
    "age": 45,
    "discipline": "PT"
  }'

# Safety check only
curl -X POST http://localhost:8080/api/v1/consult/safety \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Patient with saddle anesthesia and bladder dysfunction"
  }'
```

### Python API

```python
import asyncio
from rehab_os.main import run_consultation

result = asyncio.run(run_consultation(
    "68yo male s/p L TKA POD 2, evaluate for PT",
    discipline="PT",
    setting="inpatient",
    age=68,
    sex="male",
))

print(f"Safe to treat: {result.safety.is_safe_to_treat}")
print(f"Diagnosis: {result.diagnosis.primary_diagnosis}")
print(f"Goals: {len(result.plan.smart_goals)}")
```

## Agents

| Agent | Purpose | Key Output |
|-------|---------|------------|
| **RedFlagAgent** | Safety screening and triage | Red flags, urgency level, referral recommendations |
| **DiagnosisAgent** | Clinical classification | ICD codes, diagnosis, differential, rationale |
| **EvidenceAgent** | Guideline and literature search | Citations, evidence synthesis, recommendations |
| **PlanAgent** | Treatment planning | SMART goals, interventions, HEP, discharge criteria |
| **OutcomeAgent** | Outcome measure selection | Recommended measures, MCID/MDC, schedule |
| **DocumentationAgent** | Clinical note generation | Eval notes, daily notes, discharge summaries |
| **QALearningAgent** | Quality assurance | Critique, suggestions, uncertainty flags |

## Project Structure

```
rehab-os/
├── rehab_os/
│   ├── agents/         # Specialized agents
│   ├── api/            # REST API (FastAPI)
│   ├── cli/            # CLI (Typer)
│   ├── knowledge/      # Vector store & PubMed
│   ├── llm/            # LLM abstraction layer
│   └── models/         # Pydantic data models
├── tests/              # Test suite
├── data/
│   └── guidelines/     # CPG documents
└── pyproject.toml
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=rehab_os

# Type checking
mypy rehab_os

# Linting
ruff check rehab_os
```

## Disclaimer

This is clinical decision support information only. It is not a substitute for professional clinical judgment. All recommendations should be verified by a licensed clinician.

## License

MIT License
