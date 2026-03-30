
# Self-RAG with Query Rewrite

A complete Self-Reflective Retrieval-Augmented Generation (RAG) implementation using LangGraph. This project implements a 7-step Self-RAG pipeline with query rewriting for improved retrieval quality.

<p align="center">
  <img src="https://raw.githubusercontent.com/Shree2106/self_reflective_rag/main/self_rag_langgraph_workflow.svg" width="700"/>
</p>

## Overview

This Self-RAG implementation adds self-reflection loops to the standard RAG pipeline:

1. **Decide Retrieval** — Determines if documents are needed or if the question can be answered from general knowledge
2. **Retrieve** — Fetches relevant documents from FAISS vector store
3. **Is Relevant** — Grades each retrieved chunk for relevance to the query
4. **Generate from Context** — Produces an answer using only relevant documents
5. **IsSUP (Is Supported?)** — Verifies the answer is fully grounded in retrieved context
6. **Revise Answer** — Revises answers that are not fully supported (loop until max retries)
7. **IsUSE (Is Useful?)** — Checks if the answer actually addresses the question
8. **Query Rewrite** — If not useful, rewrites the query and retrieves again (up to 3 tries)

## Features

- **Adaptive Retrieval**: Only retrieves documents when necessary
- **Relevance Grading**: Filters out irrelevant chunks before generation
- **Grounding Verification**: Ensures answers are supported by evidence
- **Automatic Revision**: Revises unsupported answers with strict quoting
- **Query Optimization**: Rewrites ineffective queries for better retrieval
- **Loop Control**: Configurable retry limits for both revision and query rewrite

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster embeddings

### Step-by-Step Installation

**1. Clone or download the repository:**
```bash
cd main.final
```

**2. Create a virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Verify installation:**
```bash
python -c "from langgraph.graph import StateGraph; print('✅ All packages installed')"
```

### Manual Installation (Alternative)

If you prefer to install packages individually:
```bash
# Core LangChain packages
pip install langchain langchain-community langchain-openai

# LangGraph for the workflow graph
pip install langgraph

# Vector store and document processing
pip install faiss-cpu pypdf

# Data validation
pip install pydantic
```

## Usage

1. **Set your OpenAI API key** in cell 4 of the notebook:
```python
OPENAI_API_KEY = "your-openai-api-key-here"
```

2. **Prepare your documents**: Place PDF files in the `./documents/` directory
   - `Company_Policies.pdf`
   - `Company_Profile.pdf`
   - `Product_and_Pricing.pdf`

3. **Run the notebook**: Execute all cells in `self_rag_step7.ipynb`

4. **Ask a question**:
```python
question = "Describe NexaAI's company culture."
result = app.invoke({
    "question": question,
    "retrieval_query": "",
    "rewrite_tries": 0,
    "need_retrieval": False,
    "docs": [],
    "relevant_docs": [],
    "context": "",
    "answer": "",
    "issup": "",
    "evidence": [],
    "retries": 0,
    "isuse": "not_useful",
    "use_reason": "",
}, config={"recursion_limit": 80})
```

## Project Structure

```
.
├── self_rag_step7.ipynb    # Main notebook with complete implementation
├── requirements.txt        # Python dependencies
├── self_rag_flowchart.svg  # Architecture diagram
└── documents/              # PDF documents for RAG
    ├── Company_Policies.pdf
    ├── Company_Profile.pdf
    └── Product_and_Pricing.pdf
```

## Architecture

The Self-RAG pipeline is built as a **state machine** using LangGraph, where each node represents a decision point or processing step.

### Node Types (by color in diagram)

| Color | Type | Nodes | Purpose |
|-------|------|-------|---------|
| **Blue** | Processing | `decide_retrieval`, `retrieve`, `generate_from_context`, `generate_direct` | Core document processing and generation |
| **Green** | Verification | `is_relevant`, `is_sup`, `is_use` | Quality checks and validation |
| **Yellow** | Routing | `route_after_decide`, `route_after_relevance`, `route_after_issup`, `route_after_isuse` | Conditional flow control |

### Data Flow

```
START
  ↓
decide_retrieval ──(no retrieval needed)──→ generate_direct ──→ END
  ↓ (retrieval needed)
retrieve
  ↓
is_relevant ──(no relevant docs)──→ no_relevant_docs ──→ END
  ↓ (has relevant docs)
generate_from_context
  ↓
is_sup (Is Supported?)
  ├─ fully_supported ──→ is_use
  └─ not_supported ──→ revise_answer ──┐
       ↑───────────────────────────────┘ (loop until MAX_RETRIES)
       └─ after max retries → is_use

is_use (Is Useful?)
  ├─ useful ──→ END
  └─ not_useful ──→ rewrite_question ──→ retrieve (loop until MAX_REWRITE_TRIES)
       └─ after max rewrites ──→ no_relevant_docs ──→ END
```

### State Object

The `State` TypedDict carries data through the graph:

```python
class State(TypedDict):
    question: str           # Original user question
    retrieval_query: str    # Rewritten query for FAISS search
    rewrite_tries: int      # Counter for rewrite attempts
    need_retrieval: bool    # Decision from decide_retrieval node
    docs: List[Document]    # All retrieved documents
    relevant_docs: List[Document]  # Filtered relevant documents
    context: str            # Joined text of relevant documents
    answer: str             # Generated answer
    issup: str              # IsSUP result: "fully_supported" | "partially_supported" | "no_support"
    evidence: List[str]     # Supporting evidence quotes
    retries: int            # Counter for revision attempts
    isuse: str              # IsUSE result: "useful" | "not_useful"
    use_reason: str         # Explanation for usefulness judgment
```

### Key Design Patterns

1. **Conditional Edges**: Router functions determine the next node based on state
2. **Self-Loops**: `revise_answer` and `rewrite_question` loop back to verification/retrieval
3. **Recursion Limit**: LangGraph's `recursion_limit` prevents infinite loops
4. **Structured Output**: Pydantic models enforce LLM response schemas

## Configuration

### Environment Variables

Create a `.env` file or set these in your environment:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Custom model selection
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
```

### Adjustable Constants

Edit these in `self_rag_step7.ipynb` (Cell 29 and Cell 33):

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_RETRIES` | 10 | Maximum revision attempts when IsSUP fails |
| `MAX_REWRITE_TRIES` | 3 | Maximum query rewrites when IsUSE fails |

### Chunking Parameters

In Cell 10, adjust text splitting:

```python
RecursiveCharacterTextSplitter(
    chunk_size=600,    # Characters per chunk (reduce for finer granularity)
    chunk_overlap=150  # Overlap between chunks (prevents context loss)
)
```

### Retrieval Settings

In Cell 12, adjust FAISS retriever:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}  # Number of chunks to retrieve
)
```

### LLM Configuration

In Cell 14, adjust generation parameters:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Model name (gpt-4o-mini, gpt-4o, etc.)
    temperature=0          # Lower = more deterministic, higher = more creative
)
```

### Recursion Safety

In Cell 40, prevent infinite loops:

```python
config={"recursion_limit": 80}  # Maximum graph steps before timeout
```

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## License

MIT License
