# AI-Powered Multi-Agent ICD-10 Clinical Coding from Unstructured Medical Text 🏥

A sophisticated multi-agent healthcare AI system that leverages LangGraph for intelligent medical analysis and routing. The system intelligently processes clinical inputs including clinical notes, medical transcripts, and medical images, routing them to specialized agents for analysis.


## Overview

The Multi-Agent Medical System is an intelligent healthcare assistant that combines multiple specialized AI agents working together to provide comprehensive medical analysis. Using LangGraph's state-based architecture, the system intelligently routes medical inputs to the most appropriate processing agent based on the type of input provided.

### Key Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for different medical tasks
- 🧠 **Intelligent Routing**: Automatic detection and routing of medical inputs
- 📋 **ICD-10 Coding**: Automatic extraction and coding of medical conditions using ICD-10 standards
- 📝 **SOAP Note Generation**: Convert medical transcripts into structured SOAP notes
- 🖼️ **Medical Image Analysis**: Analyze medical images (X-rays, MRI, CT scans, etc.)
- 🎯 **Vision-Language Model**: Powered by MedGemma-4B, optimized for medical tasks
- 📱 **Web Interface**: User-friendly web UI for easy interaction
- 🔍 **LangSmith Integration**: Built-in monitoring and tracing for agent performance

## System Architecture

### Agent Types

The system uses a router agent that intelligently directs medical inputs to one of three specialized agents:

#### 1. **Router Agent**
- Entry point for all medical inputs
- Analyzes both textual and image inputs
- Routes to the appropriate specialized agent based on input type
- Returns one of: `"icd10"`, `"soap"`, or `"image_analysis"`

#### 2. **ICD-10 Agent**
- Extracts and codes medical conditions using ICD-10 standards
- Processes clinical notes and medical narratives
- Returns structured ICD-10 codes with descriptions
- Ideal for billing, documentation, and clinical coding

#### 3. **SOAP Generator Agent**
- Converts medical transcripts into structured SOAP notes
- Generates Subjective, Objective, Assessment, and Plan sections
- Standardizes medical documentation
- Supports clinical communication and documentation

#### 4. **Image Analyzer Agent**
- Analyzes medical images (X-rays, MRI, CT scans, ultrasounds)
- Provides comprehensive image analysis including:
  - Medical technique used
  - Detailed findings
  - Clinical impression
  - Recommendations
  - Answers to user-specific questions about the image

### LangGraph State Management

The system uses a structured state graph to manage the flow of data:

```
START → Router Agent → Conditional Routing
                        ├── ICD-10 Agent → END
                        ├── SOAP Agent → END
                        └── Image Analysis Agent → END
```

## Technology Stack

### Core Technologies
- **LangGraph**: Orchestration and state management
- **FastAPI**: Web API framework for endpoints
- **MLX**: Apple Silicon-optimized ML framework
- **MLX-LM**: Language model operations on Apple Silicon
- **MLX-VLM**: Vision-Language model operations
- **MedGemma-4B**: Medical domain foundation model (4-bit quantized)

### Key Dependencies
- **Pydantic**: Data validation and schema management
- **Langsmith**: LLM monitoring and evaluation
- **Pillow**: Image processing
- **Python-dotenv**: Environment variable management

## Project Structure

```
Multi_Agent_Medical_System/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Base agent class
│   │   ├── router_agent.py     # Routing agent
│   │   ├── icd10_agent.py      # ICD-10 coding agent
│   │   ├── soap_generator_agent.py  # SOAP note generation
│   │   └── image_analyzer_agent.py  # Medical image analysis
│   ├── api/
│   │   ├── analyze.py          # FastAPI route handlers
│   │   └── schemas.py          # Pydantic models for I/O
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration settings
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph_builder.py    # LangGraph construction
│   │   └── types.py            # Type definitions for graph state
│   ├── static/
│   │   └── index.html          # Web UI
│   └── utils/
│       ├── __init__.py
│       ├── helper.py           # Utility functions
│       ├── logger.py           # Logging configuration
│       ├── model_loader.py     # Model loading utilities
│       ├── predictor.py        # Prediction utilities
│       └── prompt_builder.py   # Prompt construction
├── artifacts/                  # Generated output files
├── evaluations/
│   └── synthetic_icd10_dataset.json
├── experiments/                # Jupyter notebooks
│   ├── ICD10_extraction_from_clinical_notes.ipynb
│   ├── image_analysis.ipynb
│   └── SOAP_generation_from_transcripts.ipynb
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8+
- macOS with Apple Silicon (M1/M2/M3) or Linux with GPU
- 8GB+ RAM
- ~4GB disk space for model

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Multi_Agent_Medical_System.git
cd Multi_Agent_Medical_System
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the root directory:
```bash
# LangSmith Configuration (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=your_project_name
```

5. **Run the application**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at: `http://localhost:8000`

## Usage

### Web Interface
Visit `http://localhost:8000` to access the web UI where you can:
- Upload clinical notes for ICD-10 coding
- Upload medical transcripts for SOAP note generation
- Upload medical images for analysis
- View structured analysis results

### API Endpoints

#### POST `/api/analyze`
Process medical input and get analysis

**Request Parameters:**
- `note` (string, optional): Clinical note or transcript text
- `image` (file, optional): Medical image file

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "note=Patient presents with acute appendicitis and fever" \
  -F "image=@chest_xray.jpg"
```

**Response Examples:**

ICD-10 Response:
```json
{
  "agent": "icd10",
  "result": [
    {
      "code": "K35.80",
      "description": "Acute appendicitis with generalized peritonitis"
    }
  ]
}
```

SOAP Response:
```json
{
  "agent": "soap",
  "result": {
    "Subjective": "Patient reports...",
    "Objective": "Vital signs: BP 120/80...",
    "Assessment": "Acute appendicitis...",
    "Plan": "Schedule for surgery..."
  }
}
```

Image Analysis Response:
```json
{
  "agent": "image_analysis",
  "result": {
    "technique": "Chest X-ray, PA view",
    "findings": "No acute findings...",
    "impression": "Normal chest X-ray",
    "recommendations": "No follow-up imaging needed",
    "answer_to_user_question": "No pneumonia detected"
  }
}
```

## Data Flow

### Clinical Note Processing
```
Clinical Note → Router Agent → Identifies as "icd10" 
             → ICD-10 Agent → Extracts codes → Response
```

### Transcript Processing
```
Medical Transcript → Router Agent → Identifies as "soap"
                  → SOAP Agent → Generates SOAP → Response
```

### Image Analysis
```
Medical Image → Router Agent → Identifies as "image_analysis"
             → Image Agent → Analyzes → Response
```

## Configuration

Edit `app/config/config.py` to customize system settings:

```python
class Config:
    MODEL_ID = "mlx-community/medgemma-4b-it-4bit"  # Model identifier
    # Add additional configuration options as needed
```

### Environment Variables

- `LANGCHAIN_TRACING_V2`: Enable LangSmith tracing (true/false)
- `LANGCHAIN_ENDPOINT`: LangSmith API endpoint
- `LANGCHAIN_API_KEY`: LangSmith API key
- `LANGCHAIN_PROJECT`: LangSmith project name

## Model Information

The system uses **MedGemma-4B**, a specialized medical language model:
- **Size**: 4B parameters (4-bit quantized for efficiency)
- **Optimization**: Apple Silicon optimized via MLX
- **Specialization**: Medical domain knowledge
- **Vision**: Vision-language capabilities for image understanding

**Model Loading**: The model is lazily loaded on first use and cached for subsequent requests, reducing memory overhead.

## Features & Capabilities

### ICD-10 Coding
- Extracts medical conditions from clinical notes
- Maps to appropriate ICD-10 codes
- Includes code descriptions
- Supports disease, symptom, and condition coding

### SOAP Note Generation
- Converts unstructured transcripts to structured SOAP format
- Generates comprehensive assessments
- Creates actionable treatment plans
- Suitable for clinical documentation

### Medical Image Analysis
- Analyzes various medical imaging types:
  - X-rays (chest, extremities, etc.)
  - MRI scans
  - CT scans
  - Ultrasound images
- Provides detailed findings and impressions
- Supports clinical questions about images

## Error Handling

The system includes comprehensive error handling:
- Invalid input validation
- JSON parsing error recovery
- Model loading error handling
- Graceful error responses

**Common Error Messages:**
- `"No input provided"`: Both note and image are empty
- `"Unexpected output type from graph"`: Internal processing error
- `"Unknown analysis type"`: Routing error detected

## Monitoring & Evaluation

### LangSmith Integration
The system integrates with LangSmith for:
- Agent performance monitoring
- Request tracing
- Cost analysis
- Performance optimization

Enable by setting environment variables (see Configuration section).

### Jupyter Notebooks
Experiment notebooks in the `experiments/` folder:
- `ICD10_extraction_from_clinical_notes.ipynb`: ICD-10 agent examples
- `image_analysis.ipynb`: Image analysis demonstrations
- `SOAP_generation_from_transcripts.ipynb`: SOAP generation examples

## Development

### Running Experiments
```bash
jupyter notebook experiments/
```

### Testing Agents
Each agent has example code (commented) that can be uncommented for testing:
```python
# In agent files, uncomment the if __name__ == "__main__" section
if __name__ == "__main__":
    agent = ICD10Agent()
    # Run agent tests
```

## Performance Considerations

- **Apple Silicon**: Optimized for M1/M2/M3 through MLX
- **Model Caching**: Single model instance shared across agents
- **Lazy Loading**: Model loaded on first request
- **Memory Efficient**: 4-bit quantization reduces memory footprint

## Future Enhancements

- [ ] Support for additional medical image types
- [ ] Integration with electronic health records (EHR)
- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] Advanced reasoning for complex cases
- [ ] Real-time streaming responses
- [ ] Batch processing capabilities

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
