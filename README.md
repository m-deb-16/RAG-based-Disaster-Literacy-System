# 🚨 Disaster Literacy RAG System

A Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware disaster preparedness and response guidance. The system supports both offline (local LLM) and online (API-based LLM) modes with multiple operational modes for different use cases.

## 🎯 Features

### Core Capabilities
- **Multi-Mode Operation**: Advisory, Educational, and Simulation modes
- **Hybrid LLM Support**: Works offline with local models or online with cloud APIs
- **Intelligent Document Processing**: OCR support for scanned PDFs, automatic chunking, and metadata extraction
- **Vector-Based Retrieval**: FAISS-powered semantic search for relevant context
- **Citation Tracking**: All responses include source citations for verification
- **Knowledge Base Management**: Admin panel for document upload, viewing, and deletion

### Operational Modes

#### 🛡️ Advisory Mode
Get immediate, actionable safety guidance during disaster situations
- Concise action checklists
- Disaster-specific filtering (Tsunami, Flood, Cyclone, Earthquake, Fire, Landslide)
- Source-grounded responses

#### 📚 Educational Mode
Learn about disaster preparedness and response strategies
- Comprehensive explanations
- Educational content with citations
- Topic-based learning

#### 🎯 Simulation Mode (Online Only)
Test your disaster preparedness knowledge with interactive scenarios
- AI-generated disaster scenarios
- Multiple-choice questions with explanations
- Instant feedback and scoring

### Prerequisites

#### Required Software
- **Python**: 3.8 or higher
- **Tesseract OCR**: For processing scanned PDFs
  - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

#### For Offline Mode
- **Offline LLM Models**: Download quantized GGUF models
  - Economy mode: `llama-2-7b-chat.Q4_K_M.gguf` (~4GB)
  - Power mode: `qwen2-7b-instruct-q4_k_m.gguf` (~4GB)
  - Place models in `./models/` directory

#### For Online Mode
- **API Keys**:
  - Google Gemini API key (free tier available)
  - OpenRouter API key (for Qwen models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/m-deb-16 RAG-based-Disaster-Literacy-System.git
   cd RAG-Proj
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and paths
   ```

4. **Download offline models** (if using offline mode)
   - Download models from [Hugging Face](https://huggingface.co/models)
   - Place in `./models/` directory
   - Update paths in `.env` if needed

5. **Set up Tesseract**
   - Install Tesseract OCR (see Prerequisites)
   - Update `TESSERACT_CMD` path in `.env` if needed

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### 1. Initialize the System

1. Open the application in your browser
2. Select your preferred mode in the sidebar:
   - **Offline**: Choose Economy or Power mode
   - **Online**: Choose Google Gemini or OpenRouter
3. Click **"🔄 Initialize System"**
4. Wait for initialization to complete

### 2. Using Advisory Mode

1. Select **Advisory** mode from the sidebar
2. Choose a disaster type (or "Any")
3. Enter your query (e.g., "What should I do during a tsunami warning?")
4. Click **"Get Advice"**
5. Review the response, action checklist, and cited sources

### 3. Using Educational Mode

1. Select **Educational** mode from the sidebar
2. Enter a topic (e.g., "Tsunami safety measures")
3. Click **"Learn"**
4. Review the educational content and sources

### 4. Using Simulation Mode (Online Only)

1. Ensure you're in **Online** mode
2. Select **Simulation** mode from the sidebar
3. Enter a scenario topic (e.g., "Coastal tsunami scenario")
4. Click **"Start Simulation"**
5. Read the scenario and answer all questions
6. Click **"Submit Answers"** to see your score
7. Review detailed explanations for each question

### 5. Managing the Knowledge Base

1. Navigate to **Admin Panel** from the sidebar
2. **Upload Documents** tab:
   - Upload PDF, TXT, or MD files
   - Select disaster category
   - Click **"Process All Documents"**
   - Wait for processing (OCR may take time for scanned PDFs)
3. **KB Statistics** tab:
   - Click **"Refresh Stats"** to view all documents
   - View document details (chunks, disaster type, etc.)
   - Delete documents if needed
4. **System Info** tab:
   - View system statistics and configuration

## 🧪 Testing

### Latency Testing

Test response times for different configurations:

```bash
python test_latency.py
```

This will test:
- Offline Economy mode
- Offline Power mode
- Online Google mode
- Online OpenRouter mode

### Accuracy Testing

Evaluate response quality with predefined test cases:

```bash
python test_accuracy.py
```

Metrics evaluated:
- Citation presence
- Response relevance
- Grounding to source material

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
