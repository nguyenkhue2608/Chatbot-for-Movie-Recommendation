# 🎬 Film Recommendation Chatbot

A Streamlit-based film recommendation system that uses ChromaDB for vector storage and OpenAI API for intelligent recommendations.

## Features

- **Film Database Management**: Add films with details like title, genre, director, year, description, timeslot, and rating
- **Semantic Search**: Search films using natural language queries with ChromaDB vector similarity
- **AI-Powered Recommendations**: Get personalized film suggestions using OpenAI's GPT model
- **Interactive Chat Interface**: Chat-based interface for natural film discovery
- **Film Catalog**: View all films stored in the database

## Setup Instructions

### 1. Activate Virtual Environment
```bash
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Azure OpenAI
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` file and add your Azure OpenAI configuration:
   ```
   AZURE_OPENAI_API_KEY=your_actual_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
   ```
3. Get these values from your Azure OpenAI resource in the Azure Portal

### 4. Run the Application
```bash
streamlit run app.py
```

## Usage

### Adding Films
1. Navigate to the "Add Films" page
2. Fill in the film details:
   - Title, Genre, Director, Year (required)
   - Description (required)
   - Rating (1-10 scale)
   - Available timeslot
3. Click "Add Film" to store in ChromaDB

### Getting Recommendations
1. Go to "Chat & Recommendations" page
2. Type natural language queries like:
   - "I want a comedy movie for tonight"
   - "Recommend a sci-fi film with high rating"
   - "What's a good drama from the 2000s?"
3. The AI will search the database and provide personalized recommendations

### Viewing All Films
- Use the "View All Films" page to see all stored films in a table format

## Technology Stack

- **Streamlit**: Web application framework
- **ChromaDB**: Vector database for semantic search
- **Azure OpenAI**: GPT model for intelligent recommendations
- **Python**: Backend logic and data processing

## Project Structure

```
Workshop/
├── venv/                 # Virtual environment
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example        # Environment template
├── .env                # Your API keys (create this)
├── chroma_db/          # ChromaDB storage (auto-created)
└── README.md           # This file
```

## Features Overview

### Data Entry Interface
- User-friendly form for adding film information
- Validation for required fields
- Dropdown selections for genres and timeslots
- Rating slider for user experience

### Semantic Search & RAG
- Films are stored as vector embeddings in ChromaDB
- Natural language queries are converted to vectors
- Cosine similarity search finds relevant films
- Retrieved films serve as context for AI recommendations

### AI Chatbot
- Conversational interface using Streamlit's chat components
- Context-aware recommendations based on search results
- Explanations for why films match user preferences
- Chat history maintained during session

## Troubleshooting

- **Azure OpenAI API Error**: Ensure all Azure OpenAI configuration values are correctly set in the `.env` file
- **ChromaDB Issues**: The database is created automatically; check file permissions
- **Dependencies**: Make sure all packages are installed with correct versions

## Future Enhancements

- User authentication and personal watchlists
- Film poster integration
- Advanced filtering options
- Export recommendations to external services
- Multi-language support