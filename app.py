import streamlit as st
import chromadb
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, time
import json
import warnings

# Suppress ChromaDB telemetry warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*config file.*")

# Load environment variables
load_dotenv()

# Initialize ChromaDB
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="films",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

# Initialize Azure OpenAI
def init_azure_openai():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([api_key, endpoint, api_version, deployment_name]):
        st.error("Please set all Azure OpenAI configuration in the .env file")
        return None, None
    
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        return client, deployment_name
    except TypeError as e:
        if "proxies" in str(e):
            st.error("OpenAI library version compatibility issue. Please update to openai>=1.12.0")
        else:
            st.error(f"Error initializing Azure OpenAI client: {str(e)}")
        return None, None

def add_film_to_db(collection, title, genre, director, year, description, timeslot, rating):
    """Add a film to ChromaDB"""
    film_data = {
        "title": title,
        "genre": genre,
        "director": director,
        "year": year,
        "description": description,
        "timeslot": timeslot,
        "rating": rating
    }
    
    # Create a comprehensive text for embedding
    film_text = f"Title: {title}. Genre: {genre}. Director: {director}. Year: {year}. Description: {description}. Rating: {rating}/10"
    
    # Generate a unique ID
    film_id = f"{title}_{year}_{datetime.now().timestamp()}"
    
    try:
        collection.add(
            documents=[film_text],
            metadatas=[film_data],
            ids=[film_id]
        )
        return True
    except Exception as e:
        st.error(f"Error adding film to database: {str(e)}")
        return False

def import_films_from_json(collection, json_data):
    """Import multiple films from JSON data"""
    success_count = 0
    error_count = 0
    errors = []
    
    try:
        films = json.loads(json_data) if isinstance(json_data, str) else json_data
        
        if not isinstance(films, list):
            return False, "JSON must contain an array of films"
        
        for i, film in enumerate(films):
            try:
                # Validate required fields
                required_fields = ['title', 'genre', 'director', 'year', 'description']
                missing_fields = [field for field in required_fields if field not in film]
                
                if missing_fields:
                    error_count += 1
                    errors.append(f"Film {i+1}: Missing fields: {', '.join(missing_fields)}")
                    continue
                
                # Set default values for optional fields
                timeslot = film.get('timeslot', 'All Day')
                rating = film.get('rating', 7)
                
                # Validate data types
                if not isinstance(film['year'], int) or film['year'] < 1900 or film['year'] > 2024:
                    error_count += 1
                    errors.append(f"Film {i+1} ({film.get('title', 'Unknown')}): Invalid year")
                    continue
                
                if not isinstance(rating, (int, float)) or rating < 1 or rating > 10:
                    error_count += 1
                    errors.append(f"Film {i+1} ({film.get('title', 'Unknown')}): Invalid rating (must be 1-10)")
                    continue
                
                # Add film to database
                if add_film_to_db(collection, film['title'], film['genre'], 
                                film['director'], film['year'], film['description'], 
                                timeslot, rating):
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"Film {i+1} ({film.get('title', 'Unknown')}): Database error")
                    
            except Exception as e:
                error_count += 1
                errors.append(f"Film {i+1}: {str(e)}")
        
        return True, f"Successfully imported {success_count} films. {error_count} errors." + (f"\n\nErrors:\n" + "\n".join(errors) if errors else "")
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return False, f"Error importing films: {str(e)}"

def search_films(collection, query, n_results=5):
    """Search films in ChromaDB"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    except Exception as e:
        st.error(f"Error searching films: {str(e)}")
        return None

def get_ai_recommendation(client, deployment_name, query, search_results):
    """Get AI recommendation based on search results"""
    if not search_results or not search_results['documents'][0]:
        return "I couldn't find any films matching your criteria. Please try a different search."
    
    # Prepare context from search results
    context = ""
    for i, (doc, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
        context += f"Film {i+1}: {metadata['title']} ({metadata['year']}) - {metadata['genre']}\n"
        context += f"Director: {metadata['director']}\n"
        context += f"Description: {metadata['description']}\n"
        context += f"Rating: {metadata['rating']}/10\n"
        context += f"Available timeslot: {metadata['timeslot']}\n\n"
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful film recommendation assistant. Based on the user's query and the available films, provide personalized recommendations with explanations."
                },
                {
                    "role": "user",
                    "content": f"User query: {query}\n\nAvailable films:\n{context}\n\nPlease recommend the most suitable films and explain why they match the user's preferences."
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI recommendation: {str(e)}"

def main():
    st.set_page_config(
        page_title="Film Recommendation Chatbot",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Film Recommendation Chatbot")
    
    # Initialize databases
    client, collection = init_chromadb()
    azure_client, deployment_name = init_azure_openai()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Add Films", "Chat & Recommendations", "View All Films"])
    
    if page == "Add Films":
        st.header("📝 Add New Film")
        
        # Create tabs for single film and bulk import
        tab1, tab2 = st.tabs(["➕ Add Single Film", "📁 Import from JSON"])
        
        with tab1:
            with st.form("add_film_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    title = st.text_input("Film Title*", placeholder="e.g., The Shawshank Redemption")
                    genre = st.selectbox("Genre*", [
                        "Action", "Adventure", "Animation", "Comedy", "Crime", 
                        "Documentary", "Drama", "Family", "Fantasy", "Horror", 
                        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
                    ])
                    director = st.text_input("Director*", placeholder="e.g., Frank Darabont")
                    year = st.number_input("Release Year*", min_value=1900, max_value=2024, value=2023)
                
                with col2:
                    rating = st.slider("Rating (1-10)", min_value=1, max_value=10, value=7)
                    timeslot = st.selectbox("Available Timeslot", [
                        "Morning (9:00-12:00)", "Afternoon (12:00-17:00)", 
                        "Evening (17:00-21:00)", "Night (21:00-24:00)", "All Day"
                    ])
                
                description = st.text_area("Description*", placeholder="Brief description of the film...")
                
                submitted = st.form_submit_button("Add Film")
                
                if submitted:
                    if title and genre and director and year and description:
                        if add_film_to_db(collection, title, genre, director, year, description, timeslot, rating):
                            st.success(f"✅ '{title}' has been added to the database!")
                            st.balloons()
                    else:
                        st.error("Please fill in all required fields (marked with *)")
        
        with tab2:
            st.subheader("📥 Import Films from JSON")
            
            # Download example JSON
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info("💡 **How to use:**\n1. Download the example JSON file\n2. Edit it with your films\n3. Upload the modified file")
            
            with col2:
                try:
                    with open("example_films.json", "r") as f:
                        example_json = f.read()
                    st.download_button(
                        label="📥 Download Example JSON",
                        data=example_json,
                        file_name="example_films.json",
                        mime="application/json",
                        help="Download this template and modify it with your films"
                    )
                except FileNotFoundError:
                    st.warning("Example JSON file not found")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a JSON file",
                type="json",
                help="Upload a JSON file containing an array of film objects"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    json_data = json.load(uploaded_file)
                    
                    # Preview the data
                    st.subheader("📋 Preview")
                    st.write(f"Found {len(json_data)} films in the uploaded file:")
                    
                    # Show first few films as preview
                    preview_df = pd.DataFrame(json_data[:3])  # Show first 3 films
                    st.dataframe(preview_df, use_container_width=True)
                    
                    if len(json_data) > 3:
                        st.write(f"... and {len(json_data) - 3} more films")
                    
                    # Import button
                    if st.button("🚀 Import All Films", type="primary"):
                        with st.spinner("Importing films..."):
                            success, message = import_films_from_json(collection, json_data)
                            
                            if success:
                                st.success(message)
                                if "Successfully imported" in message and "0 errors" in message:
                                    st.balloons()
                            else:
                                st.error(message)
                                
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON file: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
            
            # JSON format help
            with st.expander("📖 JSON Format Help"):
                st.markdown("""
                **Required fields for each film:**
                - `title` (string): Film title
                - `genre` (string): Film genre
                - `director` (string): Director name
                - `year` (integer): Release year (1900-2024)
                - `description` (string): Film description
                
                **Optional fields:**
                - `timeslot` (string): Available timeslot (default: "All Day")
                - `rating` (number): Rating 1-10 (default: 7)
                
                **Example format:**
                ```json
                [
                  {
                    "title": "The Shawshank Redemption",
                    "genre": "Drama",
                    "director": "Frank Darabont",
                    "year": 1994,
                    "description": "Two imprisoned men bond over a number of years...",
                    "timeslot": "Evening (17:00-21:00)",
                    "rating": 9
                  }
                ]
                ```
                """)
    
    elif page == "Chat & Recommendations":
        # Header with New Chat button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("💬 Chat & Get Recommendations")
        with col2:
            if st.button("🆕 New Chat", help="Start a new conversation"):
                st.session_state.messages = []
                st.rerun()
        
        if not azure_client:
            st.warning("Azure OpenAI is not configured. Please add your Azure OpenAI configuration to the .env file.")
            return
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Show chat status
        if len(st.session_state.messages) == 0:
            st.info("👋 Welcome! Start a new conversation by asking about films you'd like to watch.")
        else:
            st.caption(f"💬 Current conversation has {len(st.session_state.messages)} messages")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What kind of film are you looking for?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Search for films and get AI recommendation
            with st.chat_message("assistant"):
                with st.spinner("Searching for films and generating recommendations..."):
                    search_results = search_films(collection, prompt)
                    recommendation = get_ai_recommendation(azure_client, deployment_name, prompt, search_results)
                    st.markdown(recommendation)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": recommendation})
    
    elif page == "View All Films":
        st.header("📚 All Films in Database")
        
        try:
            # Get all films from ChromaDB
            all_films = collection.get()
            
            if all_films['metadatas']:
                films_df = pd.DataFrame(all_films['metadatas'])
                st.dataframe(films_df, use_container_width=True)
                
                st.info(f"Total films in database: {len(films_df)}")
            else:
                st.info("No films in the database yet. Add some films using the 'Add Films' page!")
        except Exception as e:
            st.error(f"Error retrieving films: {str(e)}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    Film Recommendation Chatbot powered by AI.
    
    Add films to build your database and get personalized recommendations!
    """)

if __name__ == "__main__":
    main()