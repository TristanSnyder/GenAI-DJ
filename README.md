🎵 Spotify RAG Agent
A smart music recommendation system that combines Spotify's API with Retrieval-Augmented Generation (RAG) to provide personalized music suggestions based on your mood, activity, or preferences.
## 🚀 Try It Live **[Launch the App on Hugging Face Spaces](https://huggingface.co/spaces/kts7gw/TristanSnyder)**
🌟 What It Does
The Spotify RAG Agent analyzes Spotify's Global Top 50 playlist and uses AI to understand your music preferences in natural language. Simply describe what you're looking for - like "energetic workout mix" or "chill study vibes" - and get personalized recommendations with explanations of why each song fits your request.
Key Features

Natural Language Input: Describe your mood or activity in plain English
AI-Powered Matching: Uses sentence transformers to find the best matches for your request
Intelligent Recommendations: GPT-3.5 generates personalized explanations for each recommendation
Real-Time Data: Pulls from Spotify's current Global Top 50 playlist
Beautiful Interface: Clean, user-friendly Gradio web interface

🚀 Try It Live
Launch the App on Hugging Face Spaces
🛠️ How It Works

Data Collection: Fetches current tracks from Spotify's Global Top 50 playlist
Embedding Generation: Creates semantic embeddings for each track using SentenceTransformer
Vector Storage: Stores embeddings in ChromaDB for efficient similarity search
Query Processing: Converts your natural language input into embeddings
Similarity Matching: Finds the most relevant tracks based on semantic similarity
AI Generation: Uses OpenAI's GPT-3.5 to generate personalized recommendations with explanations

🔧 Technical Stack

Python: Core programming language
Spotify API: Music data source
OpenAI GPT-3.5: Natural language generation
SentenceTransformers: Semantic embedding model
ChromaDB: Vector database for similarity search
Gradio: Web interface framework

🎯 Use Cases

Fitness Enthusiasts: Get high-energy tracks for workouts
Students: Find focus-friendly background music
Party Planners: Discover crowd-pleasing dance tracks
Mood-Based Listening: Match music to your current emotional state
Activity-Specific Playlists: Find perfect soundtracks for any occasion

🔮 Future Enhancements

Support for multiple playlist sources
User preference learning
Spotify playlist creation integration
Audio feature analysis (tempo, energy, etc.)
Personalized user profiles

🤝 Contributing
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Spotify for providing the Web API
OpenAI for GPT-3.5 Turbo
Hugging Face for the SentenceTransformers library
ChromaDB for vector storage capabilities


Ready to discover your next favorite song? Try the app now!
