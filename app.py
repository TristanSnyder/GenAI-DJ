import os
from dotenv import load_dotenv
from typing import List
import logging

import openai
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifySimpleAgent:
    def __init__(self):
        # Initialize API keys
        openai.api_key = os.getenv("OPENAI_API_KEY")
        creds = SpotifyClientCredentials(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
        )
        self.sp = Spotify(client_credentials_manager=creds)
        # Embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Setup ChromaDB collection
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("music_knowledge")
        except:
            pass
        self.collection = self.client.create_collection("music_knowledge")
        self._populate()

    def _populate(self):
        """Fetch and index Spotify playlist tracks"""
        playlist_id = "37i9dQZEVXbMDoHDwVN2tF"  # Global Top 50
        try:
            results = self.sp.playlist_items(
                playlist_id,
                fields="items.track(id,name,artists(name))",
                limit=50,
                market="US"
            )
        except Exception as e:
            logger.error(f"Spotify fetch error: {e}")
            return

        for item in results.get("items", []):
            t = item.get("track", {})
            tid = t.get("id")
            title = t.get("name")
            artists = t.get("artists") or []
            artist = artists[0].get("name") if artists else "Unknown"
            text = f"{title} by {artist}"
            emb = self.encoder.encode(text)
            # Store only embeddings and document text
            self.collection.add(
                embeddings=[emb.tolist()],
                documents=[text],
                metadatas=[{"id": tid}],
                ids=[tid]
            )
        logger.info("Indexed Spotify playlist items.")

    def _call_llm(self, user_input: str, docs: List[str]) -> str:
        """Generate a narrative recommendation with an LLM"""
        prompt = (
            f"You are a helpful music curator. The user wants: '{user_input}'.\n"
            "Here are some candidate tracks:\n" +
            "\n".join(f"- {d}" for d in docs) +
            "\nPlease write a concise, enthusiastic recommendation highlighting the top 5 songs and why they fit the request."
        )
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()

    def get_recommendations(self, user_input: str) -> str:
        """Retrieve tracks, generate narrative, and format Markdown output"""
        if not user_input.strip():
            return "**Please describe what kind of music you're looking for!**"

        # Retrieve the top 5 documents
        emb = self.encoder.encode(user_input)
        res = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=5,
            include=["documents"]
        )
        docs = res["documents"][0]

        # Generate narrative recommendation
        narrative = self._call_llm(user_input, docs)

        # Build Markdown output
        md = f"### 🎵 Recommendations for: {user_input}\n\n"
        md += narrative + "\n\n"
        md += "#### \n"
        for d in docs:
            md += f"- {d}\n"
        return md

# Instantiate the agent
agent = SpotifySimpleAgent()

# Build Gradio app
def create_app():
    with gr.Blocks() as app:
        gr.Markdown("# 🎵 Spotify RAG Agent")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(
                    label="Describe your mood or activity:",
                    lines=2,
                    placeholder="e.g., energetic workout mix"
                )
                btn = gr.Button("Recommend 🎶")
            with gr.Column(scale=2):
                out = gr.Markdown("Your personalized picks will appear here.")
        btn.click(agent.get_recommendations, inp, out)
        inp.submit(agent.get_recommendations, inp, out)
    return app

if __name__ == '__main__':
    create_app().launch(share=True)
