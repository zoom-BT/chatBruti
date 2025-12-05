from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI(title="Chat-Bruti", version="1.0")

# Client Groq (GRATUIT !)
client = Groq(api_key=os.getenv("GROQ_API_KEY", "TON_API_KEY")) # TON_API_KEY n'existe pas et permet de renvoyer une erreur s'il n'y a pas d'api key

# Modèles de données
class UserQuery(BaseModel):
    contexte: str
    question: str

    class Config:
        extra = "ignore"

class ChatResponse(BaseModel):
    response: str

#Prompt
CHAT_BRUTI_SYSTEM_PROMPT = """
Tu es Chat-Bruti, un chatbot volontairement con.
Tu ne réponds jamais directement à la question.
Tu exagères, tu inventes, tu oublies.
Tu fais de l'humour .
Tu te prends pour un philosophe du dimanche.
Les figures de style et les jeux de mots sont tes meilleurs amis.
Ta réponse doit être courte, inutile, mais drôle.
Appuie-toi très vaguement sur le contexte ci-dessous, mais détourne-le complètement.
Utilise des mots comme waouh, yeahh, oups, dans tes réponses
"""

# root
@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur Chat-Bruti API ",
        "model": "llama-3.1-70b-versatile",
        "endpoints": {
            "/ask": "POST - Pose une question",
            "/docs": "Documentation"
        }
    }

# ask
@app.post("/ask", response_model=ChatResponse)
async def ask_chatbruti(payload: UserQuery):
    try:
        context = payload.contexte.strip()
        question = payload.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question vide")
        
        user_prompt = f"""Voici le contexte récupéré de la base de connaissances : {context} ;
                        la question de l'utilisateur : {question}
                    Réponds de manière complètement absurde en détournant le contexte !"""
        # Appel à Groq 
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Modèle actuel et gratuit
            messages=[
                {"role": "system", "content": CHAT_BRUTI_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.5,      
            max_tokens=200,
            top_p=0.95
        )
        
        answer = completion.choices[0].message.content
        
        return ChatResponse(response=answer)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur : {str(e)}"
        )



