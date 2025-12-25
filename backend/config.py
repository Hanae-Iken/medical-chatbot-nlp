"""
Configuration du Chatbot Médical
Ne pas commiter ce fichier avec la vraie clé API !
"""

import os

class Config:
    """Configuration principale de l'application"""
    
    # API Gemini
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or 'VOTRE_CLE_API_ICI'
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-me'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # CORS
    CORS_ORIGINS = ['http://localhost:8000', 'http://127.0.0.1:8000']
    
    # NLP
    LANGUAGE = 'french'
    MIN_SIMILARITY_SCORE = 0.1
    TOP_RESULTS = 3
    
    # Base de données (pour future extension)
    DATABASE_URI = 'sqlite:///chatbot.db'  # ou MongoDB, PostgreSQL
    
    # Limites
    MAX_MESSAGE_LENGTH = 500
    MAX_HISTORY_PER_SESSION = 50
    SESSION_TIMEOUT_MINUTES = 30


class ProductionConfig(Config):
    """Configuration pour production"""
    DEBUG = False
    # En production, toujours utiliser variables d'environnement
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    SECRET_KEY = os.environ.get('SECRET_KEY')


class DevelopmentConfig(Config):
    """Configuration pour développement"""
    DEBUG = True


# Sélectionner la configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


# ========== Instructions pour obtenir la clé API Gemini ==========
"""
1. Aller sur https://makersuite.google.com/app/apikey
2. Se connecter avec un compte Google
3. Cliquer sur "Create API Key"
4. Copier la clé (commence par AIza...)
5. Remplacer 'VOTRE_CLE_API_ICI' ci-dessus

OU mieux encore, définir une variable d'environnement :

Windows (PowerShell):
$env:GEMINI_API_KEY="votre_cle_ici"

Puis démarrer l'application.
"""