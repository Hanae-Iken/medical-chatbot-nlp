"""
Chatbot Médical - Backend Flask avec Gemini API
Installation: pip install flask flask-cors google-generativeai pandas numpy scikit-learn nltk
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Télécharger les ressources NLTK nécessaires
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)

# Configuration Gemini API
GEMINI_API_KEY = "GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# ========== Classes de prétraitement ==========

class PreprocesseurMedical:
    def __init__(self):
        self.stemmer = SnowballStemmer("french")
        self.stop_words = set(stopwords.words('french'))
        self.stop_words_medicaux = {
            'patient', 'docteur', 'médecin', 'symptôme', 'symptômes',
            'traitement', 'maladie', 'diagnostic', 'consultation'
        }
        self.stop_words.update(self.stop_words_medicaux)

    def nettoyer_texte(self, texte):
        if pd.isna(texte):
            return ""
        texte = str(texte).lower()
        texte = re.sub(r'[^a-zàâäéèêëîïôöùûüç\s]', ' ', texte)
        texte = re.sub(r'\s+', ' ', texte).strip()
        return texte

    def tokeniser(self, texte):
        return word_tokenize(texte, language='french')

    def supprimer_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]

    def stemmer_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def preprocesser_texte(self, texte):
        texte_nettoye = self.nettoyer_texte(texte)
        tokens = self.tokeniser(texte_nettoye)
        tokens_sans_stopwords = self.supprimer_stopwords(tokens)
        tokens_stemmes = self.stemmer_tokens(tokens_sans_stopwords)
        return ' '.join(tokens_stemmes)


class AnalyseurSymptomes:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.preprocesseur = PreprocesseurMedical()
        self.df_medical = self.creer_base_donnees_medicale()
        self.preparer_donnees()

    def creer_base_donnees_medicale(self):
        """Base de données médicale enrichie"""
        data = {
            'symptomes': [
                "fièvre toux fatigue maux de tête",
                "douleur abdominale nausées vomissements",
                "douleur thoracique essoufflement palpitations",
                "maux de gorge écoulement nasal éternuements",
                "douleur articulaire gonflement rougeur",
                "vertiges nausées vision floue",
                "douleur dos difficulté mouvement raideur",
                "fatigue extrême pâleur essoufflement",
                "brûlures urinaire fréquent envie uriner",
                "éruption cutanée démangeaisons rougeur peau",
                "diarrhée crampes abdominales déshydratation",
                "migraine sensibilité lumière nausées",
                "toux persistante mucus fièvre légère",
                "douleur poitrine brûlure remontées acides",
                "tremblements sueurs palpitations anxiété",
                "engourdissement picotements faiblesse membre",
                "démangeaisons yeux larmoiement éternuements",
                "insomnie fatigue stress irritabilité",
                "douleur genou gonflement difficulté marcher",
                "saignement nez fréquent maux tête"
            ],
            'diagnostics': [
                "grippe",
                "gastro-entérite",
                "problème cardiaque",
                "rhume",
                "arthrite",
                "hypotension",
                "lombalgie",
                "anémie",
                "infection urinaire",
                "allergie cutanée",
                "intoxication alimentaire",
                "migraine",
                "bronchite",
                "reflux gastrique",
                "crise d'anxiété",
                "neuropathie",
                "conjonctivite allergique",
                "trouble du sommeil",
                "lésion méniscale",
                "hypertension"
            ],
            'specialites': [
                "médecine générale",
                "gastro-entérologie",
                "cardiologie",
                "médecine générale",
                "rhumatologie",
                "médecine générale",
                "orthopédie",
                "hématologie",
                "urologie",
                "dermatologie",
                "gastro-entérologie",
                "neurologie",
                "pneumologie",
                "gastro-entérologie",
                "psychiatrie",
                "neurologie",
                "ophtalmologie",
                "psychiatrie",
                "orthopédie",
                "cardiologie"
            ],
            'urgence': [
                "faible", "modérée", "élevée", "faible", "modérée",
                "modérée", "faible", "modérée", "modérée", "faible",
                "modérée", "faible", "faible", "faible", "modérée",
                "élevée", "faible", "faible", "modérée", "modérée"
            ]
        }
        return pd.DataFrame(data)

    def preparer_donnees(self):
        symptomes_preprocesses = [
            self.preprocesseur.preprocesser_texte(s) 
            for s in self.df_medical['symptomes']
        ]
        self.symptomes_processed = symptomes_preprocesses
        self.matrice_tfidf = self.vectorizer.fit_transform(symptomes_preprocesses)

    def rechercher_similarite(self, symptomes_utilisateur, top_n=3):
        symptomes_preprocessed = self.preprocesseur.preprocesser_texte(symptomes_utilisateur)
        symptomes_vector = self.vectorizer.transform([symptomes_preprocessed])
        similarites = cosine_similarity(symptomes_vector, self.matrice_tfidf)
        indices_similaires = similarites.argsort()[0][-top_n:][::-1]

        resultats = []
        for idx in indices_similaires:
            similarite = similarites[0][idx]
            if similarite > 0.1:
                resultats.append({
                    'diagnostic': self.df_medical.iloc[idx]['diagnostics'],
                    'specialite': self.df_medical.iloc[idx]['specialites'],
                    'urgence': self.df_medical.iloc[idx]['urgence'],
                    'score': round(float(similarite), 3)
                })
        return resultats


# Instance globale
analyseur = AnalyseurSymptomes()
historique_conversations = {}


# ========== Routes API ==========

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérifier l'état du serveur"""
    return jsonify({"status": "ok", "message": "Chatbot médical opérationnel"})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint principal du chatbot"""
    try:
        data = request.json
        message_utilisateur = data.get('message', '')
        session_id = data.get('session_id', 'default')

        if not message_utilisateur:
            return jsonify({"error": "Message vide"}), 400

        # Analyse des symptômes
        resultats_similarite = analyseur.rechercher_similarite(message_utilisateur)

        # Construire le contexte pour Gemini
        contexte = f"""Tu es un assistant médical virtuel bienveillant et professionnel.

Message du patient: {message_utilisateur}

Analyse automatique des symptômes:
"""
        if resultats_similarite:
            for r in resultats_similarite[:2]:
                contexte += f"- Diagnostic possible: {r['diagnostic']} (confiance: {r['score']})\n"
                contexte += f"  Spécialité: {r['specialite']}, Urgence: {r['urgence']}\n"
        else:
            contexte += "Aucune correspondance trouvée dans la base de données.\n"

        contexte += """
Instructions:
1. Réponds de manière empathique et rassurante
2. Utilise l'analyse fournie mais ne la cite pas directement
3. Pose 1-2 questions de suivi pertinentes pour préciser les symptômes
4. Donne des conseils généraux appropriés
5. Rappelle l'importance de consulter un professionnel si nécessaire
6. Reste concis (max 150 mots)
7. N'établis JAMAIS de diagnostic définitif
"""

        # Appel à Gemini
        response = model.generate_content(contexte)
        reponse_chatbot = response.text

        # Sauvegarder l'historique
        if session_id not in historique_conversations:
            historique_conversations[session_id] = []
        
        historique_conversations[session_id].append({
            'utilisateur': message_utilisateur,
            'chatbot': reponse_chatbot,
            'analyse': resultats_similarite
        })

        return jsonify({
            "reponse": reponse_chatbot,
            "analyse": resultats_similarite,
            "session_id": session_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyse', methods=['POST'])
def analyse_symptomes():
    """Analyse pure des symptômes sans Gemini"""
    try:
        data = request.json
        symptomes = data.get('symptomes', '')

        if not symptomes:
            return jsonify({"error": "Symptômes manquants"}), 400

        resultats = analyseur.rechercher_similarite(symptomes, top_n=5)

        return jsonify({
            "symptomes": symptomes,
            "resultats": resultats,
            "nombre_resultats": len(resultats)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/historique/<session_id>', methods=['GET'])
def get_historique(session_id):
    """Récupérer l'historique d'une session"""
    historique = historique_conversations.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "historique": historique,
        "nombre_messages": len(historique)
    })


@app.route('/api/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    """Réinitialiser une session"""
    if session_id in historique_conversations:
        del historique_conversations[session_id]
    return jsonify({"message": "Session réinitialisée", "session_id": session_id})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Statistiques du système"""
    return jsonify({
        "nombre_conditions": len(analyseur.df_medical),
        "sessions_actives": len(historique_conversations),
        "total_messages": sum(len(h) for h in historique_conversations.values())
    })


if __name__ == '__main__':
    print("🏥 Chatbot Médical démarré sur http://localhost:5000")
    print("📊 Nombre de conditions médicales:", len(analyseur.df_medical))
    print("⚠️  N'oubliez pas de configurer votre clé API Gemini!")
    app.run(debug=True, host='0.0.0.0', port=5000)