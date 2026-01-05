import spacy
from rdflib import Graph, Namespace, RDFS, URIRef, Literal
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# ===============================
# 1️⃣ Setup SpaCy
# ===============================
try:
    nlp = spacy.load("en_core_web_md")
except:
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# ===============================
# 2️⃣ Setup NLTK
# ===============================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# ===============================
# 3️⃣ Load ontologies
# ===============================
g = Graph()
g.parse("ontologies/symp.owl")
g.parse("ontologies/doid.owl")

SYM = Namespace("http://purl.obolibrary.org/obo/SYMP_")
DOID = Namespace("http://purl.obolibrary.org/obo/DOID_")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_0000115")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# ===============================
# 4️⃣ Dynamic symptom extraction from ontology with descriptions
# ===============================
def load_symptoms_from_ontology():
    """
    Dynamically load symptoms from the SYMP ontology with descriptions
    Returns: 
        - symptoms_dict: dict with symptom labels -> canonical names
        - canonical_to_uri: dict with canonical names -> URIs
        - symptom_descriptions: dict with canonical names -> descriptions
    """
    symptoms_dict = {}
    synonyms_dict = {}
    canonical_to_uri = {}
    symptom_descriptions = {}
    
    query = """
    SELECT ?symptom ?label ?desc WHERE {
        ?symptom rdfs:label ?label .
        OPTIONAL { ?symptom obo:IAO_0000115 ?desc }
        FILTER(STRSTARTS(STR(?symptom), "http://purl.obolibrary.org/obo/SYMP_"))
    }
    """
    
    results = g.query(query, initNs={"obo": OBO, "IAO": IAO})
    for row in results:
        symptom_uri = str(row.symptom)
        label = str(row.label).lower().strip()
        description = str(row.desc) if row.desc else "No description available."
        
        canonical = symptom_uri.split('_')[-1].lower()
        
        canonical_to_uri[canonical] = symptom_uri
        symptom_descriptions[canonical] = description
        
        symptoms_dict[label] = canonical
        
        words = re.split(r'[-,\s]+', label)
        if len(words) > 1:
            for word in words:
                if len(word) > 3 and word not in STOPWORDS:  
                    symptoms_dict[word] = canonical
        
        syn_query = f"""
        SELECT ?synonym WHERE {{
            <{symptom_uri}> obo:hasExactSynonym ?synonym .
        }}
        """
        
        try:
            syn_results = g.query(syn_query, initNs={"obo": OBO})
            for syn_row in syn_results:
                synonym = str(syn_row.synonym).lower().strip()
                symptoms_dict[synonym] = canonical
                synonyms_dict[synonym] = canonical
        except:
            pass
        
        rel_query = f"""
        SELECT ?related WHERE {{
            <{symptom_uri}> obo:hasRelatedSynonym ?related .
        }}
        """
        
        try:
            rel_results = g.query(rel_query, initNs={"obo": OBO})
            for rel_row in rel_results:
                related = str(rel_row.related).lower().strip()
                symptoms_dict[related] = canonical
        except:
            pass
    
    COMMON_SYMPTOMS = {
        "fever": ["fever", "high temperature", "febrile", "pyrexia"],
        "cough": ["cough", "coughing", "tussis"],
        "fatigue": ["fatigue", "tired", "exhausted", "lethargy"],
        "headache": ["headache", "migraine", "head pain", "cephalalgia"],
        "nausea": ["nausea", "vomit", "feeling sick", "queasiness"],
        "pain": ["pain", "sore", "ache", "discomfort"],
        "bruise": ["bruise", "bruising", "contusion", "ecchymosis"]
    }
    
    for canonical, variations in COMMON_SYMPTOMS.items():
        for variation in variations:
            symptoms_dict[variation.lower()] = canonical
            if canonical not in symptom_descriptions:
                symptom_descriptions[canonical] = f"Symptom: {canonical.capitalize()}"
    
    return symptoms_dict, canonical_to_uri, symptom_descriptions

SYMPTOM_TO_CANON, CANONICAL_TO_URI, SYMPTOM_DESCRIPTIONS = load_symptoms_from_ontology()
SYMPTOM_WORDS = list(SYMPTOM_TO_CANON.keys())
MULTIWORD_SYMPTOMS = sorted([s for s in SYMPTOM_TO_CANON if " " in s], key=len, reverse=True)

# ===============================
# 5️⃣ Improved symptom extraction with ontology-aware matching
# ===============================
def extract_symptoms(text):
    text = text.lower().strip()
    found = set()
    
    doc = nlp(text)
    
    # First pass: check for multi-word symptoms (longest first)
    for phrase in MULTIWORD_SYMPTOMS:
        if phrase in text:
            context_start = max(0, text.find(phrase) - 50)
            context_end = min(len(text), text.find(phrase) + len(phrase) + 50)
            context = text[context_start:context_end]
            
            if not is_symptom_negated(context, phrase):
                found.add(SYMPTOM_TO_CANON[phrase])
    
    # Second pass: check for single-word symptoms
    for token in doc:
        token_text = token.text.lower()
        
        # Skip stopwords, punctuation, and short words
        if (token_text in STOPWORDS or 
            token.is_punct or 
            len(token_text) < 3 or
            token_text in ['no', 'not', 'without', 'negative']):
            continue
        
        if token_text in SYMPTOM_TO_CANON and token_text not in found:
            # Get context around the token
            start = max(0, token.idx - 50)
            end = min(len(text), token.idx + len(token_text) + 50)
            context = text[start:end]
            
            if not is_symptom_negated(context, token_text):
                found.add(SYMPTOM_TO_CANON[token_text])
    
    # Third pass: lemmatize and check again
    lemmatized_words = [lemmatizer.lemmatize(token.text.lower()) for token in doc 
                       if token.text.lower() not in STOPWORDS and not token.is_punct]
    
    for lemma in lemmatized_words:
        if lemma in SYMPTOM_TO_CANON and lemma not in found:
            original_tokens = [t for t in doc if lemmatizer.lemmatize(t.text.lower()) == lemma]
            if original_tokens:
                token = original_tokens[0]
                start = max(0, token.idx - 50)
                end = min(len(text), token.idx + len(token.text) + 50)
                context = text[start:end]
                
                if not is_symptom_negated(context, lemma):
                    found.add(SYMPTOM_TO_CANON[lemma])
    
    return list(found) if found else []

def is_symptom_negated(sentence, symptom):
    """
    Check if a symptom is negated in the sentence using multiple strategies.
    """
    negation_words = [
        'no', 'not', "n't", 'none', 'never', 'without', 'negative', 
        'denies', 'denied', 'denying', 'free', 'free of', 'absence of', 
        'lack of', 'ruled out', 'rule out', 'r/o', 'absent', 'negative for',
        'does not have', 'do not have', 'did not have', 'has no', 'have no',
        'shows no', 'showed no', 'displaying no', 'exhibits no'
    ]
    
    symptom_pos = sentence.find(symptom)
    if symptom_pos == -1:
        return False
    
    text_before = sentence[:symptom_pos].strip()
    words_before = text_before.split()
    
    for i in range(max(0, len(words_before) - 4), len(words_before)):
        if words_before[i] in negation_words:
            return True
    
    patterns = [
        f"no {symptom}",
        f"not {symptom}",
        f"without {symptom}",
        f"negative for {symptom}",
        f"absence of {symptom}",
        f"lack of {symptom}",
        f"free of {symptom}",
        f"free from {symptom}",
        f"denies {symptom}",
        f"denied {symptom}",
        f"does not have {symptom}",
        f"do not have {symptom}",
        f"has no {symptom}",
        f"have no {symptom}",
    ]
    
    for pattern in patterns:
        if pattern in sentence:
            return True
    
    symptom_start = symptom_pos
    symptom_end = symptom_pos + len(symptom)
    
    sent_doc = nlp(sentence)
    
    for token in sent_doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        if (token_start <= symptom_end and token_end >= symptom_start):
            for ancestor in token.ancestors:
                if ancestor.text.lower() in negation_words:
                    return True
                if ancestor.dep_ in ["neg", "det"] and ancestor.text.lower() == "no":
                    return True
    
    return False

# ===============================
# 6️⃣ Enhanced ontology query with dynamic symptom matching
# ===============================
def query_ontology(symptoms):
    results = {}
    
    for symptom_canonical in symptoms:
        symptom_uri = None
        if symptom_canonical in CANONICAL_TO_URI:
            symptom_uri = CANONICAL_TO_URI[symptom_canonical]
        else:
            query_uri = f"""
            SELECT ?uri WHERE {{
                ?uri rdfs:label ?label .
                FILTER(LCASE(STR(?label)) = "{symptom_canonical}" || 
                       CONTAINS(LCASE(STR(?label)), "{symptom_canonical}"))
                FILTER(STRSTARTS(STR(?uri), "http://purl.obolibrary.org/obo/SYMP_"))
            }}
            LIMIT 1
            """
            uri_results = list(g.query(query_uri))
            if uri_results:
                symptom_uri = str(uri_results[0].uri)
        
        if not symptom_uri:
            sym_query = f"""
            SELECT ?symptom ?label ?desc ?diseaseLabel ?diseaseDesc WHERE {{
                ?symptom rdfs:label ?label .
                OPTIONAL {{ ?symptom obo:IAO_0000115 ?desc }}
                ?disease rdfs:subClassOf ?symptom .
                ?disease rdfs:label ?diseaseLabel .
                OPTIONAL {{ ?disease obo:IAO_0000115 ?diseaseDesc }}
                FILTER(CONTAINS(LCASE(STR(?label)), "{symptom_canonical}") ||
                       ?symptom obo:hasExactSynonym ?syn .
                       FILTER(CONTAINS(LCASE(STR(?syn)), "{symptom_canonical}")))
            }}
            """
        else:
            sym_query = f"""
            SELECT ?symptom ?label ?desc ?diseaseLabel ?diseaseDesc WHERE {{
                <{symptom_uri}> rdfs:label ?label .
                OPTIONAL {{ <{symptom_uri}> obo:IAO_0000115 ?desc }}
                ?disease rdfs:subClassOf <{symptom_uri}> .
                ?disease rdfs:label ?diseaseLabel .
                OPTIONAL {{ ?disease obo:IAO_0000115 ?diseaseDesc }}
            }}
            """
        
        try:
            for row in g.query(sym_query, initNs={
                "rdfs": RDFS, 
                "obo": OBO,
                "IAO": IAO
            }):
                disease_name = str(row.diseaseLabel)
                symptom_label = str(row.label)
                symptom_description = str(row.desc) if row.desc else "No description available."
                disease_description = str(row.diseaseDesc) if row.diseaseDesc else "No description available."

                if disease_name not in results:
                    results[disease_name] = {
                        "score": 1,
                        "matched": [symptom_label],
                        "symptom_descriptions": [symptom_description],
                        "disease_description": disease_description
                    }
                else:
                    results[disease_name]["score"] += 1
                    if symptom_label not in results[disease_name]["matched"]:
                        results[disease_name]["matched"].append(symptom_label)
                        results[disease_name]["symptom_descriptions"].append(symptom_description)
        except Exception as e:
            print(f"Query error for {symptom_canonical}: {e}")
            continue
    
    return results

# ===============================
# 7️⃣ Format HTML response WITH symptom descriptions
# ===============================
def format_response(symptoms, diseases):
    html = "<h3>🩺 Detected Symptoms:</h3><ul>"
    if symptoms:
        symptom_details = []
        for symptom in symptoms:
            readable_query = f"""
            SELECT ?label ?desc WHERE {{
                ?s rdfs:label ?label .
                OPTIONAL {{ ?s obo:IAO_0000115 ?desc }}
                FILTER(STRENDS(STR(?s), "_{symptom.upper()}") ||
                       CONTAINS(LCASE(STR(?label)), "{symptom}"))
                FILTER(STRSTARTS(STR(?s), "http://purl.obolibrary.org/obo/SYMP_"))
            }}
            LIMIT 1
            """
            try:
                label_results = list(g.query(readable_query, initNs={"obo": OBO, "IAO": IAO}))
                if label_results:
                    symptom_name = str(label_results[0].label)
                    symptom_desc = str(label_results[0].desc) if label_results[0].desc else SYMPTOM_DESCRIPTIONS.get(symptom, "No description available.")
                    symptom_details.append((symptom_name, symptom_desc))
                else:
                    symptom_name = symptom.capitalize()
                    symptom_desc = SYMPTOM_DESCRIPTIONS.get(symptom, "No description available.")
                    symptom_details.append((symptom_name, symptom_desc))
            except:
                symptom_name = symptom.capitalize()
                symptom_desc = SYMPTOM_DESCRIPTIONS.get(symptom, "No description available.")
                symptom_details.append((symptom_name, symptom_desc))
        
        for symptom_name, symptom_desc in symptom_details:
            html += f"<li><b>{symptom_name}</b>: {symptom_desc}</li>"
    else:
        html += "<li>No symptoms detected or all symptoms were negated</li>"
    html += "</ul>"

    if not diseases:
        html += "<h3>No conditions found. Try providing more symptoms.</h3>"
        return html

    html += "<h3>🔎 Possible Conditions:</h3><ul>"
    for name, info in sorted(diseases.items(), key=lambda x: -x[1]["score"]):
        matched = ", ".join(set(info["matched"]))  
        disease_desc = info["disease_description"]
        
        symptom_descs = []
        for symptom, desc in zip(info["matched"], info["symptom_descriptions"]):
            symptom_descs.append(f"<b>{symptom}</b>: {desc}")
        symptom_descs_text = "<br>".join(set(symptom_descs))
        
        html += f"<li><b>{name}</b> (score: {info['score']})<br>"
        html += f"<i>Disease description: {disease_desc}</i><br>"
        html += f"<i>Matched symptoms:</i><br>{symptom_descs_text}</li><br>"
    html += "</ul>"
    html += "<p>⚠️ This is informational only. Consult a healthcare professional for advice.</p>"
    return html

# ===============================
# 8️⃣ Main chatbot function
# ===============================
def chatbot_response(user_input):
    symptoms = extract_symptoms(user_input)
    print(f"Extracted symptoms: {symptoms}") 
    diseases = query_ontology(symptoms)
    return format_response(symptoms, diseases)

# ===============================
# 9️⃣ Helper function to explore ontology
# ===============================
def explore_ontology_stats():
    """Print statistics about loaded ontology"""
    query = """
    SELECT (COUNT(?symptom) as ?count) WHERE {
        ?symptom rdfs:label ?label .
        FILTER(STRSTARTS(STR(?symptom), "http://purl.obolibrary.org/obo/SYMP_"))
    }
    """
    result = list(g.query(query))
    print(f"Total symptoms in ontology: {result[0][0] if result else 0}")
    
    print(f"Loaded symptom variations: {len(SYMPTOM_TO_CANON)}")
    print(f"Multi-word symptoms: {len(MULTIWORD_SYMPTOMS)}")
    print(f"Symptoms with descriptions: {len(SYMPTOM_DESCRIPTIONS)}")
    
    print("\nSample symptoms with descriptions:")
    sample_canonicals = list(SYMPTOM_DESCRIPTIONS.keys())[:5]
    for canonical in sample_canonicals:
        for term, can in SYMPTOM_TO_CANON.items():
            if can == canonical and " " not in term:  # Use single-word term if possible
                print(f"  {term} -> {canonical}")
                print(f"    Description: {SYMPTOM_DESCRIPTIONS[canonical][:100]}...")
                break

