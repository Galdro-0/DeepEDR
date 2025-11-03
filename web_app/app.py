import streamlit as st
import torch
import json
import os
import datetime

# Importation des modules du projet
from models.edr_model import LSTMEDR
from utils.preprocessing import Vocab, tokenize_sequence, MAX_SEQ_LENGTH

# Configuration
MODEL_PATH = 'models/best_model.pth'
TOKENIZER_PATH = 'models/tokenizer.json'
ALERT_FILE = 'web_app/alert.json'

# Hyperparamètres du modèle (doivent correspondre à l'entraînement)
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 4

# Mapping des labels
CLASS_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Exécution de Payload",
    3: "Action de Ransomware"
}

@st.cache_resource
def load_model_and_vocab():
    """Charge le modèle et le vocabulaire en cache."""
    
    # 1. Charger le vocabulaire
    try:
        vocab = Vocab.load(TOKENIZER_PATH)
    except FileNotFoundError:
        st.error(f"Erreur: Fichier tokenizer non trouvé à {TOKENIZER_PATH}")
        st.stop()
        
    vocab_size = vocab.vocab_size
    
    # 2. Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMEDR(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        st.error(f"Erreur: Fichier modèle non trouvé à {MODEL_PATH}")
        st.info("Avez-vous lancé 'python training/train.py' ?")
        st.stop()
        
    model.to(device)
    model.eval() # Mode évaluation
    
    return model, vocab, device

def preprocess_input(sequence_str, vocab, max_len, device):
    """Prétraite la séquence d'entrée pour le modèle."""
    tokens = tokenize_sequence(sequence_str)
    
    indices = [vocab.word_to_idx.get(t, vocab.word_to_idx['<UNK>']) for t in tokens]
    
    # Padding / Truncation
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        padding_len = max_len - len(indices)
        indices = indices + [vocab.word_to_idx['<PAD>']] * padding_len
        
    # Conversion en Tensor et ajout d'une dimension batch (batch_size=1)
    tensor = torch.tensor(indices).unsqueeze(0).to(device)
    return tensor

def log_alert(sequence, prediction):
    """(Optionnel) Enregistre les détections dans un fichier JSON."""
    alert = {
        'timestamp': datetime.datetime.now().isoformat(),
        'sequence': sequence,
        'detected_threat': prediction,
        'level': 'CRITICAL' if prediction == CLASS_NAMES[3] else 'HIGH'
    }
    
    alerts = []
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE, 'r') as f:
                alerts = json.load(f)
        except json.JSONDecodeError:
            alerts = []
            
    alerts.insert(0, alert) # Ajoute la nouvelle alerte en haut
    
    with open(ALERT_FILE, 'w') as f:
        json.dump(alerts, f, indent=4)

def predict_sequence(model, vocab, device, sequence_str):
    """Effectue une prédiction complète."""
    
    # 1. Prétraitement
    input_tensor = preprocess_input(sequence_str, vocab, MAX_SEQ_LENGTH, device)
    
    # 2. Prédiction
    with torch.no_grad():
        logits = model(input_tensor)
        
        # Obtenir les probabilités (softmax) et la classe prédite
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        pred_label_idx = predicted_idx.item()
        pred_label_name = CLASS_NAMES[pred_label_idx]
        confidence_score = confidence.item()
        
        return pred_label_name, confidence_score

# --- Interface Streamlit ---

st.set_page_config(page_title="DeepEDR", layout="wide")
st.title("🛡️ DeepEDR - Prototype de Détection d'Anomalies")
st.markdown("Analyse de séquences de commandes par Deep Learning (LSTM)")

# Chargement des artefacts
model, vocab, device = load_model_and_vocab()
st.sidebar.success(f"Modèle chargé sur {device}. Vocabulaire: {vocab.vocab_size} tokens.")

# Zone de test principale
st.subheader("🧪 Analyse en temps réel")
default_sequence = "whoami;systeminfo;net user"
sequence_input = st.text_area(
    "Entrez une séquence de commandes (séparées par ';'):", 
    default_sequence,
    height=100
)

if st.button("Analyser la séquence", type="primary"):
    if sequence_input:
        prediction, confidence = predict_sequence(model, vocab, device, sequence_input)
        
        st.markdown("---")
        st.subheader("Résultat de l'analyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Détection", value=prediction)
        with col2:
            st.metric(label="Confiance", value=f"{confidence*100:.2f}%")
        
        if prediction != CLASS_NAMES[0]: # Si ce n'est pas "Normal"
            st.error(f"**ALERTE DE SÉCURITÉ !** Comportement suspect détecté.")
            # Log de l'alerte
            log_alert(sequence_input, prediction)
            st.toast(f"Alerte '{prediction}' enregistrée!")
        else:
            st.success("**Comportement jugé normal.** Aucune menace détectée.")
            
    else:
        st.warning("Veuillez entrer une séquence à analyser.")

# Affichage du journal d'alertes (Optionnel)
st.markdown("---")
st.subheader("🚨 Journal des Alertes Récentes")
if os.path.exists(ALERT_FILE):
    with open(ALERT_FILE, 'r') as f:
        try:
            alerts = json.load(f)
            st.dataframe(alerts, use_container_width=True)
        except json.JSONDecodeError:
            st.info("Le journal d'alertes est vide ou corrompu.")
else:
    st.info("Aucune alerte enregistrée pour le moment.")