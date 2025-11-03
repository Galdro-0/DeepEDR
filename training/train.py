import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Importation de nos modules
from utils.preprocessing import get_data_loaders, MAX_SEQ_LENGTH
from models.edr_model import LSTMEDR

# Configuration (Hyperparamètres)
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4
NUM_CLASSES = 4 # 0, 1, 2, 3
LEARNING_RATE = 0.001
NUM_EPOCHS = 15 # 15-20 époques suffisent souvent pour ce type de tâche
BATCH_SIZE = 64
MODEL_SAVE_PATH = 'models/best_model.pth'

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """Génère les graphiques de perte et de précision."""
    
    # Graphique de Perte
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Perte (Loss) par Époque')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)
    
    # Graphique de Précision
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Précision (Accuracy) par Époque')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training/performance_graphs.png')
    print("Graphiques de performance sauvegardés dans 'training/performance_graphs.png'")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Génère la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie valeur')
    plt.savefig('training/confusion_matrix.png')
    print("Matrice de confusion sauvegardée dans 'training/confusion_matrix.png'")

def train_model():
    # 0. Vérifier le GPU (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du périphérique : {device}")
    
    # 1. Chargement des données
    if not os.path.exists('data/generated_logs.csv'):
        print("Erreur: 'data/generated_logs.csv' non trouvé.")
        print("Veuillez d'abord lancer 'python utils/data_generator.py'")
        return
        
    train_loader, val_loader, vocab = get_data_loaders(BATCH_SIZE)
    vocab_size = vocab.vocab_size
    
    # 2. Initialisation du modèle, de la perte et de l'optimiseur
    model = LSTMEDR(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Pour le suivi des métriques
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    print("Début de l'entraînement...")
    
    # 3. Boucle d'entraînement
    for epoch in range(NUM_EPOCHS):
        model.train() # Mode entraînement
        train_loss, train_correct = 0, 0
        total_train = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass et optimisation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / total_train
        
        # 4. Boucle de validation
        model.eval() # Mode évaluation
        val_loss, val_correct = 0, 0
        total_val = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / total_val
        
        # Sauvegarde des métriques
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f'Époque [{epoch+1}/{NUM_EPOCHS}] - '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        
        # 5. Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Meilleur modèle sauvegardé dans {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

    print("Entraînement terminé.")
    
    # 6. Évaluation finale et graphiques
    print("\nÉvaluation finale sur le set de validation :")
    class_names = ['Normal', 'Recon', 'Execution', 'Ransomware']
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Créer le répertoire 'training' s'il n'existe pas pour les graphes
    if not os.path.exists('training'):
        os.makedirs('training')
        
    plot_metrics(history['train_loss'], history['val_loss'], 
                 history['train_acc'], history['val_acc'])
    
    plot_confusion_matrix(all_labels, all_preds, class_names)

if __name__ == "__main__":
    train_model()