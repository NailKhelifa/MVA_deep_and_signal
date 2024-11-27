import numpy as np
import torch 
import h5py # pour gérer les formats de données utilisés ici 
import torch
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_labels(open_h5_file): 
    return {
        open_h5_file['label_name'].attrs[k] : k
        for k in open_h5_file['label_name'].attrs.keys()
    }

# Fonction pour calculer des statistiques par classe
def statistiques_par_classe(signal_bruit, etiquettes_id):
    """
    Calcule et affiche des statistiques (moyenne, max, min, écart type, et nombre d'échantillons)
    pour chaque classe identifiée dans les étiquettes.

    Paramètres :
    - signal_bruit : tableau contenant les rapports signal-bruit (SNR).
    - etiquettes_id : tableau contenant les identifiants des classes.

    Retour :
    - Affiche les statistiques par classe dans la console.
    """
    # Récupérer les classes uniques dans les étiquettes
    classes = np.unique(etiquettes_id)

    # Parcourir chaque classe
    for classe in classes:
        # Extraire les valeurs SNR correspondant à la classe actuelle
        snr_classe = signal_bruit[etiquettes_id == classe]

        # Calculer les statistiques
        moyenne_snr = np.mean(snr_classe)  # Moyenne
        max_snr = np.max(snr_classe)      # Maximum
        min_snr = np.min(snr_classe)      # Minimum
        ecart_type_snr = np.std(snr_classe)  # Écart type
        nombre_echantillons = len(snr_classe)  # Nombre d'échantillons

        # Afficher les résultats pour la classe actuelle
        print(f"Classe {classe} - Moyenne: {moyenne_snr:.2f}, Max: {max_snr:.2f}, "
              f"Min: {min_snr:.2f}, Écart-type: {ecart_type_snr:.2f}, "
              f"Nombre d'échantillons: {nombre_echantillons}")

# Fonction pour tracer les dimensions réelle et complexe des signaux par classe
def tracer_reelle_vs_complexe_par_classe(signaux, etiquettes_id, k=2):
    """
    Trace les dimensions réelle et complexe des signaux pour chaque classe, 
    en sélectionnant aléatoirement un sous-ensemble de signaux.

    Paramètres :
    - signaux : tableau contenant les signaux (dimensions réelle et complexe).
    - etiquettes_id : tableau contenant les identifiants des classes.
    - k : nombre de signaux à sélectionner aléatoirement par classe (par défaut : 2).

    Retour :
    - Affiche un graphique pour chaque classe avec les points représentant les signaux.
    """
    # Récupérer les classes uniques dans les étiquettes
    classes = np.unique(etiquettes_id)

    # Définir le nombre de subplots par ligne
    n_cols = 3  # Nombre de colonnes de subplots
    n_rows = (len(classes) + n_cols - 1) // n_cols  # Calculer le nombre de lignes nécessaires

    # Créer une figure et une grille de subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))

    # Aplatir les axes pour un accès plus facile
    axes = axes.flatten()

    # Parcourir chaque classe
    for idx, classe in enumerate(classes):
        # Indices des signaux appartenant à la classe actuelle
        indices_classe = np.where(etiquettes_id == classe)[0]

        # Sélectionner aléatoirement jusqu'à k signaux de la classe
        indices_selectionnes = np.random.choice(
            indices_classe, 
            size=min(k, len(indices_classe)), 
            replace=False
        )

        # Choisir l'axe pour cette classe
        ax = axes[idx]

        # Tracer les signaux sélectionnés
        for i in indices_selectionnes:
            partie_reelle = signaux[i, :, 0]  # Première dimension : réelle
            partie_complexe = signaux[i, :, 1]  # Deuxième dimension : complexe
            ax.scatter(partie_reelle, partie_complexe, s=10, alpha=0.2)

        # Ajouter des labels et un titre
        ax.set_xlabel('Partie réelle')
        ax.set_ylabel('Partie complexe')
        ax.set_title(f'Classe {classe} - Réelle vs Complexe')

        # Ajouter une légende (si nécessaire)
        ax.legend(loc="upper right", fontsize='small')

    # Ajuster la disposition pour que les subplots ne se chevauchent pas
    plt.tight_layout()

    # Afficher la figure
    plt.show()

# Fonction pour tracer les signaux par classe
def tracer_signaux_par_classe(signaux, etiquettes_id, k=8):
    """
    Trace les signaux d'un échantillon par classe, en séparant les parties réelle et complexe.

    Paramètres :
    - signaux : tableau contenant les signaux (avec des dimensions réelle et complexe).
    - etiquettes_id : tableau contenant les identifiants des classes.
    - k : nombre de signaux à sélectionner aléatoirement par classe (par défaut : 8).

    Retour :
    - Affiche des graphiques avec les parties réelle et complexe des signaux pour chaque classe.
    """

    classes = np.unique(etiquettes_id)
    premier_titre = True 

    couleur_reelle = 'blue'
    couleur_complexe = 'red'


    for classe in classes:
        indices_classe = np.where(etiquettes_id == classe)[0]

        # Sélectionner aléatoirement jusqu'à k signaux
        indices_selectionnes = np.random.choice(
            indices_classe, 
            size=min(k, len(indices_classe)), 
            replace=False
        )

        # Créer une figure avec deux sous-graphes
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        # Tracer les signaux sélectionnés
        for i in indices_selectionnes:
            # Partie réelle (en bleu)
            axes[0].plot(signaux[i, :, 0], alpha=0.5, color=couleur_reelle)  
            # Partie complexe (en rouge)
            axes[1].plot(signaux[i, :, 1], alpha=0.5, color=couleur_complexe)  

        # Ajouter les titres aux sous-graphes 
        if premier_titre:
            axes[0].set_title('Partie réelle')
            axes[1].set_title('Partie complexe')
            premier_titre = False

        # Ajouter un titre général pour cette classe
        fig.suptitle(f'Classe {classe} - Signaux')

        # Afficher le graphique
        plt.show()

# Fonction pour tracer la Transformée de Fourier des signaux par classe
def tracer_fourier_signaux(signaux, etiquettes_id, k=1):
    """
    Trace la Transformée de Fourier des signaux pour chaque classe.

    Paramètres :
    - signaux : tableau contenant les signaux (dimensions réelle et complexe).
    - etiquettes_id : tableau contenant les identifiants des classes.
    - k : nombre de signaux à sélectionner aléatoirement par classe (par défaut : 1).

    Retour :
    - Affiche des graphiques des magnitudes de la Transformée de Fourier pour les signaux sélectionnés.
    """
    classes = np.unique(etiquettes_id)
    couleur_signal = 'green'  


    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 3 * n_classes))  

    if n_classes == 1:
        axes = [axes]


    for i, etiquette in enumerate(classes):
        # Trouver les indices des signaux appartenant à cette classe
        indices_classe = np.where(etiquettes_id == etiquette)[0]
        
        # Sélectionner aléatoirement jusqu'à k signaux de cette classe
        indices_selectionnes = np.random.choice(indices_classe, k, replace=False)
        
        # Tracer les signaux sélectionnés sur le sous-graphe correspondant
        for _, idx in enumerate(indices_selectionnes):
            # Séparer les parties réelle et imaginaire
            partie_reelle = signaux[idx, :, 0]
            partie_imaginaire = signaux[idx, :, 1]
            
            # Construire le signal complexe
            signal_complexe = partie_reelle + 1j * partie_imaginaire
            
            # Calculer la Transformée de Fourier
            fft_signal = np.fft.fft(signal_complexe)
            freqs = np.fft.fftfreq(len(partie_reelle), d=1)  # Fréquences associées
            
            # Calculer la magnitude
            magnitude = np.abs(fft_signal)
            
            # Tracer la magnitude en fonction des fréquences
            axes[i].plot(freqs, magnitude, alpha=0.5, color=couleur_signal, label=f"Signal {idx}")

        # Ajouter les titres et les labels pour chaque sous-graphe
        axes[i].set_title(f"Transformée de Fourier des signaux - Classe {etiquette}")
        axes[i].set_xlabel("Fréquence")
        axes[i].set_ylabel("Magnitude")
        axes[i].grid(True)

    # Ajuster la disposition des subplots
    plt.tight_layout()

    # Afficher la figure
    plt.show()

class DumbModel(nn.Module):  # nn.Module permet d'hériter de nombreuses fonctionnalités pratiques, par exemple si on appelle le modèle
                            # avec model(sample), cela appelle automatiquement model.forward(sample)
    def __init__(self, input_size=2048, num_classes=6, out_channels=8, kernel_size=3, stride=5):
        super(DumbModel, self).__init__()  # Initialisation du module parent nn.Module
        self.kernel_size = kernel_size  # Taille du noyau de la convolution
        self.stride = stride  # Pas de la convolution

        # Définition de la première couche de convolution 1D
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=out_channels, kernel_size=self.kernel_size, stride=self.stride)
        
        # Définition de la couche de pooling max 1D avec un noyau de taille 2
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Calcul "à la main" de la dimension de sortie de la convolution après le maxpooling
        conv_output_size = ((input_size - self.kernel_size) // self.stride) // 2  + (stride % 2)  # Calcul de la taille après convolution et pooling
        
        # Définition de la couche entièrement connectée (fully connected) qui prend la sortie aplatie de la convolution
        self.fc1 = nn.Linear(out_channels * conv_output_size, num_classes)

        # Fonction d'activation LogSoftmax pour la sortie
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Passage à travers la couche convolution, puis activation ReLU et pooling
        x = self.pool(torch.relu(self.conv1(x)))
        
        # Aplatissement de la sortie pour l'entrée dans la couche fully connected
        x = x.view(x.size(0), -1)
        
        # Passage à travers la couche fully connected
        x = self.fc1(x)
        
        # Retour des résultats après l'application de LogSoftmax pour la classification
        return self.out(x)
    
# Définition d'un Dataset personnalisé
class MyDataset(torch.utils.data.Dataset):
    """
    Classe personnalisée pour gérer un ensemble de données à partir d'un fichier HDF5.
    """

    def __init__(self, chemin_vers_donnees):
        """
        Initialise le jeu de données en chargeant les signaux, les SNR et les étiquettes.

        Paramètres :
        - chemin_vers_donnees : chemin vers le fichier HDF5 contenant les données.
        """
        # Chargement des données à partir d'un fichier HDF5
        donnees = h5py.File(chemin_vers_donnees, 'r')
        self.signaux = np.array(donnees['signaux']).transpose(0, 2, 1)  # Transpose pour le bon format
        self.snr = np.array(donnees['snr'])  # SNR (Rapport Signal-Bruit)
        self.etiquettes_id = np.array(donnees['labels'])  # Identifiants des étiquettes
        self.noms_etiquettes = get_labels(donnees)  # Récupération des noms des étiquettes (fonction externe)
        donnees.close()  # Fermeture du fichier pour libérer les ressources

    def __len__(self):
        """
        Retourne la taille du jeu de données.
        """
        return self.signaux.shape[0]

    def __getitem__(self, index):
        """
        Retourne un échantillon à l'indice donné.

        Paramètre :
        - i : indice de l'échantillon à récupérer.

        Retour :
        - Un tuple contenant (signal, SNR, étiquette_id) pour l'échantillon spécifié.
        """
        return (self.signaux[index], self.snr[index], self.etiquettes_id[index])
    

class SimpleModelTrainer(object):
    """
    Classe pour entraîner et tester un modèle de réseau neuronal.
    Elle prend en charge l'entraînement, l'évaluation sur un jeu de validation, 
    l'arrêt anticipé basé sur la performance du modèle, et la sauvegarde/chargement du modèle.
    """
    
    def __init__(self, model, verbose=True, device="cpu"):
        self.model = model
        self.verbose = verbose  # Permet d'afficher des informations détaillées pendant l'entraînement
        self.device = device  # Spécifie l'appareil (CPU ou GPU) pour l'entraînement
        self.train_loss = []
        self.accuracy_test = []
        self.test_loss = []

    def fit(self, n_epochs=100, path_to_data=False, dataloader=None, batch_size=32, lr=1e-5, valid_loader=None,
            critic_test=5, criterion=nn.NLLLoss(), model_path="model.pth", patience=5):
        """
        Entraîne le modèle sur les données d'entraînement, en effectuant des tests réguliers sur les données de validation.
        Si la performance du modèle ne s'améliore pas pendant 'patience' epochs, l'entraînement s'arrête.
        """
        self.critic_test = critic_test
        if not path_to_data and dataloader is None:
            raise ValueError("Please insert a dataloader or a path to the dataset")
        if dataloader is None:
            dataloader = DataLoader(MyDataset(path_to_data), shuffle=True, batch_size=batch_size)
        
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # Optimiseur Adam
        initial_loss = 0
        test_loss = 0
        count_patience = 0

        with tqdm(total=n_epochs, desc=f"Epoch 0/{n_epochs} - Train Loss: {initial_loss:.4f} - Test Loss: {test_loss:.4f}", leave=True) as epoch_bar:
            for epoch in range(n_epochs):
                epoch_loss = 0  # Calcul de la perte pour l'epoch
                self.model.train()          
                for signals, _, labels in dataloader:
                    signals, labels = signals.to(self.device), labels.to(self.device).long()

                    optimizer.zero_grad()  # Remise à zéro des gradients avant chaque itération

                    outputs = self.model(signals)
                    loss = criterion(outputs, labels)
                    loss.backward()  # Calcul des gradients

                    optimizer.step()  # Mise à jour des poids du modèle
                    epoch_loss += loss.item()  # Accumulation de la perte de l'epoch
                
                avg_loss = epoch_loss / len(dataloader)
                epoch_bar.set_description(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_loss:.4f} - Test Loss: {test_loss:.4f}")
                epoch_bar.update(1)
                self.train_loss.append(avg_loss)

                # Test du modèle sur les données de validation à intervalles réguliers
                if valid_loader is not None and epoch % critic_test == 1:
                    test_labels, test_preds, t_loss = self._test(valid_loader, training=True, criterion=criterion)
                    test_loss = t_loss / len(valid_loader)

                    if test_loss < min(self.test_loss) if self.test_loss else np.inf:
                        self._save_model(model_path, verbose=False)  # Sauvegarde du modèle si amélioration
                        count_patience = 0 
                    count_patience += 1
                    self.test_loss.append(test_loss)
                    self.accuracy_test.append(accuracy_score(test_labels, test_preds))

                # Arrêt anticipé si la perte de validation ne s'améliore pas
                if count_patience > patience:
                    break 
        
        self._load_model(model_path, return_model=False, verbose=False)

    def _test(self, dataloader, training=False, criterion=nn.NLLLoss(), by_snr=False):
        """
        Teste le modèle sur les données du dataloader.
        Renvoie les prédictions et la perte, ainsi qu'une évaluation par SNR si spécifié.
        """
        self.model.eval()
        if by_snr:
            all_labels = {f"{i*10}": [] for i in range(4)}
            all_preds = {f"{i*10}": [] for i in range(4)}
        else:
            all_labels = []
            all_preds = [] 
        test_loss = 0           
        with torch.no_grad():  # Désactive le calcul des gradients pendant le test
            for signals, snr, labels in dataloader if training else tqdm(dataloader):
                signals, labels = signals.to(self.device), labels.to(self.device).long()

                outputs = self.model(signals)
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                if by_snr:
                    for s, label in zip(snr, labels):
                        all_labels[str(int(s.cpu().numpy()))].append(int(label.cpu().numpy()))
                    for s, pred in zip(snr, preds):
                        all_preds[str(int(s.cpu().numpy()))].append(int(pred.cpu().numpy()))
                    continue
                else:
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

        if by_snr:
            return (all_labels, all_preds)

        report = classification_report(all_labels, all_preds, zero_division=0)
        
        if training:
            return (all_labels, all_preds, test_loss)
        else:
            print(report)
            return (all_labels, all_preds)

    def plot_loss(self):
        """
        Affiche un graphique de la perte d'entraînement et de validation au fil des epochs.
        """
        epochs = range(1, len(self.train_loss) + 1)
        test_epochs = range(self.critic_test, self.critic_test * len(self.test_loss) + 1, self.critic_test) 

        plt.figure(figsize=(10, 6))

        plt.plot(epochs, self.train_loss, label='Train Loss', color='blue')
        plt.plot(test_epochs, self.test_loss, label='Test Loss', color='red')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss per Epoch')
        plt.legend()
        plt.show()

    def full_proc(self, train_dataloader, valid_dataloader, test_dataloader, n_epochs=100, lr=1e-5,
                  critic_test=5, criterion=nn.NLLLoss(), model_path="model.pth"):
        """
        Procédure complète d'entraînement du modèle, validation et test avec affichage des pertes.
        """
        self.fit(dataloader=train_dataloader, valid_loader=valid_dataloader,
                 n_epochs=n_epochs, lr=lr, critic_test=critic_test, criterion=criterion,
                 model_path=model_path)
        
        self.plot_loss()

        lab, preds = self._test(test_dataloader)

        return accuracy_score(lab, preds)

    def _save_model(self, path="model.pth", verbose=True):
        """
        Sauvegarde les poids du modèle à l'emplacement spécifié.
        """
        torch.save(self.model.state_dict(), path)
        if verbose:
            print(f"Model saved @ {path}")

    def _load_model(self, path="model.pth", verbose=True, return_model=False):
        """
        Charge les poids du modèle depuis un fichier.
        """
        self.model.load_state_dict(torch.load(path))
        if verbose:
            print("Model successfully loaded")

        if return_model:
            return self.model


class Model2ConvNonLinearFC(nn.Module):
    def __init__(self, N=6, C=2, T=2048):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=C, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        
        self.fc = nn.Linear(16 * (T // 16), 128)  # Layer FC with 128 neurons
        
        self.fc_non_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, N)  # Output layer with N classes
        )
        
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = self.fc_non_linear(x)  # Apply non-linear activation to FC output
        return self.out(x)


class ModelDilatedConvLSTM(nn.Module):
    def __init__(self, N=6, C=2, T=2048):
        super().__init__()

        # Convolutions dilatées
        self.dilated_conv1 = nn.Conv1d(in_channels=C, out_channels=8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=4, dilation=4)
        
        # LSTM pour capturer les dépendances temporelles
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=2, batch_first=True)

        # Couche fully connected avec activation non linéaire
        self.fc = nn.Linear(64, 128)
        self.fc_non_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, N)  # Sortie avec N classes
        )
        
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Appliquer les convolutions dilatées
        x = self.dilated_conv1(x)
        x = self.dilated_conv2(x)
        
        # Transformer les données pour LSTM (dimensions : [batch_size, time_steps, features])
        x = x.transpose(1, 2)  # Changer de [batch, features, time_steps] à [batch, time_steps, features]
        
        # Appliquer LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Utiliser la dernière sortie cachée du LSTM
        x = hn[-1]  # Dernière sortie cachée pour la classification
        
        # Couche fully connected non linéaire
        x = self.fc(x)
        x = self.fc_non_linear(x)
        
        return self.out(x)
    
class UNet1D(nn.Module):
    def __init__(self, N=6, C=2, T=2048):
        super(UNet1D, self).__init__()

        # Contracting path (encodeur)
        self.enc1 = self.conv_block(C, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)

        # Bottom layer
        self.middle = self.conv_block(128, 256)

        # Expanding path (décodeur)
        self.upconv4 = self.upconv_block(256, 128)
        self.upconv3 = self.upconv_block(128, 64)
        self.upconv2 = self.upconv_block(64, 32)
        self.upconv1 = self.upconv_block(32, 16)

        # Final output layer
        self.fc = nn.Linear(16 * T // 16, N)  # Output layer with N classes

        # Activation non-linéaire pour la classification
        self.out = nn.LogSoftmax(dim=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        # Contracting path (encodeur)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottom layer
        middle = self.middle(enc4)

        # Expanding path (décodeur)
        up4 = self.upconv4(middle)
        up3 = self.upconv3(up4 + enc4)  # Skip connections
        up2 = self.upconv2(up3 + enc3)  # Skip connections
        up1 = self.upconv1(up2 + enc2)  # Skip connections

        # Flatten and apply the final fully connected layer
        x = up1.view(up1.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return self.out(x)


