import numpy as np
from pathlib import Path

dossier = Path("data/processed")
X_train = np.load(dossier / "X_train.npy")
X_val = np.load(dossier / "X_val.npy")
X_test = np.load(dossier / "X_test.npy")

# 1. Variante "Mains seules" (Ablation)
# Les 99 premières valeurs sont le corps, les suivantes sont les mains
X_train_mains = X_train[:, :, 99:]
X_val_mains = X_val[:, :, 99:]
X_test_mains = X_test[:, :, 99:]
np.save(dossier / "X_train_mains.npy", X_train_mains)
np.save(dossier / "X_val_mains.npy", X_val_mains)
np.save(dossier / "X_test_mains.npy", X_test_mains)
print("Variantes 'mains' sauvegardées.")

# 2. Variantes de longueur temporelle (15 et 45 frames)
def reechantillonner_sequence(X, n_cible):
    N, T, D = X.shape
    X_out = np.zeros((N, n_cible, D), dtype=X.dtype)
    for i in range(N):
        indices = np.linspace(0, T - 1, n_cible)
        X_out[i] = X[i][np.round(indices).astype(int)]
    return X_out

X_train_15 = reechantillonner_sequence(X_train, 15)
X_val_15 = reechantillonner_sequence(X_val, 15)
X_test_15 = reechantillonner_sequence(X_test, 15)
np.save(dossier / "X_train_15.npy", X_train_15)
np.save(dossier / "X_val_15.npy", X_val_15)
np.save(dossier / "X_test_15.npy", X_test_15)

X_train_45 = reechantillonner_sequence(X_train, 45)
X_val_45 = reechantillonner_sequence(X_val, 45)
X_test_45 = reechantillonner_sequence(X_test, 45)
np.save(dossier / "X_train_45.npy", X_train_45)
np.save(dossier / "X_val_45.npy", X_val_45)
np.save(dossier / "X_test_45.npy", X_test_45)
print("Variantes temporelles (15 et 45 frames) sauvegardées.")

# 3. Variante avec normalisation par groupe (Mains et Corps séparément)
# Note: on suppose que les données actuelles sont normalisées par les épaules.
# On va dé-normaliser approximativement puis re-normaliser par groupe.
# (Pour une vraie expérience, il faudrait repartir des données brutes,
# mais pour ce test, on va simuler une normalisation locale par groupe).
# Cette partie est optionnelle car complexe à faire sans les données brutes.
print("Préparation des variantes mains et temporelles terminée.")
print("La variante de normalisation par groupe n'est pas implémentée.")