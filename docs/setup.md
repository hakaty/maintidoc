# Installation et démarrage (Milestone 1)

## Prérequis

- Python 3.11
- Docker + Docker Compose
- ~8 Go de RAM disponibles si vous faites tourner les modèles 7B en local sur
  CPU (voir [Notes matérielles](#notes-matérielles))

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/Git Bash: source .venv/Scripts/activate)
pip install -e ".[dev]"
cp .env.example .env
```

## Démarrer la base de données

```bash
docker compose up -d
python scripts/init_db.py
```

`docker compose up -d` démarre uniquement `db` (PostgreSQL 16 + pgvector),
exposée sur `127.0.0.1:5433` uniquement (pas sur le réseau).
`scripts/init_db.py` active l'extension `vector` (idempotent, peut être relancé
sans risque).

## Installer Ollama (recommandé : natif sur Windows)

`ollama` n'est **pas** démarré par `docker compose up -d` (service sous le
profil Compose optionnel `ollama`). Sur Windows, l'installation native est
recommandée : plus rapide (accès GPU direct, pas de virtualisation Docker
supplémentaire) et plus simple à mettre à jour.

```bash
# 1. Installer Ollama nativement (Windows) : https://ollama.com/download/windows
#    puis vérifier :
ollama --version

# 2. Récupérer les modèles
ollama pull qwen2.5:7b-instruct   # LLM texte (OLLAMA_LLM_MODEL)
ollama pull qwen2.5vl:3b          # VLM (images), OLLAMA_VLM_MODEL
```

Ollama natif écoute par défaut sur `http://127.0.0.1:11434`, qui correspond
déjà à `OLLAMA_BASE_URL` dans `.env.example` -- rien à changer.

### Alternative : Ollama en conteneur

Si vous préférez ne pas installer Ollama nativement (ex. CI, machine sans
accès admin) :

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose exec ollama ollama pull qwen2.5vl:3b
```

`OLLAMA_BASE_URL` reste `http://127.0.0.1:11434` dans les deux cas (le port
du conteneur est publié sur le même port hôte) : le choix natif/conteneur est
donc transparent pour l'application.

## Vérifier que tout est prêt

```bash
python scripts/check_env.py
```

Affiche un tableau `OK` / `MISSING` pour la base (pgvector), la joignabilité du
fournisseur LLM sélectionné et la présence des modèles requis, avec la
commande de correction à lancer pour chaque ligne en échec. Aucune clé n'est
jamais affichée.

## Basculer sur OpenRouter (fallback)

Dans `.env` :

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
```

`OPENROUTER_LLM_MODEL` / `OPENROUTER_VLM_MODEL` ont des valeurs par défaut
dans `.env.example` ; ajustez-les si besoin selon le catalogue
https://openrouter.ai/models. Si `LLM_PROVIDER=openrouter` est défini sans
`OPENROUTER_API_KEY`, l'application refuse de démarrer avec un message
d'erreur explicite (voir `app/config.py`).

## Indexation PGVector (Milestone 4)

```bash
python -m app.images enrich data/processed/<sha12>        # si pas déjà fait (M3)
python -m app.indexing build data/processed/<sha12>
python -m app.indexing search "une question en français" --k 5
```

Le premier `build` télécharge `BAAI/bge-m3` (~2,2 Go) sous `models/`
(git-ignoré, resumable) -- une seule fois, les runs suivants réutilisent le
cache. `models/` sert aussi de `HF_HOME`. Les tables PGVector
(`maintidoc_text`, `maintidoc_images`, configurables dans `.env`) sont créées
automatiquement au premier `build` (extension `vector` déjà activée par
`scripts/init_db.py`).

## PC sans GPU (collègue sans carte NVIDIA)

Si vous reprenez ce projet sur une machine sans GPU (ou avec une VRAM trop
limitée), vous n'avez **pas besoin** de refaire tourner le VLM (M3) ni
l'indexation (M4) : il suffit de récupérer les données déjà calculées, puis de
basculer la génération de texte sur OpenRouter (le LLM 7B en local sur CPU
serait très lent).

### 1. Récupérer les données déjà calculées

Demandez à la personne qui a l'environnement GPU de lancer
`python scripts/export_index.py`, qui produit une archive zip (dump de la
base PGVector + `data/processed/` + `data/media/`, **jamais** `.env` ni
`models/`). Décompressez-la à la racine du projet, puis :

```bash
docker compose up -d db
python scripts/import_index.py chemin/vers/export.zip
```

`import_index.py` restaure le dump PostgreSQL et vérifie les comptages
attendus (140 nœuds texte, 52 images) avant de continuer.

### 2. Configurer `.env` pour une machine sans GPU

```bash
# Génération de texte déportée : pas de LLM 7B local sur CPU.
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...        # votre propre clé, jamais celle d'un collègue
OPENROUTER_LLM_MODEL=qwen/qwen-2.5-7b-instruct

# BGE-M3 et MobileCLIP tournent en CPU/fp32 (plus lent mais fonctionnel) :
DEVICE=cpu
```

Avec `DEVICE=cpu` (ou `DEVICE=auto` sans CUDA détecté), un message explicite
`"CPU mode"` est journalisé au chargement de BGE-M3/MobileCLIP -- jamais de
bascule silencieuse. `EMBED_QUERY_DEVICE` peut forcer un device différent pour
les embeddings de requête uniquement si besoin (rarement utile sur une machine
déjà tout-CPU).

### 3. Ordres de grandeur attendus sur CPU

Mesuré sur cette machine en **fp16/cuda** (RTX 3060 Laptop) : BGE-M3 encode
140 chunks en 12 s (~0,09 s/chunk), MobileCLIP encode une image en quelques
dizaines de ms. Sur CPU/fp32, sans mesure réelle disponible sur une machine
sans GPU, attendez-vous (ordre de grandeur usuel pour ce type de modèle,
**estimation, pas une mesure**) à un facteur **5 à 15x plus lent** pour les
embeddings (donc quelques dizaines de secondes à ~1-2 min pour réindexer 140
chunks si vous deviez le refaire -- ce que ce workflow évite justement) ; une
recherche individuelle (une seule requête embeddée) reste de l'ordre de la
seconde même sur CPU, car un seul texte court est encodé à la fois.

## Notes matérielles

- `qwen2.5:7b-instruct` / `qwen2.5vl:7b` : confortable avec un GPU récent
  (>= 8 Go de VRAM) ; sur CPU seul, attendez-vous à une génération lente.
- Les variantes `*:3b` (ex. `qwen2.5vl:3b`, valeur par défaut de
  `OLLAMA_VLM_MODEL`) sont nettement plus légères et conviennent à une machine
  de développement sans GPU dédié.
- En l'absence de GPU ou de ressources suffisantes, `LLM_PROVIDER=openrouter`
  déporte l'inférence et évite de charger un modèle localement.
