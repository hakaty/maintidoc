# MaintiDoc

RAG multimodal agentique sur un document technique de maintenance
(**Partie I** du projet intégrateur).

Un unique PDF de maintenance (texte, images, tableaux) est analysé pour
répondre à des questions techniques : extraction via LlamaIndex, embeddings
texte open-source (BGE-M3) et image (MobileCLIP via open_clip) stockés dans
PostgreSQL + pgvector, génération par un LLM open-source (Qwen via Ollama,
ou OpenRouter en repli), interface Streamlit, puis RAG agentique avec
LangChain.

Pas de serveur MCP, pas de tickets, pas de bot, pas de fine-tuning de
reranker, pas de comptes utilisateurs dans cette partie du projet.

## Avancement

| #  | Milestone                                  | Statut |
|----|---------------------------------------------|--------|
| 1  | Socle & choix du PDF                        | ✅ |
| 2  | Extraction LlamaIndex (texte, images, tableaux) | ✅ |
| 3  | Images : MobileCLIP + descriptions          | ✅ |
| 4  | Vectorisation PGVector                      | ✅ |
| 5  | RAG : requête & génération                  | ✅ |
| 6  | Interface Streamlit                         | ✅ |
| 7  | RAG agentique LangChain                     | ⏳ |
| 8  | Note technique & démo                       | ⏳ |

## Démarrage

Voir [docs/setup.md](docs/setup.md). Le code est généré milestone par
milestone (voir `prompts/`).

## Interface Streamlit

```bash
make ui
# ou directement :
python -m streamlit run ui/streamlit_app.py
```

Voir `docs/05_rapport_rag.md` pour les métriques d'évaluation du RAG
affichées en lecture seule dans l'onglet Document.

## Lancer avec Docker

Voir aussi [docs/07_docker.md](docs/07_docker.md) pour l'architecture et les
choix techniques. Commandes PowerShell, à exécuter depuis la racine du
projet.

**Prérequis** : Docker Desktop installé et démarré. Pour le PC avec carte
GPU (vous), Ollama natif sur Windows (`ollama serve`) — **jamais** dans
Docker sur cette machine.

### Vous (GPU, Ollama natif)

```powershell
Copy-Item .env.example .env
# Dans .env : LLM_PROVIDER=ollama (déjà la valeur par défaut)
docker compose --profile app build
docker compose --profile app up -d
```

Ouvrez http://127.0.0.1:8501 — les réponses passent par votre Ollama natif
(`host.docker.internal:11434`).

### Votre collègue (sans GPU, OpenRouter)

```powershell
Copy-Item .env.example .env
# Dans .env : LLM_PROVIDER=openrouter, puis collez votre clé dans
# OPENROUTER_API_KEY (jamais commitée, jamais partagée)
# Copiez le zip d'export reçu dans data/ (seul dossier monté dans le
# conteneur) avant de lancer l'import, par ex. :
Copy-Item maintidoc_export_20261122_015343.zip data\
docker compose --profile app up -d db
docker compose --profile app run --rm app python -m scripts.import_index data/maintidoc_export_20261122_015343.zip
docker compose --profile app up -d
```

Le premier `up -d db` suffit à démarrer la base pour l'import ; le dernier
`up -d` démarre l'application complète. Aucune ingestion à relancer : l'index
déjà construit est importé tel quel.

### Arrêter / redémarrer / logs / mettre à jour

```powershell
docker compose --profile app stop          # arrêter (les données persistent)
docker compose --profile app restart app   # redémarrer juste l'app
docker compose --profile app logs -f app   # suivre les logs
docker compose --profile app down          # arrêter ET supprimer les conteneurs (les volumes restent)
docker compose --profile app build --no-cache app && docker compose --profile app up -d  # mettre à jour après un changement de code
```

### Dépannage

- **Docker Desktop est arrêté** : relancez Docker Desktop, attendez l'icône
  verte, puis relancez les commandes `docker compose`.
- **Le port 8501 est déjà utilisé** : un autre programme l'occupe —
  changez le port côté hôte dans `docker-compose.yml` (`"127.0.0.1:8502:8501"`
  par exemple) et utilisez http://127.0.0.1:8502.
- **Ollama injoignable depuis le conteneur** : vérifiez qu'Ollama tourne
  bien nativement (`ollama list` dans un terminal Windows, pas dans le
  conteneur), et que `OLLAMA_BASE_URL=http://host.docker.internal:11434`
  dans `docker-compose.yml` (déjà la valeur par défaut, ne pas la remplacer
  par `127.0.0.1` qui, dans un conteneur, désigne le conteneur lui-même).
- **Premier démarrage lent** : les modèles (BGE-M3, MobileCLIP) se
  téléchargent une seule fois dans le volume `hf_cache` — les démarrages
  suivants sont rapides. Voir docs/07_docker.md pour les durées mesurées.
