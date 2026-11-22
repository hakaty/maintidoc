# M7.5 — Faire tourner l'application en Docker

## Objectif

`docker compose --profile app up -d` démarre l'interface Streamlit + le
code RAG/agent, pour deux profils d'utilisateur différents, sans
installation Python sur la machine hôte :

- **Moi (GPU, Ollama natif)** : `LLM_PROVIDER=ollama`, Ollama tourne
  nativement sur Windows avec la carte GPU — **jamais** dans Docker sur
  cette machine.
- **Mon collègue (sans GPU)** : `LLM_PROVIDER=openrouter` avec sa propre
  clé, importe l'index déjà construit (pas d'ingestion côté collègue).

## Architecture (texte)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Hôte Windows (moi)                                                   │
│                                                                       │
│  Ollama natif (GPU) ──────────────────────┐                          │
│  127.0.0.1:11434                          │ host.docker.internal     │
│                                            │ :11434                  │
│  ┌──────────────────── Docker Desktop ────┼───────────────────────┐ │
│  │  profil "app"                          │                       │ │
│  │  ┌─────────────────┐   réseau compose  │   ┌─────────────────┐ │ │
│  │  │ service "app"    │───────────────────┘   │ service "db"    │ │ │
│  │  │ Streamlit :8501  │───────────────────────▶│ pgvector :5432 │ │ │
│  │  │ (CPU par défaut) │   db:5432 (interne)    │                 │ │ │
│  │  └────────┬─────────┘                        └────────┬────────┘ │ │
│  │           │ volumes :                                 │          │ │
│  │           │  hf_cache  -> /app/models (modèles HF)     │ db_data  │ │
│  │           │  ./data    -> /app/data   (PDF, index)     │          │ │
│  └───────────┼──────────────────────────────────────────────────────┘ │
│              │                                                        │
│  127.0.0.1:8501 (navigateur, localhost uniquement)                    │
└─────────────────────────────────────────────────────────────────────┘

Collègue (pas de GPU) : même schéma, SAUF qu'il n'y a pas d'Ollama natif --
"app" pointe vers OpenRouter (son LLM_PROVIDER=openrouter + sa clé dans
.env), et il importe mon export d'index (scripts/import_index.py) au lieu
de relancer l'ingestion.
```

## Choix faits, et pourquoi

- **CPU par défaut, GPU optionnel (`--profile gpu`)** : le propriétaire du
  GPU fait tourner Ollama nativement (accès GPU direct, plus rapide) — le
  conteneur Streamlit n'a donc besoin que du CPU pour BGE-M3/MobileCLIP
  (déjà assez rapide, mesuré dans le FINAL REPORT). Le collègue sans GPU
  utilise exactement la même image CPU. Un profil `gpu` séparé existe pour
  qui voudrait aussi dockeriser les embeddings sur GPU (nécessite WSL2 +
  NVIDIA Container Toolkit, non vérifié dans ce milestone).
- **Ollama jamais dans Docker sur la machine GPU** : contrainte explicite du
  projet. Le service `ollama-docker` existe pour un utilisateur Linux sans
  Ollama natif, mais est caché derrière son propre profil
  (`--profile ollama-docker`), jamais démarré par défaut ni par
  `--profile app`.
- **Ports en 127.0.0.1 uniquement** : `8501` (app), `5433` (db),
  `11434` (ollama-docker, si utilisé) ne sont jamais exposés sur `0.0.0.0`
  — uniquement accessibles depuis la machine elle-même.
- **Utilisateur non-root** : l'image crée un utilisateur `maintidoc`
  (uid 1000), propriétaire de `/app`, et l'exécute avec `USER maintidoc`.
- **Healthchecks** : `db` (`pg_isready`, déjà présent depuis M1) et `app`
  (`curl http://127.0.0.1:8501/_stcore/health`) — `app` attend que `db`
  soit `service_healthy` avant de démarrer (`depends_on.condition`).
- **Cache des modèles HF dans un volume nommé** (`hf_cache` ->
  `/app/models`, `HF_HOME` pointe dessus) : BGE-M3 et MobileCLIP ne sont
  téléchargés qu'une fois, pas à chaque redémarrage/rebuild.
- **`data/` monté en bind volume** (pas copié dans l'image) : le PDF
  importé, les images extraites et l'index construit survivent à
  `docker compose down` / `up`, et c'est aussi là qu'atterrit un PDF
  importé depuis l'interface (section 4).
- **`.env` jamais dans l'image** : passé via `env_file`, jamais `COPY .env`
  ni variable `ARG`/`ENV` codée en dur dans le `Dockerfile`. Les seules
  valeurs qui changent entre hôte et conteneur (`DATABASE_URL`,
  `OLLAMA_BASE_URL`, `DEVICE`) sont surchargées dans `environment:` du
  service, jamais une clé.

## Durées mesurées

Voir le FINAL REPORT du commit M7.5 pour les chiffres réels (taille
d'image, temps de build, premier démarrage avec téléchargement des
modèles, démarrages suivants, temps de réponse Docker vs natif).
