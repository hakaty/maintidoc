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
