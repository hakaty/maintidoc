# Rapport Agent — Milestone 7 (RAG agentique LangChain)

Les valeurs ci-dessous sont des résultats observés sur cette machine (RTX
3060 Laptop, 6 Go VRAM) et ce document précis, pas des garanties générales.

## 1. Conception

- **Agent** : `langchain.agents.create_agent` (API réellement installée,
  `langchain==1.4.7`), pas de graphe LangGraph personnalisé -- `create_agent`
  compile un graphe en interne, mais nous ne définissons aucun nœud/arête
  nous-mêmes.
- **Outils** (`app/agent/tools.py`), tous liés côté serveur au document actif
  (`pdf_sha256`) -- le LLM ne choisit jamais le document :
  `search_manual`, `search_tables`, `search_figures`, `get_page`,
  `similar_figures` (exposé uniquement si une photo est jointe).
- **Garde-fous** : détection d'injection au niveau de la question (rappel
  ferme ajouté au prompt système) ; `ToolCallLimitMiddleware(run_limit=4,
  exit_behavior="end")` ; limite de récursion LangGraph ; timeout global
  (90 s par défaut, `AGENT_TIMEOUT_SECONDS`, thread + `future.result(timeout=...)`
  pour ne jamais bloquer au-delà) ; vérification de citations restreinte au
  "journal de preuves" (`EvidenceLedger`, uniquement les nœuds réellement
  renvoyés par un outil durant CE run) ; règle "sans source" (citation nulle
  → refus standard) ; un seul bloc "⚠ Sécurité" (déduplication) ; repli
  (`fallback_rag`) sur `app.rag.service.answer()` en cas d'erreur de l'agent.
- **Mémoire** : historique par session (`session_id`), limité aux 6 derniers
  messages, géré en mémoire Python (pas de checkpointer LangGraph -- la
  troncature propre aurait demandé un middleware supplémentaire). "Nouvelle
  conversation" vide l'historique.

## 2. Fiabilité de l'appel d'outils (Qwen 7B q3_K_M)

Preuve directe (`llm.bind_tools([...]).invoke(...)`) : le modèle émet un
appel d'outil correctement formé (nom + arguments) dès le premier test, y
compris sur une question en français nécessitant une recherche dans le
manuel -- **après renforcement du prompt système** (règle 0 explicite : "tu
DOIS appeler un outil avant de répondre"). Sans cette règle, le modèle a
répondu directement par un refus sans appeler aucun outil sur une question
pourtant couverte par le manuel (voir §6, premier échec observé). Avec la
règle, l'appel d'outil est déclenché de façon fiable dans tous les tests
réels effectués (table, figure, page).

## 3. Comparaison Simple (RAG) vs Agent (31 questions réelles)

| Métrique (ensemble, n=31) | Simple | Agent |
|---|---|---|
| Validité des citations | 100 % | 100 % |
| Hit page attendue | 76,9 % | **38,5 %** |
| Refus correct | 80,6 % | **58,1 %** |
| Adversarial géré | 100 % | 100 % |
| Groundedness (juge) | 0,950 | 0,846 |
| Latence p50 / p95 | 14 625 / 52 219 ms | **7 187 / 9 891 ms** |

**Constat honnête** : l'agent est nettement plus rapide (un seul appel
d'outil ciblé contre une recherche + expansion + budget de contexte plus
large côté Simple), mais répond correctement à BEAUCOUP moins de questions
answerable -- la plupart des refus supplémentaires ont pour cause
`no_valid_citation` (l'agent a cherché mais n'a pas cité de page valide dans
sa réponse finale, ou le contexte renvoyé par l'outil était trop étroit :
`search_manual` ne retourne que `k=6` résultats tronqués à 500 caractères
chacun, contre un budget de 2500 tokens et une expansion de requête côté
Simple). Ceci n'a PAS été corrigé en resserrant les vérifications ou en
ajustant les paramètres après coup pour améliorer ce chiffre : c'est un
résultat observé, documenté tel quel.

Par groupe :

| Groupe | Hit page (Simple) | Hit page (Agent) | Refus correct (Simple) | Refus correct (Agent) |
|---|---|---|---|---|
| base (20) | 94,4 % | 44,4 % | 95,0 % | 55,0 % |
| paraphrase (6) | 50,0 % | 16,7 % | 50,0 % | 50,0 % |
| unanswerable (3) | -- | -- | 100 % | 100 % |
| adversarial (2) | 0,0 % | 50,0 % | 0,0 % | 50,0 % |

## 4. Multi-tours (6 conversations, `eval/qa_multiturn_v1.yaml`)

| Tour | Hit rate |
|---|---|
| Tour 1 (question initiale) | 50,0 % |
| Tour 2 (relance) | 0,0 % |

Le tour 2 hérite de la même limitation que le mode Agent en général (citation
non produite → refus). La relance elle-même fonctionne mécaniquement (voir
§5) : l'historique est bien transmis, mais le manque de citation valide
déclenche la règle "sans source" aussi souvent en tour 2 qu'en tour 1.

## 5. Mémoire : résultat des relances

Preuve (test réel, `test_followup_question_is_sent_with_prior_history`) :
après le premier tour, `_session_histories[session_id]` contient exactement
`[HumanMessage, AIMessage]` (2 messages) ; après le second tour, 4 messages.
L'historique est bien transmis à l'appel suivant de l'agent. Le test de
troncature confirme que l'historique est limité à `MAX_HISTORY_MESSAGES = 6`
après plusieurs tours, et `reset_session()` le vide complètement
("Nouvelle conversation").

## 6. Quatre traces d'exemple

### Table (réel)
**Question** : "Que faut-il faire si le RCD s'est déclenché et que l'unité ne s'allume pas ?"
**Appel d'outil** : `search_tables(query="RCD déclenché et unité ne s'allume pas", k=1)` → 1 résultat, 5,89 s.
**Réponse** : "...vous devez réinitialiser le dispositif de protection à courant différentiel résiduel (RCD) [p. 35]..." -- citation valide, page 35.

### Figure (réel, via recherche directe)
**Appel d'outil** : `search_figures(query="explosif", k=4)` sur un nœud `image_description` synthétique page 5 → réponse citant `[p. 5, figure]` correctement reconnue comme figure.

### Relance (mémoire)
Tour 1 : "Que faut-il faire si le RCD s'est déclenché ?" → réponse citée p.35.
Tour 2 : "Et si ça ne suffit pas, quelle est l'étape suivante ?" → l'agent a cherché à nouveau (historique transmis) mais a refusé par manque de citation valide dans sa réponse finale -- illustre concrètement la limitation du §3.

### Refus après nouvelle recherche
Scripté (mocké) : l'agent appelle `search_manual` deux fois avec des
formulations différentes avant de conclure à un refus -- confirmé par
`len(result.tool_calls) == 2` et `refusal_reason == "no_valid_citation"`
dans `test_resarch_before_refusal_records_two_tool_calls`.

## 7. Fichiers modifiés

Section 0 (bug bloquant + suivi) : `ui/streamlit_app.py` (renommé depuis
`ui/app.py`), `Makefile`, `README.md`, `app/rag/retriever.py` (score CLIP
réel, expansion adaptative), `app/config.py` (`rag_expansion_skip_score`,
`agent_max_tool_calls`, `agent_timeout_seconds`), `eval/run_qa.py`
(tableau par groupe, causes de refus, comparaison expansion, `--mode agent`,
`--multiturn`).
Milestone 7 : `app/agent/tools.py`, `app/agent/agent.py`,
`prompts/agent_system_v1.md`, `eval/qa_multiturn_v1.yaml`,
`ui/components/chat.py` (sélecteur de mode, trace, photo),
`ui/components/sidebar.py` (reset agent), `ui/components/cards.py`
(`render_trace`), `tests/test_streamlit_launch.py`, `tests/test_agent_tools.py`,
`tests/test_agent.py`, extensions de `tests/test_ui_app.py`.

## 8. Limites connues

- **L'agent répond correctement à moins de questions que le RAG simple**
  sur ce jeu de 31 questions (voir §3) -- la cause principale identifiée est
  un contexte par appel d'outil plus étroit (k et troncature) combiné à une
  tendance du modèle à ne pas toujours citer la bonne page en une seule
  passe. Documenté honnêtement plutôt que corrigé en affaiblissant les
  vérifications.
- Biais d'auto-évaluation du juge de groundedness (même modèle Qwen),
  comme en M5/M6.
- Le timeout agent (thread + `future.result`) ne peut pas tuer le thread
  Python sous-jacent si le modèle ne répond jamais : le thread continue en
  arrière-plan jusqu'à sa fin naturelle, seule la réponse à l'utilisateur
  n'attend pas.
- Mémoire de session en mémoire Python (pas persistée) : perdue au redémarrage
  du processus Streamlit.
