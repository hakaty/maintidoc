# Problèmes connus (observés, pas corrigés dans ce milestone)

- **Confusion page/numéro dans une réponse (RAG, qualité)** : sur la question
  du pictogramme "atmosphère explosive" (p. 5), une réponse a mentionné
  « symbole de numéro 5 », confondant le numéro de page cité (p. 5) avec un
  numéro de symbole qui n'existe pas dans le manuel. Observé lors de la
  vérification visuelle de l'UI (section 6.7 du retour utilisateur) --
  volontairement NON corrigé dans ce milestone, qui ne touche que l'UI. À
  investiguer dans un futur travail sur la qualité du RAG/prompt de
  génération (`prompts/rag_answer_v1.md`).
