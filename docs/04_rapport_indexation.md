# Rapport d'indexation — Milestone 4 (Vectorisation PGVector)

Document indexé : `data/pdf/document.pdf` (notice Lowara CEA/CIE/CA, 48 pages),
`pdf_sha256=5c1a4a3e34a4ceb95fa77f47c692a65a94375aae1ee1e46fcef48029c9785143`.

Les valeurs ci-dessous sont des résultats observés sur cette machine et ce
document précis (RTX 3060 Laptop, 6 Go VRAM), pas des garanties générales.

## 1. Modèles et bibliothèques

| Composant | Valeur |
|---|---|
| Embeddings texte | `BAAI/bge-m3` via `llama-index-embeddings-huggingface` 0.8.0 (`sentence-transformers` 6.2.0) |
| Embeddings image | MobileCLIP-S2 / datacompdr (open_clip 3.3.0) — déjà en place depuis M3 |
| Device texte | `cuda`, poids chargés en fp16 (`model_kwargs={"torch_dtype": torch.float16}`) |
| Dimension texte | 1024, L2-normalisée |
| Dimension image | 512, L2-normalisée (précalculée en M3, réutilisée telle quelle) |
| Store | `llama-index-vector-stores-postgres` 0.9.0 sur PostgreSQL 16.15 + pgvector 0.5.1 |
| Index vectoriel | HNSW, `vector_cosine_ops`, `hnsw_m=16`, `hnsw_ef_construction=64`, `hnsw_ef_search=40` |
| Recherche texte | hybride (vecteur + plein texte PostgreSQL, `text_search_config="french"`), activable/désactivable par requête |

## 2. BGE-M3 : chargement et débit

| Mesure | Valeur |
|---|---|
| Premier téléchargement (réseau froid, cache vide) | ~95 s (mesuré par le test d'intégration réel) |
| Taille du cache sur disque (`models/`) | 4 362,6 MB (~4,26 GB) |
| Rechargement depuis le cache local | 7,78 s |
| Device | cuda (NVIDIA GeForce RTX 3060 Laptop GPU), fp16 |
| VRAM pic (BGE-M3 seul) | 1 416,6 MB |
| Embedding de 140 chunks (texte+tableau+figure) | 12,12 s au total, 0,087 s/chunk en moyenne |

Avant le chargement de BGE-M3, les modèles Ollama (VLM `qwen2.5vl:3b` et LLM
`qwen2.5:7b-instruct`) sont explicitement déchargés (`keep_alive=0`) pour ne
jamais faire cohabiter deux modèles sur les 6 Go de VRAM disponibles.

## 3. Comptages en base (après `python -m app.indexing build`)

| Table | Total | Détail |
|---|---|---|
| `data_maintidoc_text` | 140 | 58 `text`, 30 `table`, 52 `image_description` |
| `data_maintidoc_images` | 52 | 52 images (vecteurs CLIP précalculés en M3) |

Re-lancer `build` sans `--force` : **0 nœud ajouté**, 140/52 toujours en base
(idempotence confirmée par ré-exécution réelle, pas seulement par test).

## 4. Exemples de recherche (texte, français)

| # | Question | Rang 1 | Page | Type | Score |
|---|---|---|---|---|---|
| 1 | "Quels sont les quatre niveaux de danger utilisés dans ce manuel ?" | Introduction/Sécurité | 4 | text | 0,598 |
| 1bis | (même requête, rang 2) | Tableau "Niveaux de danger" | 4 | **table** | 0,585 |
| 2 | "Que faut-il faire si le RCD s'est déclenché ?" | Tableau "9.8 RCD déclenché" | 37 | **table** | 0,724 |
| 3 | "Quel pictogramme signale un risque d'atmosphère explosive ?" | Figure "Symboles complémentaires" | 5 | **image_description** | 0,626 |
| 4 | "À quelle température l'unité emballée doit-elle être stockée ?" | Section "2.4 Stockage" | 9 | text | 0,705 |
| 5 (mode hybride) | "Où trouver les informations sur la garantie ?" | Section "13 Garantie" | 45 | text | 0,674 |

Les questions 2 et 3 illustrent une réponse exacte par tableau et par figure
respectivement.

## 5. Recherche image (MobileCLIP)

- **Image → image** : une image découpée (moitié gauche de `5c1a4a3e_p1_i2.png`,
  sauvegardée sous `tests/fixtures/crop_p1_i2_pump.png`) retrouve l'image
  source en rang 1 avec un score de 0,959, loin devant le rang 2 (0,804).
- **Texte anglais → image** :
  - `"a centrifugal pump"` → rang 1 : `5c1a4a3e_p38_i0` (légende "P1max +
    Pmax ≤ PN", score 0,314) — résultat faible, cohérent avec le fait que ce
    manuel contient surtout des schémas techniques/pictogrammes, peu de
    photographies de pompes isolées.
  - `"a warning safety pictogram"` → rang 1 : `5c1a4a3e_p5_i1` ("Symboles
    complémentaires", score 0,354) — résultat cohérent.

CLIP text→image reste en anglais uniquement (limite documentée du modèle) ;
les scores de recherche texte→image sont globalement plus bas que
image→image, ce qui est attendu avec MobileCLIP sur des pictogrammes/schémas
plutôt que des photographies naturelles.

## 6. Évaluation de la recherche (`eval/run_retrieval.py`)

20 questions françaises (`eval/retrieval_gold_v1.yaml`), écrites en lisant le
PDF réel : 7 `text`, 6 `table`, 5 `image_description`, 2 non couvertes par le
manuel.

| Mode | hit@5 (global) | MRR@10 (global) |
|---|---|---|
| vector | 1,000 | 0,907 |
| hybrid | 1,000 | 0,907 |

Par type de contenu (identique vecteur/hybride sur ce jeu de questions) :

| content_type | n | hit@5 | MRR@10 |
|---|---|---|---|
| text | 7 | 1,000 | 1,000 |
| table | 6 | 1,000 | 0,806 |
| image_description | 5 | 1,000 | 0,900 |

Questions non couvertes (score du meilleur résultat retourné, sans bonne
réponse à trouver) :

| id | question | top_score | top_page | top_content_type |
|---|---|---|---|---|
| q19 | consommation électrique annuelle moyenne d'une pompe CEA 70/5 | 0,554 | 42 | text |
| q20 | prix de vente recommandé de la pompe CA 1206/3 | 0,493 | 39 | table |

Le mode hybride obtient ici les mêmes métriques que le mode vecteur sur ce
jeu de 20 questions : les formulations en langage naturel du gold set
recoupent déjà bien le vocabulaire du document, donc le signal plein-texte
n'apporte pas de gain supplémentaire mesurable ici. `test_hybrid_search_finds_french_keyword_match`
(test dédié, cas synthétique) démontre séparément qu'il existe des requêtes où
le mode hybride retrouve un résultat que le mode vecteur seul rate.

Ces valeurs sont des résultats observés sur ce document et ce jeu de
questions précis, pas des garanties générales de l'index.
