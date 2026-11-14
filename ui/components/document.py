"""Document page: upload a PDF, run extract -> enrich -> index, manage docs.

Bug fix (section 0e): the "Qualité des réponses" table used to read a
STATIC section out of docs/05_rapport_rag.md (frozen at whatever the M5
report said). It now reads the CURRENT eval/results_*.json files (written by
eval/run_qa.py), so the numbers shown always match the latest real run.
"""

import hashlib
import json
from pathlib import Path

import streamlit as st

from app.config import Settings
from app.images.pipeline import run_enrichment
from app.indexing.stores import build_index
from app.ingestion.pipeline import run_extraction
from ui.components.uploads import UploadValidationError, generated_pdf_filename, validate_pdf_bytes
from ui.format import fr_decimal, fr_percent, fr_seconds
from ui.icons import ICON_CHECK_CIRCLE, ICON_PDF

PROCESSED_ROOT = Path("data/processed")
PDF_ROOT = Path("data/pdf")
RESULTS_PATHS = {
    "Simple": Path("eval") / "results_simple_after.json",
    "Agent": Path("eval") / "results_agent_after.json",
}
MULTITURN_PATH = Path("eval") / "results_multiturn_after.json"


def _list_documents() -> list[dict]:
    docs = []
    if not PROCESSED_ROOT.is_dir():
        return docs
    for directory in sorted(PROCESSED_ROOT.iterdir()):
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            docs.append({"sha12": directory.name, "manifest": manifest})
    return docs


def _render_stat_tiles(manifest: dict) -> None:
    """Bug fix (section 0g): tiles are now one single HTML block each (value
    + label together, wrapped in a bordered white ".md-tile" card like the
    reference's `.tile` class) -- values/labels are manifest-derived
    integers, never user/document text, so no escaping is needed here."""
    counts = manifest.get("counts", {})
    stats = [
        ("Pages", manifest.get("total_pages", 0)),
        ("Passages", counts.get("text_chunks", 0)),
        ("Tableaux", counts.get("tables", 0)),
        ("Figures décrites", counts.get("images_kept", 0)),
    ]
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats, strict=True):
        with col:
            st.markdown(
                f'<div class="md-tile"><span class="md-stat-value">{value}</span>'
                f'<span class="md-stat-label">{label}</span></div>',
                unsafe_allow_html=True,
            )


def _render_active_manual_card(manifest: dict) -> None:
    with st.container(border=True):
        cols = st.columns([1, 5], vertical_alignment="center")
        with cols[0]:
            st.markdown(ICON_PDF, unsafe_allow_html=True)
        with cols[1]:
            source = Path(manifest.get("source_file", "manuel.pdf")).name
            st.markdown(f"**{source}**")
            st.caption(f"Français · {manifest.get('total_pages', 0)} pages")
        st.markdown(
            f'<span class="md-chip-success">{ICON_CHECK_CIRCLE} Prêt à interroger</span>',
            unsafe_allow_html=True,
        )


def _render_preparation_timeline(manifest: dict) -> None:
    st.markdown("###### Préparation du manuel")
    counts = manifest.get("counts", {})
    enrichment = manifest.get("images_enrichment", {})
    steps = [
        (
            "Extraction",
            f"{counts.get('text_chunks', 0)} passages · {counts.get('tables', 0)} tableaux · "
            f"{counts.get('images_kept', 0)} images",
        ),
        (
            "Description des figures",
            f"{enrichment.get('images_total', 0)} figures · "
            f"{enrichment.get('total_vlm_seconds', 0):.0f} s au total "
            f"({enrichment.get('avg_vlm_seconds', 0):.1f} s/figure)"
            if enrichment
            else "non exécutée pour ce document",
        ),
        (
            "Indexation",
            f"{counts.get('text_chunks', 0) + counts.get('tables', 0)} passages indexés · "
            f"{counts.get('images_kept', 0)} figures indexées",
        ),
    ]
    for label, detail in steps:
        cols = st.columns([1, 10], vertical_alignment="top")
        with cols[0]:
            st.markdown(
                '<span style="flex:none;width:32px;height:32px;border-radius:50%;'
                "background:#157347;display:flex;align-items:center;justify-content:center;"
                f'color:#FFFFFF;">{ICON_CHECK_CIRCLE}</span>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(f"**{label}**")
            st.caption(detail)


def _load_eval_overall(mode_label: str) -> dict | None:
    path = RESULTS_PATHS.get(mode_label)
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["report"]["overall"]
    except Exception:  # noqa: BLE001 -- missing/malformed file just hides the row
        return None


def _load_multiturn_hit_rate() -> float | None:
    if not MULTITURN_PATH.is_file():
        return None
    try:
        data = json.loads(MULTITURN_PATH.read_text(encoding="utf-8"))
        by_turn = data.get("by_turn_hit_rate", {})
        return by_turn.get("turn_2")
    except Exception:  # noqa: BLE001
        return None


def _pct(value: float | None) -> str:
    # French decimals (bug fix, section 0i): "80,8 %", never "81 %".
    return fr_percent(value) or "—"


def _ms(value: float | None) -> str:
    # ms -> s, French decimal (bug fix, section 0i): "14,9 s", never "14875 ms".
    return fr_seconds(value) or "—"


def _groundedness(value: float | None) -> str:
    return fr_decimal(value) or "—"


def render_eval_quality_table() -> None:
    """"Qualité des réponses" -- read from the latest real eval/*.json files,
    never from a static doc (section 0e fix)."""
    st.markdown("###### Qualité des réponses")
    simple = _load_eval_overall("Simple")
    agent = _load_eval_overall("Agent")
    if simple is None and agent is None:
        st.caption("Aucun résultat d'évaluation disponible (lancez eval/run_qa.py).")
        return

    turn2_hit = _load_multiturn_hit_rate()
    rows = [
        ("Bonne page trouvée", "expected_page_hit_rate", _pct),
        ("Citations valides", "citation_validity_rate", _pct),
        ("Refus corrects", "refusal_accuracy", _pct),
        ("Fidélité au manuel", "groundedness_avg", _groundedness),
        ("Temps de réponse médian", "latency_ms_p50", _ms),
    ]
    table_md = "| Métrique | Simple | Agent |\n|---|---|---|\n"
    for label, key, fmt in rows:
        simple_val = fmt(simple.get(key)) if simple else "—"
        agent_val = fmt(agent.get(key)) if agent else "—"
        table_md += f"| {label} | {simple_val} | {agent_val} |\n"
    table_md += f"| Questions de suivi (2ᵉ tour) | — | {_pct(turn2_hit)} |\n"
    st.markdown(table_md)
    st.caption("31 questions de test · résultats observés, pas des garanties.")


def _run_pipeline(pdf_path: Path, use_vlm: bool, settings: Settings) -> tuple[dict, int]:
    with st.status("Traitement du document…", expanded=True) as status:
        st.write("Extraction…")
        extraction = run_extraction(pdf_path, PROCESSED_ROOT, settings.media_dir)
        st.write(":material/check_circle: Extraction terminée")

        st.write("Images…")
        run_enrichment(extraction.processed_dir, use_vlm=use_vlm, settings=settings)
        st.write(":material/check_circle: Images terminées")

        st.write("Indexation…")
        build_result = build_index(extraction.processed_dir, settings=settings)
        st.write(":material/check_circle: Indexation terminée")

        status.update(label="Manuel prêt", state="complete")

    manifest = json.loads(
        (extraction.processed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest, build_result.text_nodes_total


def render_document_tab(settings: Settings) -> None:
    header = st.columns([4, 1])
    with header[1]:
        import_clicked = st.button(
            "Importer un autre PDF", icon=":material/upload_file:", type="secondary"
        )

    docs = _list_documents()
    if docs:
        active_manifest = docs[-1]["manifest"]
        _render_active_manual_card(active_manifest)
        _render_stat_tiles(active_manifest)
        _render_preparation_timeline(active_manifest)
    else:
        st.caption("Aucun document indexé pour le moment.")

    render_eval_quality_table()

    if import_clicked or not docs:
        st.markdown("###### Importer un manuel")
        uploaded = st.file_uploader("Manuel PDF", type=["pdf"])
        use_vlm = st.checkbox(
            "Décrire les images avec le VLM (plus lent, plusieurs minutes)", value=True
        )
        if uploaded is not None and st.button(
            "Indexer ce document", icon=":material/dataset:", type="primary"
        ):
            data = uploaded.getvalue()
            try:
                validate_pdf_bytes(data)
            except UploadValidationError as exc:
                st.error(str(exc))
            else:
                sha256 = hashlib.sha256(data).hexdigest()
                if (PROCESSED_ROOT / sha256[:12] / "manifest.json").is_file():
                    st.info("Ce document est déjà indexé.", icon=":material/info:")
                    st.session_state["active_pdf_sha256"] = sha256[:12]
                else:
                    PDF_ROOT.mkdir(parents=True, exist_ok=True)
                    pdf_path = PDF_ROOT / generated_pdf_filename(uploaded.name)
                    pdf_path.write_bytes(data)
                    manifest, _ = _run_pipeline(pdf_path, use_vlm, settings)
                    st.session_state["active_pdf_sha256"] = sha256[:12]
                    _render_active_manual_card(manifest)

    if len(docs) > 1:
        st.markdown("###### Documents indexés")
        options = [d["sha12"] for d in docs]
        active = st.selectbox("Document actif", options, index=len(options) - 1)
        st.session_state["active_pdf_sha256"] = active
