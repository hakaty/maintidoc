"""Generate a synthetic 8-page maintenance PDF for tests.

Produces, deterministically:
- 8 pages of French maintenance text (native, not scanned);
- a running header/footer ("MDX-100 - Manuel de maintenance" / "Page N")
  repeated on every page (exercises repeated-header/footer removal);
- a small logo image repeated on every page (exercises the "repeated
  image / logo" filter in app.ingestion.extract_images);
- 1 tiny decorative image on page 1 (exercises the "too small" filter);
- 3 distinct diagram-like images, one per page, each captioned
  "Figure N : ..." (kept, kind="figure");
- 1 icon-range embedded image on page 8 (kept, kind="icon");
- 2 tables ("Tableau des pannes", "Caracteristiques techniques"); the
  "pannes" table uses a narrow column so its header/cells wrap onto two
  lines (exercises the cell-spacing fix in app.ingestion.extract_tables);
- 1 page (7) with a vector-drawn diagram (canvas primitives, no embedded
  raster image) captioned "Figure 4 : ..." (exercises vector-figure
  detection in app.ingestion.extract_vector_figures).

scripts/inspect_pdf.py judges the result "suitable" (native text, >= 3
kept-size images, >= 2 tables).
"""

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DEFAULT_OUTPUT = Path("tests/fixtures/sample_maintenance.pdf")

RUNNING_HEADER = "MDX-100 - Manuel de maintenance"

LOGO_SIZE_PX = 100
TINY_IMAGE_SIZE_PX = 30
DIAGRAM_SIZE_PX = (220, 140)
ICON_IMAGE_SIZE_PX = 120

PANNES_HEADER = ["Panne", "Cause probable", "Action corrective"]
PANNES_ROWS = [
    ["Surchauffe moteur", "Filtre a air colmate", "Remplacer le filtre"],
    ["Vibrations anormales", "Roulement use", "Remplacer le roulement"],
    ["Fuite d'huile", "Joint spi deteriore", "Remplacer le joint"],
]

CARACTERISTIQUES_HEADER = ["Parametre", "Valeur", "Unite"]
CARACTERISTIQUES_ROWS = [
    ["Pression de service", "8", "bar"],
    ["Debit nominal", "120", "L/min"],
    ["Temperature max.", "85", "degC"],
    ["Masse", "45", "kg"],
]


class VectorDiagram(Flowable):
    """4 touching vector shapes (rect-pipe-circle-pipe), black -- a schematic,
    not raster: exercises app.ingestion.extract_vector_figures."""

    def __init__(self, width: float = 190, height: float = 90) -> None:
        super().__init__()
        self._width = width
        self._height = height

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        return self._width, self._height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setStrokeColor(colors.black)
        canvas.setFillColor(colors.black)
        canvas.setLineWidth(1.5)
        canvas.rect(10, 40, 80, 40, stroke=1, fill=0)  # pump body
        canvas.rect(85, 55, 30, 10, stroke=1, fill=1)  # pipe to valve
        canvas.circle(130, 60, 20, stroke=1, fill=0)  # valve
        canvas.rect(145, 55, 35, 10, stroke=1, fill=1)  # pipe to outlet


def _make_png(path: Path, size_px: int | tuple[int, int], color: tuple[int, int, int]) -> Path:
    size = (size_px, size_px) if isinstance(size_px, int) else size_px
    PILImage.new("RGB", size, color=color).save(path, format="PNG")
    return path


def _draw_header_footer(logo_path: Path):
    def _on_page(canvas, doc) -> None:
        canvas.saveState()
        page_width, page_height = A4
        logo_size = 1.2 * cm
        canvas.drawImage(
            str(logo_path),
            page_width - logo_size - 1.5 * cm,
            page_height - logo_size - 1.5 * cm,
            width=logo_size,
            height=logo_size,
            mask="auto",
        )
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.5 * cm, page_height - 1.2 * cm, RUNNING_HEADER)
        canvas.drawString(1.5 * cm, 1.0 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return _on_page


def _wrapping_table() -> Table:
    styles = getSampleStyleSheet()
    header = [Paragraph(text, styles["BodyText"]) for text in PANNES_HEADER]
    rows = [[Paragraph(cell, styles["BodyText"]) for cell in row] for row in PANNES_ROWS]
    flowable = Table([header, *rows], colWidths=[3.5 * cm, 2.2 * cm, 3.5 * cm])
    flowable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]
        )
    )
    return flowable


def _plain_table(header: list[str], rows: list[list[str]]) -> Table:
    flowable = Table([header, *rows])
    flowable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]
        )
    )
    return flowable


def make_pdf(output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem

    logo_path = output_path.with_name(f"{stem}_logo.png")
    tiny_path = output_path.with_name(f"{stem}_tiny.png")
    icon_path = output_path.with_name(f"{stem}_icon.png")
    diagram_paths = [output_path.with_name(f"{stem}_diagram{i}.png") for i in range(1, 4)]

    _make_png(logo_path, LOGO_SIZE_PX, (20, 20, 20))
    _make_png(tiny_path, TINY_IMAGE_SIZE_PX, (180, 180, 180))
    _make_png(icon_path, ICON_IMAGE_SIZE_PX, (120, 120, 200))
    _make_png(diagram_paths[0], DIAGRAM_SIZE_PX, (41, 98, 255))
    _make_png(diagram_paths[1], DIAGRAM_SIZE_PX, (76, 175, 80))
    _make_png(diagram_paths[2], DIAGRAM_SIZE_PX, (255, 87, 34))

    styles = getSampleStyleSheet()

    story = [
        # Page 1: title, intro text, tiny decorative image (too-small filter).
        Paragraph("Manuel de maintenance - Equipement de test MDX-100", styles["Title"]),
        Paragraph(
            "Ce document synthetique decrit une procedure de maintenance preventive "
            "fictive pour l'equipement MDX-100, utilisee uniquement pour les tests "
            "automatises du pipeline d'extraction.",
            styles["BodyText"],
        ),
        Spacer(1, 0.3 * cm),
        Image(str(tiny_path), width=0.4 * cm, height=0.4 * cm),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "L'equipement MDX-100 est un groupe hydraulique destine aux lignes de "
            "production continues. Avant toute intervention, consigner l'equipement "
            "conformement a la procedure de securite en vigueur.",
            styles["BodyText"],
        ),
        PageBreak(),
        # Page 2: text + diagram 1.
        Paragraph("Principe de fonctionnement", styles["Heading2"]),
        Paragraph(
            "Le circuit hydraulique alimente le verin principal via une pompe a "
            "cylindree variable. Le schema ci-dessous presente l'implantation "
            "generale des composants.",
            styles["BodyText"],
        ),
        Spacer(1, 0.4 * cm),
        Image(str(diagram_paths[0]), width=6 * cm, height=4 * cm),
        Paragraph("Figure 1 : Schema du circuit hydraulique principal", styles["Italic"]),
        PageBreak(),
        # Page 3: Tableau des pannes, narrow middle column -> wrapped cells
        # (kept on its own page -- see extract_tables.py docstring for why).
        Paragraph("Tableau 1 : Tableau des pannes", styles["Heading2"]),
        _wrapping_table(),
        PageBreak(),
        # Page 4: text + diagram 2.
        Paragraph("Emplacement des capteurs", styles["Heading2"]),
        Paragraph(
            "Les capteurs de pression sont repartis le long du circuit "
            "principal, comme indique ci-dessous.",
            styles["BodyText"],
        ),
        Spacer(1, 0.4 * cm),
        Image(str(diagram_paths[1]), width=6 * cm, height=4 * cm),
        Paragraph("Figure 2 : Emplacement des capteurs de pression", styles["Italic"]),
        PageBreak(),
        # Page 5: text + diagram 3.
        Paragraph("Procedure de controle visuel", styles["Heading2"]),
        Paragraph(
            "Controler l'absence de fuite au niveau des raccords, l'etat des "
            "flexibles et le niveau d'huile dans le reservoir avant chaque "
            "demarrage.",
            styles["BodyText"],
        ),
        Spacer(1, 0.4 * cm),
        Image(str(diagram_paths[2]), width=6 * cm, height=4 * cm),
        Paragraph("Figure 3 : Points de controle visuel quotidien", styles["Italic"]),
        PageBreak(),
        # Page 6: Caracteristiques techniques + closing text.
        Paragraph("Tableau 2 : Caracteristiques techniques", styles["Heading2"]),
        _plain_table(CARACTERISTIQUES_HEADER, CARACTERISTIQUES_ROWS),
        Spacer(1, 0.4 * cm),
        Paragraph(
            "Pour toute anomalie non couverte par ce document, contacter le "
            "service de maintenance avant de remettre l'equipement en service.",
            styles["BodyText"],
        ),
        PageBreak(),
        # Page 7: vector-drawn diagram, no embedded raster image, no table.
        Paragraph("Schema de principe (trace vectoriel)", styles["Heading2"]),
        Spacer(1, 0.4 * cm),
        VectorDiagram(),
        Paragraph("Figure 4 : Schema de principe pompe-vanne", styles["Italic"]),
        PageBreak(),
        # Page 8: icon-range embedded image + closing text.
        Paragraph("Pictogramme de securite", styles["Heading2"]),
        Spacer(1, 0.4 * cm),
        Image(str(icon_path), width=2.5 * cm, height=2.5 * cm),
        Paragraph("Figure 5 : Pictogramme de securite generique", styles["Italic"]),
        Spacer(1, 0.4 * cm),
        Paragraph(
            "Fin du document synthetique de test.",
            styles["BodyText"],
        ),
    ]

    on_page = _draw_header_footer(logo_path)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    for temp_image in (tiny_path, logo_path, icon_path, *diagram_paths):
        temp_image.unlink()

    return output_path


def main() -> None:
    path = make_pdf()
    print(f"Generated {path}")


if __name__ == "__main__":
    main()
