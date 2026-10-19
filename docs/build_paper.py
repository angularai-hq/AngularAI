"""Build the short paper. Run: uv run python docs/build_paper.py"""

import sys
from collections.abc import Callable
from pathlib import Path

from reportlab.lib.colors import Color, black, white
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

_MARGIN = 64
_INK = black
_QUIET = Color(0.32, 0.32, 0.32)
_HAIR = Color(0.75, 0.75, 0.75)
_PAGE = A4
Paint = Callable[["object", float, float], None]


def build(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(path),
        pagesize=_PAGE,
        title="AngularAI",
        author="Teodor Walter Vido",
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=62,
        bottomMargin=54,
    )
    doc.pageCompression = 0
    frame = Frame(
        _MARGIN,
        54,
        _PAGE[0] - 2 * _MARGIN,
        _PAGE[1] - 62 - 54,
        id="body",
        showBoundary=0,
    )
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_paint_page)])
    styles = _styles()
    story: list[Flowable] = [
        _Title(),
        Spacer(1, 16),
        Paragraph(_INTRO[0], styles["prose"]),
        Paragraph(_INTRO[1], styles["prose"]),
        _Figure(132, _membrane),
        Paragraph(
            "Figure 1. The membrane. The model sees the workspace. The file keeps the rest.",
            styles["caption"],
        ),
        Paragraph(_INTRO[2], styles["prose"]),
        Paragraph("What a pass writes", styles["section"]),
        Paragraph(_WRITE[0], styles["prose"]),
        Paragraph(_WRITE[1], styles["prose"]),
        _outcomes(),
        Spacer(1, 8),
        Paragraph(_WRITE[2], styles["prose"]),
        _Figure(128, _one_pass),
        Paragraph(
            "Figure 2. One question, after the fact is on disk. "
            "Catering shares no words with the question, so it is not sent.",
            styles["caption"],
        ),
        Paragraph(_WRITE[3], styles["prose"]),
        _Figure(64, _budget),
        Paragraph(
            "Figure 3. The budget. The filled part is one call. The open part stayed on disk.",
            styles["caption"],
        ),
        Paragraph("The session", styles["section"]),
        Paragraph(_SESSION[0], styles["prose"]),
        Paragraph(_SESSION[1], styles["prose"]),
    ]
    doc.build(story)


_INTRO = (
    "A language model answers inside one call. When the call ends, that window is gone. "
    "AngularAI sits in front of the API you already use. The model still answers. "
    "What the agent has learned stays in a file on the machine that asked.",
    "Each call is packed to a fixed budget. Attention ranks the file against the words "
    "of the question, and the workspace takes only what fits. The rest is not sent. "
    "It can return on a later pass, when a question shares its words. "
    "The vendor's weights do not change. The plasticity is the cortex.",
    "The budget is an estimate: four ASCII characters to one token. "
    "A wide character (CJK, kana, Hangul, fullwidth) is one, two above U+FFFF, "
    "like most emoji. Any other character is half. Rounded up. "
    "AngularAI enforces that estimate. It is not the provider's tokenizer. "
    "A short call and a long life can share one budget. The file may outgrow it. The call may not.",
)

_WRITE = (
    "A secret in the input is refused before the call, and before any line is written. "
    "The key is not a field in the cortex. It is read from the environment when you speak. "
    "OpenAI, Anthropic, or a compatible endpoint: the same rule.",
    "The reply comes back through a value lock. A phrase you forbade, or a secret in the reply, "
    "holds it. You see one sentence: Value lock held this reply. The held text is not stored. "
    "A reply that was not judged cannot leave.",
    "A finished pass appends one line to the log. Folding that log produces the traces "
    "and the episodes. A trace the model was not shown cannot be superseded. "
    "A superseded trace stays in the file and leaves the workspace.",
    "Recall is that same overlap. What color is the notebook? meets The notebook is blue. "
    "What shade is it? does not. A long episode is stored in full. The call is shown "
    "the sentences that share words with the question, and the opening stretch when none do. "
    "One pass keeps at most twenty-four traces. "
    "Each trace is at most two hundred and forty characters.",
)

_SESSION = (
    "In a terminal, angularai opens this cortex. The transcript is the sitting. "
    "The status line is the last pass: which model, how much of the budget was sent, "
    "how many traces are current. angularai say is one turn with no screen. "
    "The reply is the only thing it prints.",
    "The directory is ~/.angularai/default unless you point elsewhere. "
    "cortex.json is the wiring, the budget, and the locks. "
    "passes.jsonl is the life. It is append-only, and it is readable only by you.",
)


class _Title(Flowable):
    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        self.height = 74
        return avail_width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFillColor(_INK)
        canvas.setFont("Helvetica-Bold", 14)
        cursor = 0.0
        for glyph in "ANGULARAI":
            canvas.drawString(cursor, 48, glyph)
            cursor += canvas.stringWidth(glyph, "Helvetica-Bold", 14) + 2.2
        canvas.setFont("Helvetica", 11)
        canvas.drawString(0, 28, "A cortex in front of a language model")
        canvas.setFillColor(_QUIET)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(0, 8, "Teodor Walter Vido")
        canvas.drawRightString(self.width, 8, "2026")
        canvas.setStrokeColor(_INK)
        canvas.setLineWidth(0.6)
        canvas.line(0, 0, self.width, 0)


class _Figure(Flowable):
    def __init__(self, height: float, paint: Paint) -> None:
        super().__init__()
        self._height = height
        self._paint = paint

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        self.height = self._height
        return avail_width, self._height

    def draw(self) -> None:
        self._paint(self.canv, float(self.width), float(self.height))


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "prose": ParagraphStyle(
            "prose",
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=_INK,
            alignment=TA_JUSTIFY,
            spaceAfter=9,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=_INK,
            spaceBefore=8,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=_QUIET,
            spaceBefore=3,
            spaceAfter=10,
        ),
    }


def _outcomes() -> Table:
    rows = [
        ["Completed", "The reply, and up to twenty-four traces."],
        ["Held", "The hold sentence. No new traces."],
        ["Failed", "The episode and the error. No new traces."],
        ["Refused", "A secret in the input. Nothing is written."],
    ]
    table = Table(rows, colWidths=[78, 386])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, -1), _INK),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, _HAIR),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 8),
                ("RIGHTPADDING", (1, 0), (1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _membrane(canvas: object, width: float, height: float) -> None:
    gap = 28.0
    box_w = (width - 2 * gap) / 3
    box_h = 86.0
    y = 30.0
    boxes = (
        ("On disk", ("passes.jsonl", "episodes", "traces", "superseded stay")),
        ("Workspace", ("word overlap", "what still fits", "at most the budget")),
        ("The model", ("one reply", "traces to keep", "or a hold")),
    )
    xs = (0.0, box_w + gap, 2 * (box_w + gap))
    for x, (title, lines) in zip(xs, boxes, strict=True):
        _box(canvas, x, y, box_w, box_h, title, lines)
    mid = y + box_h / 2
    _arrow(canvas, xs[0] + box_w + 3, xs[1] - 3, mid, left=False)
    _arrow(canvas, xs[1] + box_w + 3, xs[2] - 3, mid, left=False)
    _caption_at(canvas, (xs[0] + box_w + xs[1]) / 2, mid + 8, "rank")
    _caption_at(canvas, (xs[1] + box_w + xs[2]) / 2, mid + 8, "send")
    _return(canvas, xs[0] + 10, xs[2] + box_w - 10, y, 14)


def _one_pass(canvas: object, width: float, height: float) -> None:
    gap = 22.0
    box_w = (width - 2 * gap) / 3
    box_h = 108.0
    y = 8.0
    xs = (0.0, box_w + gap, 2 * (box_w + gap))
    titles = ("On disk", "This call", "Reply")
    for x, title in zip(xs, titles, strict=True):
        _frame(canvas, x, y, box_w, box_h, title)
    _line(canvas, xs[0] + 8, y + box_h - 34, "The notebook is blue.", quiet=False)
    _line(canvas, xs[0] + 8, y + box_h - 50, "Catering is at noon.", quiet=True)
    _line(canvas, xs[1] + 8, y + box_h - 34, "TRACES", quiet=True)
    _line(canvas, xs[1] + 8, y + box_h - 48, "The notebook is blue.", quiet=False)
    _line(canvas, xs[1] + 8, y + box_h - 66, "INPUT", quiet=True)
    _line(canvas, xs[1] + 8, y + box_h - 80, "Which notebook?", quiet=False)
    _line(canvas, xs[2] + 8, y + box_h - 34, "The notebook is blue.", quiet=False)
    mid = y + box_h / 2
    _arrow(canvas, xs[0] + box_w + 2, xs[1] - 2, mid, left=False)
    _arrow(canvas, xs[1] + box_w + 2, xs[2] - 2, mid, left=False)


def _budget(canvas: object, width: float, height: float) -> None:
    bar_y = 26.0
    bar_h = 16.0
    filled = width * 0.2
    pen = _pen(canvas)
    pen.setStrokeColor(_INK)
    pen.setLineWidth(0.8)
    pen.setFillColor(white)
    pen.rect(0, bar_y, width, bar_h, stroke=1, fill=0)
    pen.setFillColor(_INK)
    pen.rect(0, bar_y, filled, bar_h, stroke=0, fill=1)
    pen.setFillColor(_INK)
    pen.setFont("Helvetica", 8)
    pen.drawString(0, 8, "this call")
    pen.drawRightString(width, 8, "the rest, on disk")


def _box(
    canvas: object,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...],
) -> None:
    _frame(canvas, x, y, width, height, title)
    for index, line in enumerate(lines):
        _line(canvas, x + 8, y + height - 34 - index * 13, line, quiet=False)


def _frame(canvas: object, x: float, y: float, width: float, height: float, title: str) -> None:
    pen = _pen(canvas)
    pen.setStrokeColor(_INK)
    pen.setFillColor(white)
    pen.setLineWidth(0.8)
    pen.rect(x, y, width, height, stroke=1, fill=0)
    pen.setFillColor(_INK)
    pen.setFont("Helvetica-Bold", 8)
    pen.drawString(x + 8, y + height - 14, title)
    pen.setStrokeColor(_HAIR)
    pen.setLineWidth(0.4)
    pen.line(x + 8, y + height - 18, x + width - 8, y + height - 18)


def _line(canvas: object, x: float, y: float, text: str, *, quiet: bool) -> None:
    pen = _pen(canvas)
    pen.setFillColor(_QUIET if quiet else _INK)
    pen.setFont("Helvetica", 8)
    pen.drawString(x, y, text)


def _caption_at(canvas: object, x: float, y: float, text: str) -> None:
    pen = _pen(canvas)
    pen.setFillColor(_INK)
    pen.setFont("Helvetica", 7)
    pen.drawCentredString(x, y, text)


def _return(canvas: object, left: float, right: float, box_bottom: float, rail: float) -> None:
    pen = _pen(canvas)
    pen.setStrokeColor(_INK)
    pen.setFillColor(_INK)
    pen.setLineWidth(0.9)
    pen.line(right, box_bottom, right, rail)
    pen.line(right, rail, left + 7, rail)
    path = pen.beginPath()
    path.moveTo(left, rail)
    path.lineTo(left + 7, rail + 3)
    path.lineTo(left + 7, rail - 3)
    path.close()
    pen.drawPath(path, stroke=0, fill=1)
    _caption_at(canvas, (left + right) / 2, rail + 8, "append one line")


def _arrow(canvas: object, start: float, end: float, y: float, *, left: bool) -> None:
    pen = _pen(canvas)
    pen.setStrokeColor(_INK)
    pen.setFillColor(_INK)
    pen.setLineWidth(0.7)
    tip = end if left else end
    tail = start
    pen.line(tail, y, tip + (6 if left else -6), y)
    path = pen.beginPath()
    if left:
        path.moveTo(tip, y)
        path.lineTo(tip + 6, y + 3)
        path.lineTo(tip + 6, y - 3)
    else:
        path.moveTo(tip, y)
        path.lineTo(tip - 6, y + 3)
        path.lineTo(tip - 6, y - 3)
    path.close()
    pen.drawPath(path, stroke=0, fill=1)


def _pen(canvas: object) -> object:
    return canvas


def _paint_page(canvas: object, doc: object) -> None:
    if int(getattr(doc, "page", 1)) > 1:
        pen = _pen(canvas)
        pen.saveState()
        width, height = _PAGE
        pen.setStrokeColor(_INK)
        pen.setLineWidth(0.4)
        pen.line(_MARGIN, height - 40, width - _MARGIN, height - 40)
        pen.setFillColor(_QUIET)
        pen.setFont("Helvetica", 8)
        pen.drawString(_MARGIN, height - 34, "AngularAI")
        pen.drawRightString(width - _MARGIN, height - 34, "A cortex in front of a language model")
        pen.restoreState()
    _footer(canvas, doc)


def _footer(canvas: object, doc: object) -> None:
    pen = _pen(canvas)
    pen.saveState()
    width, _height = _PAGE
    pen.setStrokeColor(_INK)
    pen.setLineWidth(0.4)
    pen.line(_MARGIN, 40, width - _MARGIN, 40)
    pen.setFillColor(_QUIET)
    pen.setFont("Helvetica", 8)
    pen.drawString(_MARGIN, 26, "Teodor Walter Vido")
    pen.drawRightString(width - _MARGIN, 26, str(getattr(doc, "page", "")))
    pen.restoreState()


def main() -> None:
    if len(sys.argv) > 1:
        destination = Path(sys.argv[1])
    else:
        destination = Path(__file__).with_name("angularai.pdf")
    build(destination)


if __name__ == "__main__":
    main()
