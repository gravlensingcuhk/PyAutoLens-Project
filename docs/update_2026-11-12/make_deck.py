"""Build the 2026-11-12 status deck. Run: uv run python docs/update_2026-11-12/make_deck.py"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
IMG = HERE / "img"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def bullets(title, items, sub=None):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0))
    tb.text_frame.text = title
    tb.text_frame.paragraphs[0].font.size = Pt(32)
    tb.text_frame.paragraphs[0].font.bold = True
    body = s.shapes.add_textbox(Inches(0.7), Inches(1.4), Inches(12.0), Inches(5.6))
    tf = body.text_frame
    tf.word_wrap = True
    for i, (lvl, text) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = lvl
        p.font.size = Pt(20 if lvl == 0 else 17)
        p.space_after = Pt(8)
    if sub:
        sb = s.shapes.add_textbox(Inches(0.7), Inches(6.9), Inches(12.0), Inches(0.5))
        sb.text_frame.text = sub
        sb.text_frame.paragraphs[0].font.size = Pt(12)
        sb.text_frame.paragraphs[0].font.italic = True
    return s


def image_slide(title, images, note=None):
    """images: list of (path, caption) laid out horizontally."""
    from PIL import Image as PILImage
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    tb.text_frame.text = title
    tb.text_frame.paragraphs[0].font.size = Pt(28)
    tb.text_frame.paragraphs[0].font.bold = True
    n = len(images)
    slot = 12.5 / n
    top = 1.4
    for i, (path, caption) in enumerate(images):
        w_in = min(slot - 0.3, 6.5)
        pw, ph = PILImage.open(path).size
        h_in = w_in * ph / pw
        if h_in > 5.0:
            h_in = 5.0
            w_in = h_in * pw / ph
        left = Inches(0.4 + i * slot)
        s.shapes.add_picture(str(path), left, Inches(top), width=Inches(w_in))
        cb = s.shapes.add_textbox(left, Inches(top + h_in + 0.15), Inches(slot - 0.2), Inches(1.2))
        cb.text_frame.text = caption
        cb.text_frame.word_wrap = True
        cb.text_frame.paragraphs[0].font.size = Pt(13)
    if note:
        nb = s.shapes.add_textbox(Inches(0.5), Inches(6.55), Inches(12.3), Inches(0.8))
        nb.text_frame.text = note
        nb.text_frame.word_wrap = True
        nb.text_frame.paragraphs[0].font.size = Pt(15)
        nb.text_frame.paragraphs[0].font.italic = True
    return s


def fit_slide(title, fit, zoom, facts):
    """Full fit figure left, source-plane zoom right, key numbers below."""
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.9))
    tb.text_frame.text = title
    tb.text_frame.paragraphs[0].font.size = Pt(28)
    tb.text_frame.paragraphs[0].font.bold = True
    s.shapes.add_picture(str(fit), Inches(0.3), Inches(1.15), height=Inches(5.6))
    s.shapes.add_picture(str(zoom), Inches(8.15), Inches(1.6), height=Inches(3.9))
    cb = s.shapes.add_textbox(Inches(8.15), Inches(5.6), Inches(4.9), Inches(1.8))
    tf = cb.text_frame
    tf.word_wrap = True
    for i, t in enumerate(facts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t
        p.font.size = Pt(14)
        p.space_after = Pt(4)
    return s


# --- Slide 1: title ---
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(2.5))
tf = tb.text_frame
tf.text = "COWLS strong-lens modelling — status"
tf.paragraphs[0].font.size = Pt(40)
tf.paragraphs[0].font.bold = True
p = tf.add_paragraph(); p.text = "COSJ100018+022138 (J18) · COSJ100024+021749 (J24) · COSJ095921+020638 (J0959)"; p.font.size = Pt(20)
p = tf.add_paragraph(); p.text = "12 November 2026"; p.font.size = Pt(16)

# --- Slide 2: goal and problems ---
bullets("Goal and the three problems", [
    (0, "Goal: smooth mass models stable enough that a subhalo search can run on the residuals."),
    (0, "Problem 1 — the mass model wants a large quadrupole:"),
    (1, "all three lenses fit large external shear (|γ| ≈ 0.1–0.2) under a power-law model"),
    (1, "real external shear (groups), or the power law absorbing internal structure (stars+dark split)?"),
    (0, "Problem 2 — source reconstruction pathologies:"),
    (1, "J18's source is tiny (~0.05″) and highly magnified → pixelized reconstruction fragments into fake clumps"),
    (1, "J0959 has compact (AGN-like) knots some meshes fail to reconstruct at all"),
    (0, "Problem 3 — nuisance systematics: PSF mismatch in lens cores, noise maps, nearby galaxies, practical run failures"),
])

# --- Slide 3: ruled out ---
bullets("What we have ruled out", [
    (0, "Group environment alone ≠ the shear: J18 and J24 sit in spectroscopically confirmed groups, but"),
    (1, "group shear is only ~1/3–1/2 of the fitted value (|γ| ≈ 0.05–0.09, roughly the right direction)"),
    (1, "J0959 has NO detected group — its quadrupole cannot be environmental"),
    (0, "The data are not broken: PSFs, noise maps, masks, positions all audited"),
    (1, "PSF-error noise floor fixed J24's core residuals; shear unchanged"),
    (1, "J18's shear orientation repeats in a second band (F277W) — not a single-band artefact"),
    (0, "The old 13-day non-convergence is fixed (discontinuous regularization bookkeeping + GPU sparse path)"),
    (0, "ConstantSplit regularization is ruled out for compact sources (irons the source flat; ±10σ knot residuals)"),
    (0, "Lens light stealing arc flux is fixed (single-component MGE light model)"),
])

# --- Slide 4: environment figure ---
image_slide("The group environment", [
    (IMG / "group_fields_F444W.png", "90″ F444W mosaics with spectroscopically confirmed group members marked: J18 (z=1.53, 11 members) and J24 (z=0.36, 8 members)."),
], note="Group shear |γ| ≈ 0.05–0.09 in roughly the fitted direction for both; J0959: no group found.")

# --- Slide 5: shear table ---
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
tb.text_frame.text = "Where does the quadrupole live? External shear |γ|"
tb.text_frame.paragraphs[0].font.size = Pt(28)
tb.text_frame.paragraphs[0].font.bold = True
rows = [
    ("Lens", "Power-law (EPL)", "Stars + dark halo", "Group supplies"),
    ("J18", "0.20", "0.04 (collapsed)", "≈ 0.06–0.09"),
    ("J24", "0.17", "0.15–0.18 (survives)", "≈ 0.05"),
    ("J0959", "0.14", "0.09 (partly absorbed)", "≈ 0 (no group)"),
]
tbl = s.shapes.add_table(len(rows), 4, Inches(0.8), Inches(1.5), Inches(11.5), Inches(2.6)).table
for c, h in enumerate(rows[0]):
    cell = tbl.cell(0, c); cell.text = h
    cell.text_frame.paragraphs[0].font.bold = True
    cell.text_frame.paragraphs[0].font.size = Pt(18)
for r, row in enumerate(rows[1:], start=1):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c); cell.text = v
        cell.text_frame.paragraphs[0].font.size = Pt(18)
nb = s.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(11.5), Inches(2.4))
tf = nb.text_frame; tf.word_wrap = True
for i, t in enumerate([
    "Stars+dark test: stellar component fixed to the lens light, NFW halo with near-spherical prior, small centre freedom.",
    "J18: shear collapses to the group level → the power law was faking it (internal quadrupole).",
    "J24: shear survives in every variant → genuine external (group) + elliptical halo (e ≈ 0.2).",
    "J0959: halved but not eliminated; nothing environmental to supply it — open question.",
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = t; p.font.size = Pt(16); p.space_after = Pt(6)

# --- Slides 6-8: per-lens fits ---
fit_slide("J18 (z=1.53) — stars+dark fit: shear collapses",
          IMG / "j18_mld_rtu_fit.png", IMG / "j18_mld_rtu_source_zoom.png",
          ["γ = (−0.02, +0.04),  |γ| ≈ 0.04  (EPL wanted 0.20)",
           "Halo near-round (e ≈ 0.04), centre offset 0.06″",
           "M₂₀₀ ≈ 3.9×10¹³ M⊙",
           "Residuals at noise level; source compact",
           "BUT: source-plane result not yet reproduced on the second mesh family"])

fit_slide("J24 (z=0.36) — stars+dark fit: shear survives",
          IMG / "j24_mld_rtu_fit.png", IMG / "j24_mld_rtu_source_zoom.png",
          ["γ = (−0.10, +0.11),  |γ| ≈ 0.15  (two mesh families agree)",
           "Halo elliptical (e ≈ 0.2)",
           "Group supplies ≈ 0.05 in the right direction — needs more",
           "Good fit on every variant; closest to done"])

fit_slide("J0959 (spec-z, no group) — stars+dark fit",
          IMG / "j0959_mld_rtu_fit.png", IMG / "j0959_mld_rtu_source_zoom.png",
          ["γ = (−0.005, +0.089),  |γ| ≈ 0.09  (EPL wanted 0.14)",
           "Compact knots mostly reconstructed now",
           "M₂₀₀ ≈ 4.9×10¹³ M⊙",
           "Einstein radius 0.86″ vs 0.71″ in Anguita+09 — unresolved",
           "Compact residual spots remain (lensed AGN?)"])

# --- Slide 9: the source reconstruction problem ---
image_slide("The source reconstruction is the systematic (J18)", [
    (IMG / "j18_cs_sp2_source_zoom.png", "ConstantSplit: over-smoothed to one blob → ±10σ knot residuals. Ruled out."),
    (IMG / "j18_as_sp2_source_zoom.png", "AdaptSplit: clean image fit, but clumps scattered on the caustics — fragmentation."),
    (IMG / "j18_rtu_sp2_source_zoom.png", "RTU (current production): contiguous source, but mesh never zooms (structural)."),
], note="The inferred mass model moves 10–20% (slope, γ₂) purely from source-reconstruction freedom. Setup is now chosen by source-plane behaviour, not image-plane residuals.")

# --- Slide 10: next steps ---
bullets("Next steps", [
    (0, "J18 (critical path): pick the source scheme that gives a compact, stable source;"),
    (1, "AdaptSplitPower (λ² regularization) test running now; mesh-independence of the shear collapse still open"),
    (0, "J0959: finish wide-shear chains; understand r_E offset vs Anguita+09; AGN point-source component if knots persist"),
    (0, "J24: essentially done — write down the group + elliptical-halo shear story"),
    (0, "Then: freeze the production chain per lens → merge to master → HTCondor subhalo tiling"),
    (0, "Icebox: model the confirmed group members explicitly; joint multi-band fits; multipoles on the stars+dark model"),
])

out = HERE / "cowls_status_2026-11-12.pptx"
prs.save(out)
print("saved", out)
