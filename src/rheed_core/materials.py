"""
Materials database and expected in-plane d-spacing lookup.

Mirrors the MAT_DB / expectedD() used client-side in the Streak Spacing
and Growth Analysis tools (rheed_webapp/templates/index.html), so the
same reference values are available to non-browser consumers (MCP tools,
scripts) without needing to scrape the JS.
"""
import math

MATERIALS = {
    # ── Perovskite oxide substrates (cubic / pseudo-cubic) ──────────────
    "SrTiO3":      {"a": 3.905, "system": "cubic"},
    "NbSTO":       {"a": 3.905, "system": "cubic"},   # Nb-doped SrTiO3
    "BaTiO3":      {"a": 3.994, "system": "cubic"},
    "LaAlO3":      {"a": 3.790, "system": "cubic"},
    "LSAT":        {"a": 3.868, "system": "cubic"},   # (LaAlO3)0.3(Sr2AlTaO6)0.7
    "DyScO3":      {"a": 3.944, "system": "cubic"},   # pseudo-cubic (orthorhombic in reality)
    "GdScO3":      {"a": 3.967, "system": "cubic"},   # pseudo-cubic
    "NdGaO3":      {"a": 3.858, "system": "cubic"},   # pseudo-cubic
    "NdScO3":      {"a": 4.006, "system": "cubic"},   # pseudo-cubic
    "PrScO3":      {"a": 4.022, "system": "cubic"},   # pseudo-cubic
    "SmScO3":      {"a": 3.989, "system": "cubic"},   # pseudo-cubic
    "TbScO3":      {"a": 3.951, "system": "cubic"},   # pseudo-cubic
    "LaGaO3":      {"a": 3.892, "system": "cubic"},   # pseudo-cubic
    "LaSrAlO4":    {"a": 3.756, "system": "cubic"},
    "KTaO3":       {"a": 3.989, "system": "cubic"},
    "CaTiO3":      {"a": 3.795, "system": "cubic"},   # pseudo-cubic
    "PbTiO3":      {"a": 3.904, "system": "cubic"},
    "BiFeO3":      {"a": 3.965, "system": "cubic"},
    "LSMO":        {"a": 3.873, "system": "cubic"},   # La0.7Sr0.3MnO3
    "LaMnO3":      {"a": 3.946, "system": "cubic"},
    "LaNiO3":      {"a": 3.838, "system": "cubic"},
    "LaFeO3":      {"a": 3.930, "system": "cubic"},
    "SrVO3":       {"a": 3.842, "system": "cubic"},
    "SrRuO3":      {"a": 3.923, "system": "cubic"},   # pseudo-cubic
    "SrMnO3":      {"a": 3.805, "system": "cubic"},
    "YBa2Cu3O7":   {"a": 3.823, "system": "cubic"},

    # ── Rock-salt oxides ─────────────────────────────────────────────────
    "MgO":         {"a": 4.211, "system": "cubic"},
    "NiO":         {"a": 4.177, "system": "cubic"},
    "CoO":         {"a": 4.260, "system": "cubic"},

    # ── Spinels / fluorites ──────────────────────────────────────────────
    "MgAl2O4":     {"a": 8.083, "system": "cubic"},
    "YSZ":         {"a": 5.139, "system": "cubic"},   # yttria-stabilized zirconia
    "CeO2":        {"a": 5.411, "system": "cubic"},

    # ── Elemental / III-V / II-VI semiconductors (cubic, zinc-blende
    #    or diamond) ────────────────────────────────────────────────────
    "Si":          {"a": 5.431, "system": "cubic"},
    "Ge":          {"a": 5.658, "system": "cubic"},
    "GaAs":        {"a": 5.653, "system": "cubic"},
    "GaP":         {"a": 5.450, "system": "cubic"},
    "GaSb":        {"a": 6.096, "system": "cubic"},
    "InAs":        {"a": 6.058, "system": "cubic"},
    "InP":         {"a": 5.869, "system": "cubic"},
    "InSb":        {"a": 6.479, "system": "cubic"},
    "AlAs":        {"a": 5.661, "system": "cubic"},
    "AlSb":        {"a": 6.136, "system": "cubic"},
    "ZnSe":        {"a": 5.668, "system": "cubic"},
    "ZnTe":        {"a": 6.103, "system": "cubic"},
    "CdTe":        {"a": 6.482, "system": "cubic"},
    "3C-SiC":      {"a": 4.360, "system": "cubic"},

    # ── Hexagonal substrates ─────────────────────────────────────────────
    "Al2O3":       {"a": 4.759, "c": 12.991, "system": "hexagonal"},  # sapphire
    "GaN":         {"a": 3.189, "c": 5.185,  "system": "hexagonal"},
    "AlN":         {"a": 3.112, "c": 4.982,  "system": "hexagonal"},
    "ZnO":         {"a": 3.250, "c": 5.207,  "system": "hexagonal"},
    "4H-SiC":      {"a": 3.073, "c": 10.053, "system": "hexagonal"},
    "6H-SiC":      {"a": 3.081, "c": 15.117, "system": "hexagonal"},
    "TiO2":        {"a": 4.594, "c": 2.959,  "system": "hexagonal"},  # rutile, treated approx.
}


def find_material_key(raw: str):
    """Case-insensitive lookup, ignoring the user's exact key formatting."""
    norm = raw.strip().lower()
    for key in MATERIALS:
        if key.lower() == norm:
            return key
    return None


def expected_d(material: str, zone: str):
    """
    Expected in-plane d-spacing (Å) for a material + RHEED zone axis.
    `zone` is one of '100','010','001','110','111' (string, no brackets).
    Returns None if the material/zone combination isn't recognized.
    """
    m = MATERIALS.get(material)
    if m is None:
        return None
    a = m["a"]
    if m["system"] == "cubic":
        if zone in ("100", "001", "010"):
            return a
        if zone in ("110", "111"):
            return a / math.sqrt(2)
    elif m["system"] == "hexagonal":
        if zone in ("100", "010"):
            return a * math.sqrt(3) / 2
        if zone == "001":
            return a
    return None
