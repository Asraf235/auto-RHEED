"""Electron wavelength from beam energy (relativistic de Broglie)."""


def lambda_angstrom(v_kev: float) -> float:
    """Relativistic de Broglie wavelength (Å) for a beam energy in keV."""
    v_ev = v_kev * 1000.0
    return 12.2643 / (v_ev * (1 + 0.9788e-6 * v_ev)) ** 0.5


def lambda_cm(v_kev: float) -> float:
    """Same as lambda_angstrom but in cm — convenient for d = λL/Δx math
    where L is in cm and Δx (in cm) comes from pixel_count * scale_cm_px."""
    return lambda_angstrom(v_kev) * 1e-8
