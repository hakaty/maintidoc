"""French number formatting helpers (section 0i of the UI pixel-match task):
the reference uses French decimals/units -- "14,9 s", "80,8 %", "0,976" --
never raw milliseconds or dot decimals."""


def fr_seconds(value_ms: float | None) -> str | None:
    if value_ms is None:
        return None
    return f"{value_ms / 1000:.1f}".replace(".", ",") + " s"


def fr_percent(fraction: float | None) -> str | None:
    if fraction is None:
        return None
    return f"{fraction * 100:.1f}".replace(".", ",") + " %"


def fr_decimal(value: float | None, digits: int = 3) -> str | None:
    if value is None:
        return None
    return f"{value:.{digits}f}".replace(".", ",")
