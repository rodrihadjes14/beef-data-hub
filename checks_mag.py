# checks_mag.py
from fastapi import APIRouter, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import date as Date
import os, re, httpx

router = APIRouter(prefix="/checks", tags=["checks"])

# --- Seguridad simple por x-api-key ---
def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("BEEF_HUB_API_KEY", "")
    if not expected or x_api_key != expected:
        # Usamos JSON limpio en vez de excepción para no llenar logs
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")

# --- Modelos de respuesta ---
class CanalRange(BaseModel):
    min: float
    avg: float
    max: float

class MagToCanalResponse(BaseModel):
    ok: bool
    source: str
    date: str
    rendimiento: float
    unit: str
    canal_range: CanalRange
    note: str | None = None

# --- Helpers de red y parsing ---
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "15"))
HTTP_UA = os.environ.get("HTTP_UA", "Mozilla/5.0 (compatible; BeefDataHub/1.0; +https://beef-data-hub)")

MAG_URL_PRIMARY = os.environ.get(
    "MAG_URL_PRIMARY",
    "https://www.mercadoagroganadero.com.ar/dll/hacienda1.dll/haciinfo000002"  # Precios por Categoría (RUCA)
)
MAG_URL_FALLBACK = os.environ.get(
    "MAG_URL_FALLBACK",
    "https://ganaderiaynegocios.com/precios-mercado-agroganadero-canuelas/"     # Agregador confiable
)

_NUM_RE = r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)"  # 3.123,45 o 2.900 etc.

def _to_float_ars(s: str) -> float:
    s = s.strip()
    s = s.replace(".", "").replace(",", ".")
    return float(s)

def _extract_promedio_novillos_from_primary(html: str) -> float | None:
    """
    Busca en la página oficial del MAG una fila 'NOVILLOS ...' y extrae el 'Promedio' ($/kg vivo).
    El layout puede cambiar; probamos varios patrones.
    """
    txt = " ".join(html.split())  # colapsar espacios
    # Patrón genérico: ... NOVILLOS ... Min ... Max ... Promedio <num> ...
    patterns = [
        rf"NOVILLOS.*?M[ií]nimo.*?{_NUM_RE}.*?M[aá]ximo.*?{_NUM_RE}.*?Prom(?:\.|edio)?\s*{_NUM_RE}",
        rf"NOVILLOS[^<]*{_NUM_RE}[^<]*{_NUM_RE}[^<]*{_NUM_RE}"  # fila con 3 números seguidos (min/max/prom)
    ]
    for pat in patterns:
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            # Tomamos el último número capturado como 'Promedio'
            prom = m.groups()[-1]
            return _to_float_ars(prom)
    return None

def _extract_promedio_novillos_from_fallback(html: str) -> float | None:
    """
    En el fallback (agregador) aparecen varias filas 'NOVILLOS ...' con Min/Max/Promedio.
    Tomamos el promedio de los 'Promedios' listados para NOVILLOS.
    """
    txt = " ".join(html.split())
    # filas que contengan 'NOVILLOS' y luego tres números; el 3ro lo tomamos como 'Promedio'.
    rows = re.findall(rf"(NOVILLOS[^<]{{0,120}}?){_NUM_RE}[^<]{{0,40}}?{_NUM_RE}[^<]{{0,40}}?{_NUM_RE}", txt, re.IGNORECASE)
    proms: list[float] = []
    for row in rows:
        try:
            prom = row[-1]  # tercer número capturado
            proms.append(_to_float_ars(prom))
        except Exception:
            continue
    if proms:
        # promedio simple de los 'Promedios' de NOVILLOS
        return round(sum(proms) / len(proms), 2)
    return None

def _fetch_mag_promedio_novillo_vivo() -> tuple[float | None, str, str]:
    """
    Devuelve (prom_vivo, source, note). Intenta sitio oficial y si falla, recurre al fallback.
    """
    headers = {"User-Agent": HTTP_UA}
    last_err = None

    # 1) Intento sitio oficial
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
            r = client.get(MAG_URL_PRIMARY)
            r.raise_for_status()
            prom = _extract_promedio_novillos_from_primary(r.text)
            if prom:
                return prom, "MAG (sitio oficial)", f"URL: {MAG_URL_PRIMARY}"
    except Exception as e:
        last_err = e

    # 2) Fallback agregador
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, headers=headers, follow_redirects=True) as client:
            r = client.get(MAG_URL_FALLBACK)
            r.raise_for_status()
            prom = _extract_promedio_novillos_from_fallback(r.text)
            if prom:
                return prom, "MAG (via agregador)", f"URL: {MAG_URL_FALLBACK}"
    except Exception as e:
        last_err = e

    return None, "MAG (indisponible)", f"Error: {last_err!r}"

@router.get(
    "/mag_to_canal",
    response_model=MagToCanalResponse,
    dependencies=[Depends(require_api_key)]
)
async def get_mag_to_canal(date: Date, rend: float = 0.56):
    """
    Lee NOVILLOS ($/kg vivo) del MAG (oficial o fallback) y convierte a $/kg canal usando 'rend'.
    Devuelve un rango ±5% como referencia.
    """
    prom_vivo, source, note = _fetch_mag_promedio_novillo_vivo()
    if not prom_vivo:
        # No rompemos: devolvemos un rango vacío informativo
        raise HTTPException(status_code=502, detail=f"No se pudo leer MAG. {note}")

    canal_avg = prom_vivo / rend
    rango = CanalRange(
        min=round(canal_avg * 0.95, 2),
        avg=round(canal_avg, 2),
        max=round(canal_avg * 1.05, 2),
    )

    return MagToCanalResponse(
        ok=True,
        source=source,
        date=str(date),
        rendimiento=rend,
        unit="ARS/kg canal (estimado)",
        canal_range=rango,
        note=note + " | Base: NOVILLOS ($/kg vivo). Conversión a canal por rendimiento."
    )
