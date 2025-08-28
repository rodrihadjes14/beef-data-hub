# checks_mag.py
from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from datetime import date as Date
import os

router = APIRouter(prefix="/checks", tags=["checks"])

# --- Seguridad simple por x-api-key ---
def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("BEEF_HUB_API_KEY", "")
    if not expected or x_api_key != expected:
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

@router.get(
    "/mag_to_canal",
    response_model=MagToCanalResponse,
    dependencies=[Depends(require_api_key)]
)
async def get_mag_to_canal(date: Date, rend: float = 0.56):
    """
    Convierte un precio promedio de hacienda en pie (MAG) a canal usando 'rend'.
    Por ahora usa un placeholder para el precio en pie; luego se reemplaza por lectura real.
    """
    # (1) Placeholder: precio promedio en pie (ARS/kg vivo). Reemplazar por lectura real de MAG.
    avg_en_pie_arskg = 1740.0

    # (2) Conversión a canal
    canal_avg = avg_en_pie_arskg / rend

    # (3) Rango simple ±5% como referencia
    rango = CanalRange(
        min=round(canal_avg * 0.95, 2),
        avg=round(canal_avg, 2),
        max=round(canal_avg * 1.05, 2),
    )

    return MagToCanalResponse(
        ok=True,
        source="MAG Cañuelas",
        date=str(date),
        rendimiento=rend,
        unit="ARS/kg canal (estimado)",
        canal_range=rango,
        note="Lectura MAG placeholder; sustituir por fuente real en el próximo paso."
    )
