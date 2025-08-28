import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from checks_mag import router as checks_router

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

API_KEY = os.environ.get("BEEF_HUB_API_KEY", "dev-123")
# Rendimiento res/dim por defecto para pasar de "vivo" -> "canal" y viceversa
# (fuente típica 0.56; ajustable por env)
REN_DIM = float(os.environ.get("BEEF_RENDIMIENTO", "0.56"))

SIO_ENDPOINT = "https://siocarnes.magyp.gob.ar/api/Reportes/GetDatosMonitor"

# Timeouts/network
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "15"))  # segundos
HTTP_RETRIES = int(os.environ.get("HTTP_RETRIES", "2"))
HTTP_UA = os.environ.get(
    "HTTP_UA",
    "Mozilla/5.0 (compatible; BeefDataHub/1.0; +https://beef-data-hub)"
)

# Ventana por defecto para series cortas
DEFAULT_LOOKBACK = int(os.environ.get("DEFAULT_LOOKBACK", "6"))  # días

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

app = FastAPI(title="Beef Data Hub", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(checks_router)



# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def iso_to_sio(date_iso: str) -> str:
    """
    Convierte 'YYYY-MM-DD' a 'D/M/YYYY' (formato que usa el endpoint del SIO).
    """
    dt = datetime.fromisoformat(date_iso)
    return f"{dt.day}/{dt.month}/{dt.year}"

def num_ar_to_float(value: Any) -> Optional[float]:
    """
    Convierte valores estilo '1.800,00' a float 1800.00. Acepta números y retorna None si vacío.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    # quitar puntos de miles y cambiar coma decimal a punto
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def build_sio_params(
    dia: str,
    categoria: int | str = 14,
    subcategoria: Optional[str] = None,
    zona: Optional[int | str] = None,
    raza: Optional[int | str] = None,
) -> Dict[str, str]:
    """
    Replica el estilo de la UI del SIO:
    - 'dia' y 'categoria' siempre presentes
    - 'subcatergoria' (sic) vacío si no hay valor (NO 'null' literal)
    - 'zona' y 'raza' omitidos si representan 'todas'
    """
    params: Dict[str, str] = {"dia": dia, "categoria": str(categoria)}

    # Clave mal escrita en el backend del SIO: subcatergoria
    if not subcategoria or subcategoria.lower() == "null":
        params["subcatergoria"] = ""
    else:
        params["subcatergoria"] = str(subcategoria)

    def set_optional(name: str, v: Optional[int | str]) -> None:
        if v is None:
            return
        if isinstance(v, str) and v.strip() == "":
            # si se manda explicitamente vacío, mantenerlo
            params[name] = ""
            return
        # tratar -1 / "−1" / "-1" como 'todas' => omitir
        if str(v).strip() in {"-1", "−1"}:
            return
        params[name] = str(v)

    set_optional("zona", zona)
    set_optional("raza", raza)
    return params

def convert_from_canal(v: Optional[float], target_unit: str) -> Tuple[Optional[float], str, str]:
    """
    Convierte un precio expresado en kg canal a la unidad pedida.
    """
    if v is None:
        return None, "ARS/kg_canal", target_unit
    if target_unit == "canal":
        return float(v), "ARS/kg_canal", "ARS/kg_canal"
    if target_unit == "vivo":
        return float(v) * REN_DIM, "ARS/kg_canal", "ARS/kg_vivo"
    raise ValueError("target_unit inválida: use 'vivo' o 'canal'")

def convert_from_vivo(v: Optional[float], target_unit: str) -> Tuple[Optional[float], str, str]:
    """
    Convierte un precio expresado en kg vivo a la unidad pedida.
    """
    if v is None:
        return None, "ARS/kg_vivo", target_unit
    if target_unit == "vivo":
        return float(v), "ARS/kg_vivo", "ARS/kg_vivo"
    if target_unit == "canal":
        # canal ~= vivo / REN_DIM
        return float(v) / REN_DIM, "ARS/kg_vivo", "ARS/kg_canal"
    raise ValueError("target_unit inválida: use 'vivo' o 'canal'")

def http_get_sio(params: Dict[str, str]) -> Dict[str, Any]:
    """
    Llama al endpoint del SIO con httpx; agrega User-Agent y retries simples.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, HTTP_RETRIES + 2):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": HTTP_UA}) as client:
                r = client.get(SIO_ENDPOINT, params=params)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(0.2 * attempt)
    raise HTTPException(status_code=502, detail=f"SIO fetch failed: {last_exc}")

def normalize_sio_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Del objeto del SIO (cada fila de 'precios') extrae las 5 métricas básicas y arma nuestro esquema.
    Campos esperados (con título del SIO):
      - 'fecha' (p.ej '10/8/2025')
      - 'precioFrecuente', 'precioMaximo', 'precioMinimo', 'precioPromedio'
    """
    fecha = row.get("fecha") or row.get("Fecha") or row.get("FECHA")
    return {
        "fecha": str(fecha) if fecha else None,
        "precio_frecuente": num_ar_to_float(row.get("precioFrecuente")),
        "precio_minimo":    num_ar_to_float(row.get("precioMinimo")),
        "precio_maximo":    num_ar_to_float(row.get("precioMaximo")),
        "precio_promedio":  num_ar_to_float(row.get("precioPromedio")),
        "unidad": "ARS/kg_vivo",  # OJO: el SIO reporta *en pie* (vivo) en la UI del Monitor
        "source": "SIO Carnes",
    }

def sio_fetch_and_normalize(
    dia: str,
    categoria: int | str = 14,
    subcategoria: Optional[str] = None,
    zona: Optional[int | str] = None,
    raza: Optional[int | str] = None,
) -> List[Dict[str, Any]]:
    """
    Trae el JSON del SIO para 'dia' y devuelve una lista de filas normalizadas (vivo).
    """
    params = build_sio_params(dia=dia, categoria=categoria, subcategoria=subcategoria, zona=zona, raza=raza)
    raw = http_get_sio(params)
    precios = raw.get("precios") or raw.get("Precios") or []
    out: List[Dict[str, Any]] = []
    for item in precios:
        norm = normalize_sio_row(item)
        out.append(norm)
    return out

def pick_daily_aggregate(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Dado el array de 'precios' de un día, tomar la *última* entrada (la UI del SIO suele mostrar un consolidado por día).
    Si no hubiese orden claro, tomamos la que tenga 'precio_promedio' no nulo.
    """
    if not rows:
        return None
    # preferimos la ultima
    candidate = rows[-1]
    if candidate.get("precio_promedio") is not None:
        return candidate
    # buscar alguna con promedio presente
    for r in reversed(rows):
        if r.get("precio_promedio") is not None:
            return r
    return rows[-1]

# ----------------------------------------------------------------------------
# Auth simple por header x-api-key (excepto /health)
# ----------------------------------------------------------------------------
ALLOWED_EXACT = {
    "/",
    "/health",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/docs/oauth2-redirect",
    "/favicon.ico",
}
ALLOWED_PREFIXES = (
    "/docs",    # assets de Swagger UI
    "/static",  # si servís estáticos
    "/assets",  # si tu UI los usa
)



@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    path = request.url.path

    # Rutas públicas (no requieren x-api-key)
    if path in ALLOWED_EXACT or any(path.startswith(p) for p in ALLOWED_PREFIXES):
        return await call_next(request)

    # Resto: exigir x-api-key válida
    api_key = request.headers.get("x-api-key")
    expected = API_KEY
    if not expected or api_key != expected:
        return JSONResponse(status_code=401, content={"detail": "missing or invalid x-api-key"})

    return await call_next(request)


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "beef-data-hub", "time_utc": datetime.utcnow().isoformat()}

@app.get("/siocarnes/raw")
def siocarnes_raw(
    dia: str = "18/8/2025",
    categoria: int = 14,
    subcategoria: Optional[str] = None,
    zona: Optional[int] = None,
    raza: Optional[int] = None,
):
    """
    Proxy en vivo al endpoint del SIO (Monitor). Devuelve el JSON original del SIO,
    solo para diagnóstico/validación.
    """
    params = build_sio_params(dia=dia, categoria=categoria, subcategoria=subcategoria, zona=zona, raza=raza)
    data = http_get_sio(params)
    # Para inspección rápida de estructura
    sample = {"precios_first_keys": None, "precios_first_item": None}
    try:
        if isinstance(data.get("precios"), list) and data["precios"]:
            first = data["precios"][0]
            sample["precios_first_keys"] = sorted(list(first.keys()))
            sample["precios_first_item"] = {
                "fecha": first.get("fecha"), "precioFrecuente": first.get("precioFrecuente"),
                "precioMaximo": first.get("precioMaximo"), "precioMinimo": first.get("precioMinimo"),
                "precioPromedio": first.get("precioPromedio")
            }
    except Exception:
        pass
    return {"ok": True, "params_used": params, "sample": sample, "raw": data}

@app.get("/siocarnes/novillo_by_date")
def siocarnes_novillo_by_date(date: str = "2025-08-18", n: int = DEFAULT_LOOKBACK):
    """
    Serie diaria *normalizada* a partir del SIO (base: 'vivo'), para los últimos n días hasta 'date' (ISO).
    Devuelve en *canal* como unidad base (para compatibilidad con tu pipeline histórico).
    """
    try:
        n = max(1, min(int(n), 10))
    except Exception:
        n = DEFAULT_LOOKBACK

    end = datetime.fromisoformat(date)
    days: List[Dict[str, Any]] = []
    for k in range(n):
        dt = end - timedelta(days=(n - 1 - k))
        sio_day = f"{dt.day}/{dt.month}/{dt.year}"
        rows_vivo = sio_fetch_and_normalize(dia=sio_day, categoria=14, subcategoria=None, zona=None, raza=None)
        agg = pick_daily_aggregate(rows_vivo)
        if not agg:
            continue
        # Convertimos a *canal* (tu base analítica)
        pf, _, _ = convert_from_vivo(agg.get("precio_frecuente"), "canal")
        pmin, _, _ = convert_from_vivo(agg.get("precio_minimo"), "canal")
        pmax, _, _ = convert_from_vivo(agg.get("precio_maximo"), "canal")
        pavg, _, _ = convert_from_vivo(agg.get("precio_promedio"), "canal")
        days.append({
            "fecha": agg.get("fecha"),
            "precio_frecuente": pf,
            "precio_minimo": pmin,
            "precio_maximo": pmax,
            "precio_promedio": pavg,
            "unidad": "ARS/kg_canal",
            "source": "SIO Carnes"
        })

    return {"series": "siocarnes_novillo_arskg_canal", "count": len(days), "data": days}

@app.get("/siocarnes/novillo_by_date_compact_v2")
def siocarnes_novillo_by_date_compact_v2(
    date: str = "2025-08-18",
    n: int = 3,
    unit: str = "vivo",
):
    """
    Compacto para Actions: promedio de los últimos n días respecto a 'date'.
    'unit' puede ser 'vivo' (UI del SIO) o 'canal' (convertido).
    """
    base = siocarnes_novillo_by_date(date=date, n=max(1, n))
    rows = base.get("data", []) if isinstance(base, dict) else []
    # rows están en *canal* (por diseño de siocarnes_novillo_by_date)
    # Si pidieron 'vivo', convertimos; si pidieron 'canal', dejamos.
    converted: List[Dict[str, Any]] = []
    for r in rows:
        if unit == "vivo":
            pf, _, _ = convert_from_canal(r.get("precio_frecuente"), "vivo")
            pmin, _, _ = convert_from_canal(r.get("precio_minimo"), "vivo")
            pmax, _, _ = convert_from_canal(r.get("precio_maximo"), "vivo")
            pavg, _, _ = convert_from_canal(r.get("precio_promedio"), "vivo")
            converted.append({**r, "precio_frecuente": pf, "precio_minimo": pmin, "precio_maximo": pmax, "precio_promedio": pavg, "unidad": "ARS/kg_vivo"})
        else:
            converted.append({**r})

    # promedio simple del precio_promedio
    vals = [x.get("precio_promedio") for x in converted if isinstance(x.get("precio_promedio"), (int, float))]
    avg = sum(vals) / len(vals) if vals else None
    unit_label = converted[0].get("unidad") if converted else ("ARS/kg_vivo" if unit == "vivo" else "ARS/kg_canal")

    # recorte "first_rows" para payload chico
    return {
        "ok": True,
        "date": date,
        "unit": unit_label,
        "count": len(converted),
        "avg_precio_promedio": avg,
        "first_rows": converted[:3],
        "note": "Fuente SIO (vivo) + conversión opcional a 'canal' mediante REN_DIM"
    }

# (Opcional) Debug: comparación lado a lado para una fecha
@app.get("/debug/siocarnes/by_date_compare")
def debug_siocarnes_by_date_compare(date: str = "2025-08-18", n: int = 3):
    base = siocarnes_novillo_by_date(date=date, n=n)
    rows = base.get("data", [])
    canal_view = rows[:n]
    vivo_view = []
    for r in canal_view:
        pf, _, _ = convert_from_canal(r.get("precio_frecuente"), "vivo")
        pmin, _, _ = convert_from_canal(r.get("precio_minimo"), "vivo")
        pmax, _, _ = convert_from_canal(r.get("precio_maximo"), "vivo")
        pavg, _, _ = convert_from_canal(r.get("precio_promedio"), "vivo")
        vivo_view.append({**r, "precio_frecuente": pf, "precio_minimo": pmin, "precio_maximo": pmax, "precio_promedio": pavg, "unidad": "ARS/kg_vivo"})
    return {
        "ok": True,
        "date": date,
        "ren_dim": REN_DIM,
        "base_unit_assumption": "ARS/kg_canal",
        "count_in_window": len(rows),
        "first_rows_canal": canal_view,
        "first_rows_vivo": vivo_view,
        "note": "Comparación directa base canal vs convertido a vivo"
    }

    
    
    
# ---------------------------------------------------------------------------
# UI Match: Promedio semanal tal cual la UI del SIO (Monitor)
# ---------------------------------------------------------------------------
def _parse_any_date_to_iso(s: str) -> str:
    """
    Acepta 'YYYY-MM-DD' o 'D/M/YYYY' y retorna ISO 'YYYY-MM-DD'.
    """
    s = s.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # D/M/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d).date().isoformat()
    # fallback: intenta fromisoformat (puede lanzar)
    return datetime.fromisoformat(s).date().isoformat()

def _within_week(iso_day: str, iso_from: str, iso_to: str) -> bool:
    d = datetime.fromisoformat(iso_day).date()
    f = datetime.fromisoformat(iso_from).date()
    t = datetime.fromisoformat(iso_to).date()
    return f <= d <= t

@app.get("/siocarnes/monitor_week")
def siocarnes_monitor_week(
    date: str = "2025-08-23",
    categoria: int = 14,
    unit: str = "vivo"  # 'vivo' (igual a UI) o 'canal' (convertido)
):
    """
    Devuelve el 'promedio semanal' exactamente como lo expone la UI del SIO,
    a partir del mismo feed (GetDatosMonitor). Si unit='canal', aplica conversión por REN_DIM.
    """
    # 1) Normalizar la fecha a ISO (si llega 'D/M/YYYY' la convertimos)
    date_iso = _parse_any_date_to_iso(date)
    dia_sio = iso_to_sio(date_iso)  # 'D/M/YYYY' para el endpoint del SIO

    # 2) Llamada 'en vivo' con parámetros estilo UI (subcatergoria vacía, zona/raza omitidos)
    params = build_sio_params(dia=dia_sio, categoria=categoria, subcategoria=None, zona=None, raza=None)
    raw = http_get_sio(params)

    # 3) Buscar la semana que contenga 'date_iso'
    semanas = raw.get("promedioSemanal") or []
    chosen = None
    for w in semanas:
        try:
            f_iso = w.get("fechaDesdeDT")[:10]
            t_iso = w.get("fechaHastaDT")[:10]
            if _within_week(date_iso, f_iso, t_iso):
                chosen = {
                    "from": f_iso,
                    "to": t_iso,
                    "descripcion": w.get("descripcion"),
                    # precio viene en 'vivo'
                    "precio_vivo": float(w.get("precio")) if w.get("precio") is not None else None,
                    "precio_vivo_str": w.get("precioString"),
                }
                break
        except Exception:
            continue

    if not chosen:
        # Si no hay match exacto, tomar la última semana disponible (UI suele mostrar la más reciente)
        if semanas:
            w = semanas[0]
            try:
                chosen = {
                    "from": w.get("fechaDesdeDT")[:10],
                    "to": w.get("fechaHastaDT")[:10],
                    "descripcion": w.get("descripcion"),
                    "precio_vivo": float(w.get("precio")) if w.get("precio") is not None else None,
                    "precio_vivo_str": w.get("precioString"),
                }
            except Exception:
                chosen = None

    if not chosen or chosen["precio_vivo"] is None:
        raise HTTPException(status_code=404, detail="No hay promedio semanal disponible para esa fecha.")

    # 4) Convertir a 'canal' si corresponde
    if unit == "canal":
        precio_conv, _, unit_label = convert_from_vivo(chosen["precio_vivo"], "canal")
    else:
        precio_conv, _, unit_label = convert_from_vivo(chosen["precio_vivo"], "vivo")

    # 5) Resumen compacto con eco de params
    return {
        "ok": True,
        "ui_equivalent": "promedioSemanal",
        "date_requested": date_iso,
        "week": {
            "from": chosen["from"],
            "to": chosen["to"],
            "descripcion": chosen["descripcion"]
        },
        "unit": unit_label,
        "precio_promedio": precio_conv,
        "precio_vivo_base": chosen["precio_vivo"],
        "precio_vivo_str": chosen["precio_vivo_str"],
        "params_used": params,
        "note": "Este endpoint replica el 'promedio semanal' del Monitor SIO; la UI suele mostrar esto como referencia."
    }
