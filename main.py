from datetime import datetime, timezone
from fastapi import FastAPI
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import re
import logging
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger("uvicorn.error")




app = FastAPI(title="Beef Data Hub", version="0.1.0")

def to_number_ar(s: str) -> float | None:
    """
    Convierte textos en formato argentino (p.ej. '1.800,00') a float (1800.00).
    - Acepta '1.800,00', '1800,00', '1.800', '1800', etc.
    - Ignora símbolos ($, espacios, NBSP) y letras.
    - Devuelve None si no hay dígitos.
    """
    if s is None:
        return None
    # normalizar: quitar espacios, NBSP y símbolos comunes
    t = str(s).replace('\xa0', ' ').strip()
    t = re.sub(r'[^0-9.,-]', '', t)          # deja solo dígitos, coma, punto, signo
    if not re.search(r'\d', t):
        return None
    # si hay coma como separador decimal (formato AR), quitar puntos de miles y cambiar coma por punto
    # casos: "1.800,00" -> "1800.00"; "1800,5" -> "1800.5"
    if ',' in t:
        t = t.replace('.', '').replace(',', '.')
    # si no hay coma, podía venir "2183.30" ya en formato punto decimal; o "2183" entero
    try:
        return float(t)
    except ValueError:
        return None




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # en MVP permitimos todos los orígenes
    allow_credentials=False,
    allow_methods=["GET"],        # solo GET por ahora
    allow_headers=["*"],
)


# --- API KEY simple (MVP) ---
import os
from starlette.responses import JSONResponse

API_KEY = os.environ.get("BEEF_HUB_API_KEY", "dev-123")
# Rendimiento canal (kg canal / kg vivo). Usado para convertir vivo→canal.
REN_DIM = float(os.environ.get("BEEF_RENDIMIENTO", "0.56"))
_PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def _api_key_guard(request, call_next):
    # Log siempre (diagnóstico)
    path = request.url.path
    key_present = "x-api-key" in request.headers
    ua = request.headers.get("user-agent", "")[:80]
    logger.info(f"[AUTH] path={path} key_present={key_present} ua={ua}")

    if path in _PUBLIC_PATHS:
        return await call_next(request)

    key = request.headers.get("x-api-key")
    if key != API_KEY:
        return JSONResponse({"error": "unauthorized", "detail": "missing or invalid x-api-key"}, status_code=401)

    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"[VALIDATION] path={request.url.path} errors={exc.errors()[:2]}")
    return JSONResponse(
        {
            "ok": False,
            "error": "validation_error",
            "path": str(request.url.path),
            "details": exc.errors(),
        },
        status_code=200,  # <-- 200 para que Actions no lo descarte
    )

@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    logger.exception(f"[UNHANDLED] path={request.url.path} {exc.__class__.__name__}: {exc}")
    return JSONResponse(
        {
            "ok": False,
            "error": "server_error",
            "path": str(request.url.path),
            "detail": str(exc)[:200],  # limitar tamaño
        },
        status_code=200,  # <-- 200 para evitar 5xx en el cliente
    )


#-----------#

@app.get("/")
def root():
    return {
        "name": "Beef Data Hub",
        "version": "0.1.0",
        "message": "OK. Visita /health para chequeo de liveness."
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "beef-data-hub",
        "time_utc": datetime.now(timezone.utc).isoformat()
    }



import httpx

@app.get("/siocarnes/raw")
def siocarnes_raw(
    dia: str = "15/8/2025", 
    categoria: int = 14, 
    subcategoria: str = "null", 
    zona: int = -1, 
    raza: int = -1
):
    """
    Llama directamente al endpoint oficial de SIO Carnes y devuelve la respuesta cruda.
    Parámetros por defecto: Novillo, día de ejemplo.
    """
    url = "https://siocarnes.magyp.gob.ar/api/Reportes/GetDatosMonitor"
    params = {
        "dia": dia,
        "categoria": str(categoria),
        "subcatergoria": subcategoria,  # Ojo: el endpoint usa 'subcatergoria' con error de ortografía
        "zona": str(zona),
        "raza": str(raza)
    }

    try:
        resp = httpx.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

    
@app.get("/siocarnes/raw/inspect")
def siocarnes_raw_inspect(
    dia: str = "15/8/2025", 
    categoria: int = 14, 
    subcategoria: str = "null", 
    zona: int = -1, 
    raza: int = -1
):
    """
    Llama al endpoint oficial y resume la estructura recibida:
    - cantidad de filas
    - tipo de dato de nivel superior
    - claves del primer registro (si aplica)
    """
    url = "https://siocarnes.magyp.gob.ar/api/Reportes/GetDatosMonitor"
    params = {
        "dia": dia,
        "categoria": str(categoria),
        "subcatergoria": subcategoria,  # el endpoint usa este nombre con typo
        "zona": str(zona),
        "raza": str(raza)
    }

    try:
        import httpx
        r = httpx.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        summary = {
            "top_level_type": type(data).__name__,
            "count": None,
            "first_item_keys": None
        }

        if isinstance(data, list):
            summary["count"] = len(data)
            if data and isinstance(data[0], dict):
                summary["first_item_keys"] = sorted(list(data[0].keys()))
        elif isinstance(data, dict):
            # Algunos endpoints devuelven { "data": [...] } o similar
            summary["first_item_keys"] = sorted(list(data.keys()))
            # Si tiene una lista principal en alguna clave común, intenta contar
            for k in ("data", "rows", "result", "items"):
                if k in data and isinstance(data[k], list):
                    summary["count"] = len(data[k])
                    break

        return {"ok": True, "summary": summary, "params_used": params}
    except Exception as e:
        return {"ok": False, "error": str(e), "params_used": params}

    
@app.get("/siocarnes/raw/sample")
def siocarnes_raw_sample(
    dia: str = "15/8/2025", 
    categoria: int = 14, 
    subcategoria: str = "null", 
    zona: int = -1, 
    raza: int = -1
):
    """
    Devuelve una muestra mínima para entender el esquema real:
    - claves del primer item de 'precios'
    - el primer item en sí (recortado)
    """
    import httpx
    url = "https://siocarnes.magyp.gob.ar/api/Reportes/GetDatosMonitor"
    params = {
        "dia": dia,
        "categoria": str(categoria),
        "subcatergoria": subcategoria,  # typo requerido por el endpoint oficial
        "zona": str(zona),
        "raza": str(raza),
    }

    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    sample = {"precios_first_keys": None, "precios_first_item": None}
    precios = data.get("precios")

    if isinstance(precios, list) and precios:
        first = precios[0]
        if isinstance(first, dict):
            sample["precios_first_keys"] = sorted(list(first.keys()))
            # recorte para no devolver objetos muy grandes
            sample["precios_first_item"] = {
                k: first[k] for k in sorted(first.keys())[:12]  # primeras 12 claves
            }

    return {"ok": True, "sample": sample}

@app.get("/siocarnes/novillo")
def siocarnes_novillo(
    dia: str = "15/8/2025", 
    categoria: int = 14, 
    subcategoria: str = "null", 
    zona: int = -1, 
    raza: int = -1
):
    """
    Endpoint normalizado: devuelve precios de Novillo en ARS/kg_canal como float.
    """
    import httpx
    url = "https://siocarnes.magyp.gob.ar/api/Reportes/GetDatosMonitor"
    params = {
        "dia": dia,
        "categoria": str(categoria),
        "subcatergoria": subcategoria,  # ojo: typo oficial
        "zona": str(zona),
        "raza": str(raza),
    }

    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    precios = data.get("precios", [])
    salida = []

    for item in precios:
        salida.append({
            "fecha": item.get("fecha"),
            "precio_frecuente": to_number_ar(item.get("precioFrecuente")),
            "precio_minimo": to_number_ar(item.get("precioMinimo")),
            "precio_maximo": to_number_ar(item.get("precioMaximo")),
            "precio_promedio": to_number_ar(item.get("precioPromedio")),
            "unidad": "ARS/kg_canal",
            "source": "SIO Carnes",
        })

    return {
        "series": "siocarnes_novillo_arskg_canal",
        "count": len(salida),
        "data": salida[:10]  # solo primeras 10 filas para no desbordar
    }

    
@app.get("/siocarnes/novillo_cached")
def siocarnes_novillo_cached(
    dia: str = "15/8/2025",
    categoria: int = 14,
    subcategoria: str = "null",
    zona: int = -1,
    raza: int = -1,
    ttl_seconds: int = 6 * 60 * 60,  # 6 horas
):
    """
    Igual que /siocarnes/novillo pero con caché en memoria (TTL por defecto: 6 h).
    Evita pegarle a SIO en cada request.
    """
    import time, json, httpx

    # caché simple ligado a la función
    if not hasattr(siocarnes_novillo_cached, "_cache"):
        siocarnes_novillo_cached._cache = {}  # type: ignore[attr-defined]
    cache = siocarnes_novillo_cached._cache  # type: ignore[attr-defined]

    # clave única por parámetros
    key = json.dumps(
        {
            "dia": dia,
            "categoria": categoria,
            "subcategoria": subcategoria,
            "zona": zona,
            "raza": raza,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    now = time.time()
    hit = cache.get(key)
    if hit and (now - hit["ts"] < ttl_seconds):
        return hit["value"]

    # si no hay caché válido, llamamos a SIO y normalizamos (mismo formato del endpoint normalizado)
    url = "https://siocarnes.magyp.gob.ar/api/Reportes/GetDatosMonitor"
    params = {
        "dia": dia,
        "categoria": str(categoria),
        "subcatergoria": subcategoria,  # typo oficial
        "zona": str(zona),
        "raza": str(raza),
    }

    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    precios = data.get("precios", [])
    salida = []
    for item in precios:
        salida.append({
            "fecha": item.get("fecha"),
            "precio_frecuente": to_number_ar(item.get("precioFrecuente")),
            "precio_minimo": to_number_ar(item.get("precioMinimo")),
            "precio_maximo": to_number_ar(item.get("precioMaximo")),
            "precio_promedio": to_number_ar(item.get("precioPromedio")),
            "unidad": "ARS/kg_canal",
            "source": "SIO Carnes",
        })

    result = {
        "series": "siocarnes_novillo_arskg_canal",
        "count": len(salida),
        "data": salida[:10]
    }

    cache[key] = {"ts": now, "value": result}
    return result

from datetime import datetime

def iso_to_sio(d: str) -> str:
    """
    Convierte 'YYYY-MM-DD' -> 'D/M/YYYY' (formato que usa SIO).
    Ej.: '2025-08-15' -> '15/8/2025'
    """
    dt = datetime.strptime(d, "%Y-%m-%d")
    return f"{dt.day}/{dt.month}/{dt.year}"

def convert_price_per_kg(p_vivo, unit: str):
    """
    Convierte un precio por kg EN PIE (vivo) a la unidad solicitada.
    - unit == "vivo": retorna p_vivo (ARS/kg_vivo)
    - unit == "canal": retorna p_vivo / REN_DIM (ARS/kg_canal)
    Retorna: (precio_convertido | None, unidad_str, detalle_conv)
    """
    if p_vivo is None:
        # sin dato, preservamos None
        return None, "ARS/kg_vivo", "none"
    if unit == "canal":
        try:
            return float(p_vivo) / REN_DIM, "ARS/kg_canal", f"divide_by_r={REN_DIM}"
        except Exception:
            return None, "ARS/kg_canal", "error"
    # por defecto, vivo
    try:
        return float(p_vivo), "ARS/kg_vivo", "identity"
    except Exception:
        return None, "ARS/kg_vivo", "error"


@app.get("/siocarnes/novillo_by_date")
def siocarnes_novillo_by_date(date: str = "2025-08-15", request: Request = None):
    sio_day = iso_to_sio(date)
    res = siocarnes_novillo_cached(dia=sio_day)
    try:
        has_key = (request is not None) and ("x-api-key" in request.headers)
        ua = request.headers.get("user-agent", "")[:60] if request else ""
        logger.info(f"[DEBUG] /siocarnes/novillo_by_date date={date} count={res.get('count')} has_key={has_key} ua={ua}")
    except Exception:
        pass
    return res






@app.get("/sources")
def sources_status():
    """
    Resume el estado de las fuentes conocidas y cuándo se actualizaron por última vez.
    Por ahora: inspecciona el caché de /siocarnes/novillo_cached.
    """
    import time
    status = {
        "siocarnes_novillo": {
            "last_fetch_epoch": None,
            "last_fetch_iso": None,
            "entries_cached": 0,
            "notes": "Se nutre del caché interno de /siocarnes/novillo_cached"
        }
    }

    fn = siocarnes_novillo_cached
    if hasattr(fn, "_cache"):  # type: ignore[attr-defined]
        cache = fn._cache  # type: ignore[attr-defined]
        status["siocarnes_novillo"]["entries_cached"] = len(cache)
        if cache:
            last_ts = max(entry["ts"] for entry in cache.values())
            status["siocarnes_novillo"]["last_fetch_epoch"] = last_ts
            try:
                from datetime import datetime, timezone
                status["siocarnes_novillo"]["last_fetch_iso"] = datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()
            except Exception:
                pass

    status["service_time_epoch"] = time.time()
    return {"ok": True, "sources": status}



@app.get("/forecast/siocarnes/novillo_simple")
def forecast_novillo_simple(h: int = 30):
    """
    Proyección simple (MVP): ajusta una recta y proyecta h días usando los
    precios 'precio_promedio' disponibles desde /siocarnes/novillo_cached.
    - Sin dependencias externas.
    - Devuelve punto y bandas +/-10% (provisorias).
    """
    # 1) Traer datos crudos (usa el día por defecto del cached, que devuelve una ventana reciente)
    base = siocarnes_novillo_cached()
    rows = base.get("data", [])

    # 2) Armar series limpias: (t, y) con t=0..n-1 en orden cronológico
    # La API suele venir ordenada por fecha ascendente; si no, ordenamos.
    def parse_dmy(fecha_str: str):
        # "10/8/2025" -> (2025, 8, 10)
        d, m, y = fecha_str.split("/")
        return (int(y), int(m), int(d))

    rows = [r for r in rows if r.get("precio_promedio") is not None and r.get("fecha")]
    rows.sort(key=lambda r: parse_dmy(r["fecha"]))
    y = [r["precio_promedio"] for r in rows]
    n = len(y)

    if n < 3:
        return {
            "ok": False,
            "error": "Datos insuficientes para proyectar (se requieren al menos 3 observaciones).",
            "available_points": n
        }

    # 3) Ajuste de tendencia lineal simple y = a + b*t (mínimos cuadrados)
    # Fórmulas cerradas (sin numpy): b = cov(t,y)/var(t), a = mean(y) - b*mean(t)
    t = list(range(n))
    mean_t = sum(t) / n
    mean_y = sum(y) / n
    var_t = sum((ti - mean_t) ** 2 for ti in t) or 1e-9
    cov_ty = sum((ti - mean_t) * (yi - mean_y) for ti, yi in zip(t, y))
    b = cov_ty / var_t
    a = mean_y - b * mean_t

    # 4) Proyectar los próximos h puntos: t = n, n+1, ...
    point = []
    for k in range(1, h + 1):
        t_future = n - 1 + k
        y_hat = a + b * t_future
        point.append(float(y_hat))

    # 5) Bandas (provisorias): +/-10% del punto (MVP)
    low = [max(0.0, v * 0.90) for v in point]
    high = [v * 1.10 for v in point]

    return {
        "ok": True,
        "model": "trend_linear_simple",
        "history_count": n,
        "h": h,
        "unit": "ARS/kg_canal",
        "last_observation": rows[-1],
        "point_forecast": point,
        "low": low,
        "high": high,
        "notes": [
            "Proyección MVP con tendencia lineal y bandas +/-10%.",
            "Sustituir por modelos estacionales/ARIMA en etapas siguientes."
        ]
    }


from datetime import datetime, timedelta

def _iso_range(dfrom: str, dto: str):
    """Genera fechas ISO día a día (incluyendo extremos)."""
    start = datetime.strptime(dfrom, "%Y-%m-%d").date()
    end   = datetime.strptime(dto,   "%Y-%m-%d").date()
    if end < start:
        raise ValueError("to < from")
    cur = start
    while cur <= end:
        yield cur.isoformat()
        cur += timedelta(days=1)

@app.get("/siocarnes/novillo_range")
def siocarnes_novillo_range(from_date: str, to_date: str, limit: int = 200):
    """
    Trae Novillo SIO para un rango ISO [from_date, to_date], agregando resultados día a día.
    Reutiliza el endpoint cacheado por cada fecha para no sobrecargar SIO.
    """
    # 1) Iterar fechas y acumular
    all_rows = []
    seen = set()  # para deduplicar por (fecha, precio_promedio)
    for iso_day in _iso_range(from_date, to_date):
        # Reusar el cacheador con fecha ISO -> SIO (via wrapper)
        res = siocarnes_novillo_by_date(date=iso_day)
        data = res.get("data", []) if isinstance(res, dict) else []
        for r in data:
            key = (r.get("fecha"), r.get("precio_promedio"))
            if key not in seen:
                seen.add(key)
                all_rows.append(r)
                
    # ...
    all_rows = [
    r for r in all_rows
    if datetime.strptime(from_date, "%Y-%m-%d").date()
       <= datetime.strptime(r["fecha"], "%d/%m/%Y").date()
       <= datetime.strptime(to_date, "%Y-%m-%d").date()
    ]


    # 2) Ordenar por fecha D/M/YYYY asc
    def _key_dmy(row):
        d, m, y = row["fecha"].split("/")
        return (int(y), int(m), int(d))
    all_rows.sort(key=_key_dmy)

    # 3) Limitar salida si es muy grande
    out = all_rows[:max(1, limit)]

    return {
        "series": "siocarnes_novillo_arskg_canal",
        "from": from_date,
        "to": to_date,
        "count": len(out),
        "unit": "ARS/kg_canal",
        "data": out
    }

    
    
@app.get("/siocarnes/novillo_agg")
def siocarnes_novillo_agg(from_date: str, to_date: str, freq: str = "w"):
    """
    Agrega los datos diarios a frecuencia semanal ('w') o mensual ('m').
    - Usa /siocarnes/novillo_range para obtener los días del rango.
    - Calcula promedio simple de 'precio_promedio' por grupo.
    - Devuelve {periodo, count, precio_promedio}.
    """
    # 1) Traer diarios del rango solicitado (ya filtrados)
    base = siocarnes_novillo_range(from_date=from_date, to_date=to_date, limit=10_000)
    rows = base.get("data", []) if isinstance(base, dict) else []

    if freq not in ("w", "m"):
        return {"ok": False, "error": "freq debe ser 'w' (semanal) o 'm' (mensual)."}

    # 2) Agrupar por semana ISO o por mes calendario
    from collections import defaultdict
    import math

    buckets = defaultdict(list)

    for r in rows:
        # fecha viene como "D/M/YYYY"
        dt = datetime.strptime(r["fecha"], "%d/%m/%Y").date()
        if freq == "w":
            year, week, _ = dt.isocalendar()  # semana ISO
            key = f"{year}-W{week:02d}"
        else:  # "m"
            key = f"{dt.year}-{dt.month:02d}"

        y = r.get("precio_promedio")
        if isinstance(y, (int, float)):
            buckets[key].append(float(y))

    # 3) Armar salida: promedio por grupo
    out = []
    for key, vals in buckets.items():
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        out.append({
            "periodo": key,
            "count": len(vals),
            "precio_promedio": float(avg),
            "unidad": "ARS/kg_canal",
            "source": "SIO Carnes"
        })

    # 4) Orden cronológico del periodo
    def _key_period(k: str):
        if freq == "w":
            y, w = k.split("-W")
            return (int(y), int(w))
        else:
            y, m = k.split("-")
            return (int(y), int(m))

    out.sort(key=lambda x: _key_period(x["periodo"]))

    return {
        "ok": True,
        "series": f"siocarnes_novillo_arskg_canal_{'weekly' if freq=='w' else 'monthly'}",
        "from": from_date,
        "to": to_date,
        "freq": freq,
        "unit": "ARS/kg_canal",
        "groups": out
    }

    
@app.get("/siocarnes/novillo_by_date_compact")
def siocarnes_novillo_by_date_compact(
    date: str = "2025-08-15",
    n: int = 3,
    unit: str = "vivo"  # <-- nuevo: "vivo" (default) o "canal"
):
    """
    Versión ultra-compacta para integraciones (customGPT):
    - Responde solo conteo, promedio y primeras N filas.
    - Soporta unidad 'vivo' (en pie) o 'canal' (convierte usando REN_DIM).
    """
    # Reusar la lógica existente (incluye caché)
    base = siocarnes_novillo_by_date(date=date)
    rows = base.get("data", []) if isinstance(base, dict) else []

    # Sanitizar N
    try:
        n = max(0, min(int(n), 5))
    except Exception:
        n = 3

    out_rows = []
    vals = []
    unit_out = "ARS/kg_vivo"  # se actualiza en convert_price_per_kg

    for r in rows:
        # Convertimos TODOS los campos de precio a la unidad pedida
        pf_out, unit_out, _ = convert_price_per_kg(r.get("precio_frecuente"), unit)
        pmin_out, _, _       = convert_price_per_kg(r.get("precio_minimo"), unit)
        pmax_out, _, _       = convert_price_per_kg(r.get("precio_maximo"), unit)
        pavg_out, _, _       = convert_price_per_kg(r.get("precio_promedio"), unit)

        out_rows.append({
            "fecha": r.get("fecha"),
            "precio_frecuente": pf_out,
            "precio_minimo": pmin_out,
            "precio_maximo": pmax_out,
            "precio_promedio": pavg_out,
            "unidad": unit_out,
            "source": r.get("source", "SIO Carnes"),
        })

        if isinstance(pavg_out, (int, float)):
            vals.append(float(pavg_out))

    avg = (sum(vals) / len(vals)) if vals else None

    return {
        "ok": True,
        "date": date,
        "unit": unit_out,
        "count": len(out_rows),
        "avg_precio_promedio": avg,
        "first_rows": out_rows[:n],
        "note": "Fuente SIO: precios reportados en pie; conversión opcional a 'canal' vía BEEF_RENDIMIENTO."
    }


    
@app.get("/siocarnes/novillo_agg_compact")
def siocarnes_novillo_agg_compact(from_date: str, to_date: str, freq: str = "w", k: int = 3):
    """
    Agregación compacta para Actions:
    - Reusa /siocarnes/novillo_agg (semanal o mensual).
    - Devuelve solo los últimos k grupos y un promedio simple de esos grupos.
    - Mantiene 'ok: true' y payload pequeño para evitar límites del Builder.
    """
    base = siocarnes_novillo_agg(from_date=from_date, to_date=to_date, freq=freq)
    groups = []
    if isinstance(base, dict):
        groups = base.get("groups", []) or []
    # Sanitizar k (máximo 6 para mantener chico el payload)
    try:
        k = max(1, min(int(k), 6))
    except Exception:
        k = 3

    tail = groups[-k:] if groups else []
    vals = [float(g.get("precio_promedio")) for g in tail if isinstance(g.get("precio_promedio"), (int, float))]
    avg = (sum(vals) / len(vals)) if vals else None

    return {
        "ok": True,
        "from": base.get("from") if isinstance(base, dict) else from_date,
        "to": base.get("to") if isinstance(base, dict) else to_date,
        "freq": freq,
        "unit": "ARS/kg_canal",
        "group_count": len(groups),
        "avg_precio_promedio_last_k": avg,
        "last_k_groups": tail,
        "note": "Compact aggregation for Actions; use /siocarnes/novillo_agg for full payload."
    }

    
@app.get("/siocarnes/novillo_range_compact")
def siocarnes_novillo_range_compact(from_date: str, to_date: str, n: int = 5):
    """
    Resumen compacto para un rango [from_date, to_date] (ISO YYYY-MM-DD).
    - Devuelve ok, unidad, cantidad de días, stats (min/avg/max del precio_promedio)
      y hasta n filas del principio y del final para referencia.
    - Mantiene payload chico para Actions.
    """
    base = siocarnes_novillo_range(from_date=from_date, to_date=to_date, limit=10000)
    rows = []
    if isinstance(base, dict):
        rows = base.get("data", []) or []

    # Limitar n para payload pequeño
    try:
        n = max(1, min(int(n), 5))
    except Exception:
        n = 5

    # Extraer valores numéricos del campo precio_promedio
    vals = []
    for r in rows:
        v = r.get("precio_promedio")
        if isinstance(v, (int, float)):
            vals.append(float(v))

    stats = {
        "count_days": len(rows),
        "min": min(vals) if vals else None,
        "avg": (sum(vals) / len(vals)) if vals else None,
        "max": max(vals) if vals else None,
    }

    head = rows[:n] if rows else []
    tail = rows[-n:] if rows else []

    return {
        "ok": True,
        "from": from_date,
        "to": to_date,
        "unit": "ARS/kg_canal",
        "stats": stats,
        "head": head,
        "tail": tail,
        "note": "Compact range summary; use /siocarnes/novillo_range for full series."
    }
