# Sistema de Features del Modelo

Este documento detalla todas las 152+ features que el modelo utiliza para tomar decisiones de trading.

---

## Resumen de Features

| Categoria | Cantidad | Descripcion |
|-----------|----------|-------------|
| Timeframe Features | 96 (16 x 6) | Features por cada temporalidad |
| Cross-Timeframe | 12 | Relaciones entre temporalidades |
| Macro | 24 | Datos macroeconomicos |
| Calendar | 8 | Eventos economicos |
| Microstructure | 12 | Sesiones y liquidez |
| **TOTAL** | **152** | |

---

## 1. TIMEFRAME FEATURES (96 features)

**Archivo:** `features/timeframe_features.py`

Se calculan **16 features identicas** para cada una de las **6 temporalidades**:
- M5 (5 minutos)
- M15 (15 minutos)
- H1 (1 hora)
- H4 (4 horas)
- D1 (diario)
- W1 (semanal) - opcional

### 1.1 Price Action Features (5 por temporalidad)

| # | Feature | Formula | Que Busca |
|---|---------|---------|-----------|
| 1 | `{TF}_return` | `close.pct_change()` | Retorno del ultimo periodo |
| 2 | `{TF}_volatility` | `returns.rolling(20).std()` | Volatilidad reciente (20 periodos) |
| 3 | `{TF}_momentum_5` | `close.pct_change(5)` | Momentum corto plazo |
| 4 | `{TF}_momentum_10` | `close.pct_change(10)` | Momentum medio plazo |
| 5 | `{TF}_momentum_20` | `close.pct_change(20)` | Momentum largo plazo |

**Ejemplo con M5:**
```
M5_return = -0.0012      # Bajo 0.12% en ultimos 5 min
M5_volatility = 0.0008   # Volatilidad del 0.08%
M5_momentum_5 = -0.003   # Bajo 0.3% en ultimos 25 min
M5_momentum_10 = 0.001   # Subio 0.1% en ultimos 50 min
M5_momentum_20 = 0.005   # Subio 0.5% en ultimos 100 min
```

### 1.2 Trend Indicators (4 por temporalidad)

| # | Feature | Formula | Que Busca |
|---|---------|---------|-----------|
| 6 | `{TF}_ma_fast` | `(close - MA10) / close` | Distancia a media rapida |
| 7 | `{TF}_ma_slow` | `(close - MA50) / close` | Distancia a media lenta |
| 8 | `{TF}_ma_diff` | `(MA10 - MA50) / close` | Diferencia entre medias |
| 9 | `{TF}_trend` | `+1 si MA10 > MA50, -1 si no` | Direccion de tendencia |

**Interpretacion:**
- `ma_fast > 0`: Precio arriba de MA rapida (alcista corto plazo)
- `ma_slow > 0`: Precio arriba de MA lenta (alcista largo plazo)
- `ma_diff > 0`: MA rapida arriba de lenta (tendencia alcista)
- `trend = +1`: Confirmacion de tendencia alcista

### 1.3 Technical Indicators (4 por temporalidad)

| # | Feature | Formula | Rango | Que Busca |
|---|---------|---------|-------|-----------|
| 10 | `{TF}_rsi` | RSI(14) / 100 | 0-1 | Sobrecompra/sobreventa |
| 11 | `{TF}_macd` | (MACD - Signal) / close | -inf a +inf | Momentum normalizado |
| 12 | `{TF}_atr_pct` | ATR(14) / close | 0-inf | Volatilidad como % del precio |
| 13 | `{TF}_bb_position` | Posicion en Bollinger Bands | 0-1 | Posicion relativa |

**Interpretacion RSI:**
- `< 0.3` = Sobreventa (posible rebote)
- `> 0.7` = Sobrecompra (posible caida)
- `0.5` = Neutral

**Interpretacion BB Position:**
- `0` = En banda inferior
- `0.5` = En media
- `1` = En banda superior

### 1.4 Volume & Support/Resistance (3 por temporalidad)

| # | Feature | Formula | Que Busca |
|---|---------|---------|-----------|
| 14 | `{TF}_volume_ratio` | `volume / MA20(volume)` | Volumen relativo |
| 15 | `{TF}_dist_to_high` | `(close - High50) / close` | Distancia a resistencia |
| 16 | `{TF}_dist_to_low` | `(close - Low50) / close` | Distancia a soporte |

**Interpretacion Volume Ratio:**
- `> 2.0` = Volumen alto (movimiento significativo)
- `< 0.5` = Volumen bajo (consolidacion)

**Interpretacion Distancias:**
- `dist_to_high = 0` = En maximo de 50 periodos
- `dist_to_low = 0` = En minimo de 50 periodos

---

## 2. CROSS-TIMEFRAME FEATURES (12 features)

**Archivo:** `features/cross_timeframe.py`

Capturan relaciones entre temporalidades para detectar confluencias.

### 2.1 Trend Alignment (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `trend_alignment_all` | Promedio de trends de M5-D1 | Consenso de tendencia |
| `trend_strength_cascade` | D1_trend * H4_trend * H1_trend * M15_trend | Fuerza de tendencia |
| `trend_divergence` | std(trends de todos los TF) | Conflicto entre TF |

**Interpretacion:**
```
trend_alignment_all = +1.0  # Todos alcistas (senal fuerte de compra)
trend_alignment_all = -1.0  # Todos bajistas (senal fuerte de no comprar)
trend_alignment_all = 0.0   # Mixto (sin senal clara)

trend_strength_cascade = +1.0  # Tendencia fuerte confirmada
trend_strength_cascade = -1.0  # Contra-tendencia en algun TF

trend_divergence = 0.0   # Todos de acuerdo
trend_divergence = 1.0   # Maximo desacuerdo
```

### 2.2 Momentum Cascade (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `momentum_d1_h1` | D1_momentum_10 * H1_momentum_10 | Flujo D1 -> H1 |
| `momentum_h4_h1` | H4_momentum_10 * H1_momentum_10 | Flujo H4 -> H1 |
| `momentum_h1_m15` | H1_momentum_10 * M15_momentum_10 | Flujo H1 -> M15 |

**Interpretacion:**
```
momentum_d1_h1 > 0  # D1 y H1 van en la misma direccion
momentum_d1_h1 < 0  # Divergencia entre D1 y H1
```

### 2.3 Volatility Regime (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `volatility_regime` | vol_actual / vol_promedio_100 | Vol vs historico |
| `volatility_spike` | 1 si vol > 2x promedio reciente | Spike de volatilidad |
| `volatility_compression` | 1 si vol < 0.5x promedio | Compresion (pre-breakout) |

**Interpretacion:**
```
volatility_regime = 2.0   # Volatilidad 2x normal (mercado activo)
volatility_spike = 1.0    # Detectado spike de vol
volatility_compression = 1.0  # Volatilidad comprimida (breakout inminente)
```

### 2.4 Pattern Confluence (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `support_confluence` | % de TF cerca de soporte | Soporte multi-TF |
| `resistance_confluence` | % de TF cerca de resistencia | Resistencia multi-TF |
| `breakout_alignment` | +1/-1 si todos TF tienen mismo momentum | Breakout confirmado |

**Interpretacion:**
```
support_confluence = 0.8   # 80% de TF cerca de soporte (rebote probable)
resistance_confluence = 0.8  # 80% de TF cerca de resistencia (rechazo probable)
breakout_alignment = +1.0  # Breakout alcista confirmado en todos TF
```

---

## 3. MACRO FEATURES (24 features)

**Archivo:** `features/macro_features.py`

Datos de otros mercados que correlacionan con el oro.

### 3.1 DXY - Dollar Index (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `dxy_return` | DXY.pct_change() | Movimiento del dolar |
| `dxy_momentum` | DXY.pct_change(20) | Tendencia del dolar |
| `gold_dxy_correlation` | corr(Gold, DXY, 120 dias) | Correlacion inversa |

**Relacion con Oro:**
- Dolar sube -> Oro tiende a bajar (correlacion negativa)
- `gold_dxy_correlation` tipicamente entre -0.3 y -0.7

### 3.2 SPX - S&P 500 (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `spx_return` | SPX.pct_change() | Movimiento de acciones |
| `spx_momentum` | SPX.pct_change(20) | Tendencia de acciones |
| `gold_spx_correlation` | corr(Gold, SPX, 120 dias) | Risk-on/Risk-off |

**Relacion con Oro:**
- Risk-on (SPX sube): Inversores salen del oro
- Risk-off (SPX baja): Inversores compran oro como refugio

### 3.3 US10Y - Treasury Yields (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `us10y_change` | US10Y.diff() | Cambio en rendimientos |
| `us10y_momentum` | US10Y.diff(20) | Tendencia de rendimientos |
| `gold_yields_correlation` | corr(Gold, US10Y, 120 dias) | Relacion inversa |

**Relacion con Oro:**
- Yields suben -> Oro tiende a bajar (mayor costo de oportunidad)
- Yields bajan -> Oro tiende a subir

### 3.4 VIX - Fear Index (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `vix_level` | VIX / 50 | Nivel de miedo normalizado |
| `vix_change` | VIX.diff() | Cambio en volatilidad |
| `vix_regime` | +1 si VIX > 20, -1 si no | Regimen de miedo |

**Interpretacion:**
```
vix_level < 0.3 (VIX < 15)  # Mercado tranquilo
vix_level > 0.5 (VIX > 25)  # Mercado con miedo
vix_regime = +1.0           # Alto miedo (oro puede subir)
```

### 3.5 Oil - WTI Crude (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `oil_return` | Oil.pct_change() | Movimiento del petroleo |
| `oil_momentum` | Oil.pct_change(20) | Tendencia del petroleo |
| `gold_oil_correlation` | corr(Gold, Oil, 120 dias) | Correlacion commodities |

### 3.6 Bitcoin (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `btc_return` | BTC.pct_change() | Movimiento de Bitcoin |
| `btc_momentum` | BTC.pct_change(20) | Tendencia de Bitcoin |
| `gold_btc_correlation` | corr(Gold, BTC, 120 dias) | Risk sentiment |

### 3.7 EURUSD (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `eur_return` | EURUSD.pct_change() | Movimiento del Euro |
| `eur_momentum` | EURUSD.pct_change(20) | Tendencia del Euro |
| `gold_eur_correlation` | corr(Gold, EUR, 120 dias) | Proxy del dolar |

### 3.8 Silver & GLD (3 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `gold_silver_ratio` | (Gold/Silver) / 80 | Ratio normalizado |
| `gold_silver_correlation` | corr(Gold, Silver, 120 dias) | Correlacion metales |
| `gld_flow` | GLD.pct_change() | Flujos institucionales |

**Interpretacion Gold/Silver Ratio:**
```
gold_silver_ratio > 1.0  # Ratio alto (>80), oro caro vs plata
gold_silver_ratio < 1.0  # Ratio bajo (<80), plata cara vs oro
```

---

## 4. CALENDAR FEATURES (8 features)

**Archivo:** `features/calendar_features.py`

Informacion sobre eventos economicos importantes.

### 4.1 Event Timing (3 features)

| Feature | Formula | Rango | Que Busca |
|---------|---------|-------|-----------|
| `hours_to_event` | horas_hasta_evento / 168 | 0-1 | Proximidad a evento |
| `days_since_event` | dias_desde_evento / 30 | 0-1 | Tiempo desde ultimo |
| `event_density` | eventos_proximos_7d / 10 | 0-1 | Concentracion de eventos |

**Interpretacion:**
```
hours_to_event = 0.01  # Evento en ~1.5 horas (cuidado!)
hours_to_event = 1.0   # Evento en 1 semana o mas
event_density = 0.5    # 5 eventos en proximos 7 dias
```

### 4.2 Event Impact (3 features)

| Feature | Valor | Que Significa |
|---------|-------|---------------|
| `is_high_impact` | 0 o 1 | Proximo evento es de alto impacto |
| `in_event_window` | 0 o 1 | Dentro de +-2 horas de evento |
| `event_volatility_expected` | 1.0, 1.5, o 2.0 | Multiplicador de volatilidad |

**Interpretacion:**
```
is_high_impact = 1.0   # NFP, FOMC, CPI (volatilidad extrema)
in_event_window = 1.0  # PELIGRO: dentro de ventana de evento
event_volatility_expected = 2.0  # Esperar 2x volatilidad normal
```

### 4.3 Event Type (2 features)

| Feature | Valor | Que Significa |
|---------|-------|---------------|
| `event_type_nfp` | 0 o 1 | Proximo evento es NFP |
| `event_type_fomc` | 0 o 1 | Proximo evento es FOMC |

**NFP (Non-Farm Payrolls):**
- Primer viernes de cada mes
- 13:30 UTC
- Impacto: ~200 pips tipico

**FOMC (Federal Reserve):**
- 8 veces al ano
- 19:00 UTC
- Impacto: ~250 pips tipico

---

## 5. MICROSTRUCTURE FEATURES (12 features)

**Archivo:** `features/microstructure_features.py`

Patrones intradiarios y de liquidez.

### 5.1 Session Features (4 features)

| Feature | Horario UTC | Que Representa |
|---------|-------------|----------------|
| `session_asian` | 00:00 - 09:00 | Sesion de Asia |
| `session_london` | 08:00 - 17:00 | Sesion de Londres |
| `session_ny` | 13:00 - 22:00 | Sesion de Nueva York |
| `session_overlap` | 13:00 - 17:00 | Overlap Londres+NY |

**Importancia:**
- Overlap (13:00-17:00 UTC) = Mayor liquidez y volatilidad
- Asian = Menor volatilidad, rangos estrechos
- London open (08:00) = Inicio de volatilidad europea

### 5.2 Time Features (4 features)

| Feature | Rango | Que Representa |
|---------|-------|----------------|
| `hour_of_day` | 0-1 (0=00:00, 1=23:00) | Hora del dia |
| `day_of_week` | 0-1 (0=Lunes, 1=Domingo) | Dia de la semana |
| `week_of_month` | 0, 0.5, 1 | Semana del mes |
| `month_of_year` | 0-1 (0=Enero, 1=Diciembre) | Mes del ano |

**Patrones tipicos:**
- Lunes: Menor volatilidad
- Viernes: Cierre de posiciones semanales
- Primera semana: NFP
- Fin de mes: Rebalanceo institucional

### 5.3 Volume Features (2 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `volume_profile` | Percentil del volumen | Volumen relativo |
| `volume_imbalance` | (vol_compra - vol_venta) / vol_total | Presion direccional |

**Interpretacion Volume Imbalance:**
```
volume_imbalance > 0   # Mas presion compradora
volume_imbalance < 0   # Mas presion vendedora
```

### 5.4 Liquidity Features (2 features)

| Feature | Formula | Que Busca |
|---------|---------|-----------|
| `spread_m5` | (high - low) / close | Spread proxy |
| `liquidity_regime` | +1 si spread bajo, -1 si alto | Regimen de liquidez |

**Interpretacion:**
```
liquidity_regime = +1   # Alta liquidez (spreads bajos, ejecucion facil)
liquidity_regime = -1   # Baja liquidez (spreads altos, slippage)
```

---

## Flujo de Datos

```
                        DATOS CRUDOS
                             |
         +-------------------+-------------------+
         |                   |                   |
    +---------+        +-----------+       +-----------+
    | XAUUSD  |        |   Macro   |       | Calendar  |
    | M5-W1   |        | VIX, DXY  |       |   JSON    |
    +---------+        +-----------+       +-----------+
         |                   |                   |
         v                   v                   v
  +-------------+     +-------------+     +-------------+
  | Timeframe   |     |   Macro     |     |  Calendar   |
  | Features    |     |  Features   |     |  Features   |
  | (96)        |     |  (24)       |     |  (8)        |
  +-------------+     +-------------+     +-------------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                    +--------+--------+
                    |                 |
              +----------+      +------------+
              | Cross-TF |      | Microstruc |
              | (12)     |      | (12)       |
              +----------+      +------------+
                    |                 |
                    +--------+--------+
                             |
                             v
                    +----------------+
                    |  152 FEATURES  |
                    |   COMBINADAS   |
                    +----------------+
                             |
                             v
                    +----------------+
                    |   MODELO PPO   |
                    +----------------+
```

---

## Archivos de Datos Necesarios

| Archivo | Contenido | Obligatorio |
|---------|-----------|-------------|
| `data/xauusd_m5.csv` | OHLCV 5 minutos | SI |
| `data/xauusd_m15.csv` | OHLCV 15 minutos | SI |
| `data/xauusd_h1.csv` | OHLCV 1 hora | SI |
| `data/xauusd_h4.csv` | OHLCV 4 horas | SI |
| `data/xauusd_d1.csv` | OHLCV diario | SI |
| `data/xauusd_w1.csv` | OHLCV semanal | NO |
| `data/vix_daily.csv` | VIX | NO (rellena con 0) |
| `data/oil_wti_daily.csv` | Petroleo | NO |
| `data/bitcoin_daily.csv` | Bitcoin | NO |
| `data/eurusd_daily.csv` | EURUSD | NO |
| `data/silver_daily.csv` | Plata | NO |
| `data/gld_etf_daily.csv` | GLD ETF | NO |
| `data/economic_events_2015_2025.json` | Eventos | NO |

---

## Como Modificar Features

### Agregar una nueva feature a una temporalidad:

**Archivo:** `features/timeframe_features.py`

```python
def compute_timeframe_features(df, tf_name):
    # ... features existentes ...

    # Agregar nueva feature (ejemplo: Williams %R)
    high_14 = df['high'].rolling(14).max()
    low_14 = df['low'].rolling(14).min()
    result[f'{tf_name}_williams_r'] = (high_14 - df['close']) / (high_14 - low_14)

    return result
```

### Agregar una nueva fuente macro:

**Archivo:** `features/macro_features.py`

1. Agregar archivo a `macro_files`:
```python
macro_files = {
    # ... existentes ...
    'vvix': 'vvix_daily.csv',  # VIX del VIX
}
```

2. Crear funcion de features:
```python
def compute_vvix_features(gold_prices, vvix_prices):
    result = pd.DataFrame(index=gold_prices.index)
    vvix_aligned = normalize_timezone(vvix_prices, gold_prices.index)
    result['vvix_level'] = vvix_aligned / 100.0
    result['vvix_change'] = vvix_aligned.diff()
    result['vvix_regime'] = np.where(vvix_aligned > 25, 1.0, -1.0)
    return result
```

3. Llamar en `compute_macro_features()`:
```python
if 'vvix' in macro_dict:
    feature_dfs.append(compute_vvix_features(gold_daily, macro_dict['vvix']))
```

### Cambiar periodos de indicadores:

**Archivo:** `features/timeframe_features.py`

```python
# Cambiar RSI de 14 a 7 periodos:
result[f'{tf_name}_rsi'] = compute_rsi(df['close'], period=7) / 100.0

# Cambiar medias moviles:
ma_fast = df['close'].rolling(5).mean()   # Cambiar de 10 a 5
ma_slow = df['close'].rolling(20).mean()  # Cambiar de 50 a 20
```

---

## Normalizacion de Features

Todas las features estan normalizadas para facilitar el aprendizaje:

| Tipo | Rango Tipico | Metodo |
|------|--------------|--------|
| Retornos | -0.05 a +0.05 | Sin normalizar (ya son %) |
| RSI | 0 a 1 | Dividido por 100 |
| Trend | -1 a +1 | Discreto |
| Correlaciones | -1 a +1 | Natural |
| Horas | 0 a 1 | Dividido por max |
| Binarios | 0 o 1 | Natural |

---

## Debugging de Features

Para verificar que las features se calculan correctamente:

```python
from features.ultimate_150_features import make_ultimate_features

# Generar features
X, returns, timestamps = make_ultimate_features(base_timeframe='M5')

# Verificar shapes
print(f"Features shape: {X.shape}")  # (N, 152)
print(f"Returns shape: {returns.shape}")  # (N,)

# Verificar valores
print(f"Features mean: {X.mean():.4f}")
print(f"Features std: {X.std():.4f}")
print(f"NaN count: {np.isnan(X).sum()}")
print(f"Inf count: {np.isinf(X).sum()}")
```
