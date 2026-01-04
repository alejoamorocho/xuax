# Sistema de Rewards (Recompensas) del Modelo PPO

Este documento explica en detalle como funciona el sistema de recompensas del entorno de trading.

---

## Resumen Ejecutivo

El modelo usa un sistema de recompensas que busca:
1. **Maximizar ganancias** cuando el mercado sube (si esta en posicion Long)
2. **Minimizar costos** de transaccion
3. **Evitar cambios excesivos** de posicion (flip-flop)
4. **Incentivar exposicion** al mercado (penaliza estar sin posicion)

---

## Formula Principal del Reward

```
reward = PnL - trade_cost - turnover_penalty - flat_penalty + hold_bonus
```

Donde cada componente se calcula asi:

---

## 1. PnL (Profit and Loss)

**Archivo:** `env/xauusd_env.py` linea 87

```python
pnl = self.pos * self.r[self.t]
```

### Que significa:
- `self.pos` = Posicion actual (0 = sin posicion, 1 = Long)
- `self.r[self.t]` = Retorno del precio en este periodo (% de cambio)

### Comportamiento:
| Posicion | Mercado Sube (+1%) | Mercado Baja (-1%) |
|----------|-------------------|-------------------|
| Long (1) | +0.01 (gana) | -0.01 (pierde) |
| Flat (0) | 0 (nada) | 0 (nada) |

### Implicacion:
- El modelo SOLO puede ganar dinero si esta en posicion Long Y el precio sube
- Si esta Flat, no gana ni pierde por movimiento del precio
- **NO hay opcion de Short** (solo Long o Flat)

---

## 2. Trade Cost (Costo por Transaccion)

**Archivo:** `env/xauusd_env.py` lineas 80-83

```python
delta = abs(new_pos - self.pos)  # 0 o 1
trade_cost = self.cost * delta
```

### Parametro por defecto:
```python
cost_per_trade: float = 0.0001  # 0.01% o 1 pip
```

### Comportamiento:
| Accion | Delta | Costo |
|--------|-------|-------|
| Mantener Flat -> Flat | 0 | 0 |
| Mantener Long -> Long | 0 | 0 |
| Abrir posicion (Flat -> Long) | 1 | 0.0001 |
| Cerrar posicion (Long -> Flat) | 1 | 0.0001 |

### Implicacion:
- Cada vez que cambia de posicion, paga 0.01%
- Simula spread + comisiones del broker
- **Puedes modificar** este valor si tu broker tiene costos diferentes

---

## 3. Turnover Penalty (Penalizacion por Rotacion)

**Archivo:** `env/xauusd_env.py` linea 84

```python
turnover_penalty = self.turnover_coef * delta
```

### Parametro por defecto:
```python
turnover_coef: float = 0.0002  # 0.02%
```

### Proposito:
- **Adicional** al costo de transaccion
- Penaliza el "flip-flop" (cambiar posicion constantemente)
- Incentiva al modelo a mantener posiciones mas tiempo

### Comportamiento:
| Accion | Penalizacion |
|--------|--------------|
| Sin cambio | 0 |
| Cambio de posicion | -0.0002 |

### Implicacion:
- El costo TOTAL de cambiar posicion es: `0.0001 + 0.0002 = 0.0003` (0.03%)
- Esto es significativo - el modelo necesita estar seguro antes de cambiar

---

## 4. Flat Penalty (Penalizacion por Inactividad)

**Archivo:** `env/xauusd_env.py` linea 90

```python
flat_pen = self.flat_penalty if new_pos == 0 else 0.0
```

### Parametro por defecto:
```python
flat_penalty: float = 0.00002  # 0.002%
```

### Proposito:
- Penaliza estar sin posicion (Flat)
- Incentiva al modelo a estar expuesto al mercado
- Evita que el modelo aprenda a "no hacer nada"

### Comportamiento:
| Posicion | Penalizacion |
|----------|--------------|
| Long | 0 |
| Flat | -0.00002 |

### Implicacion:
- Es una penalizacion MUY pequena
- Pero acumulada en miles de pasos, motiva al modelo a operar
- **Cuidado:** Si el mercado es bajista, esta penalizacion puede ser contraproducente

---

## 5. Hold Bonus (Bonus por Mantener)

**Archivo:** `env/xauusd_env.py` linea 93

```python
hold_bonus = self.hold_bonus if delta == 0 else 0.0
```

### Parametro por defecto:
```python
hold_bonus: float = 0.00002  # 0.002%
```

### Proposito:
- Recompensa mantener la misma posicion
- Incentiva estabilidad en las decisiones
- Reduce el "ruido" en las senales de trading

### Comportamiento:
| Accion | Bonus |
|--------|-------|
| Mantener posicion | +0.00002 |
| Cambiar posicion | 0 |

---

## Ejemplo Completo de Calculo

### Escenario 1: Mantener Long mientras el precio sube 0.5%

```
Estado inicial: pos = 1 (Long)
Accion: 1 (mantener Long)
Retorno del mercado: +0.005 (+0.5%)

Calculos:
- delta = |1 - 1| = 0
- pnl = 1 * 0.005 = +0.005
- trade_cost = 0.0001 * 0 = 0
- turnover_penalty = 0.0002 * 0 = 0
- flat_penalty = 0 (no esta Flat)
- hold_bonus = 0.00002

REWARD = 0.005 - 0 - 0 - 0 + 0.00002 = +0.00502
```

### Escenario 2: Abrir posicion Long mientras el precio sube 0.5%

```
Estado inicial: pos = 0 (Flat)
Accion: 1 (abrir Long)
Retorno del mercado: +0.005 (+0.5%)

Calculos:
- delta = |1 - 0| = 1
- pnl = 0 * 0.005 = 0  (posicion anterior era 0!)
- trade_cost = 0.0001 * 1 = -0.0001
- turnover_penalty = 0.0002 * 1 = -0.0002
- flat_penalty = 0 (nueva posicion es Long)
- hold_bonus = 0 (cambio de posicion)

REWARD = 0 - 0.0001 - 0.0002 - 0 + 0 = -0.0003
```

**IMPORTANTE:** El PnL se calcula con la posicion ANTERIOR, no la nueva. Esto evita "mirar al futuro".

### Escenario 3: Mantener Flat mientras el precio baja 1%

```
Estado inicial: pos = 0 (Flat)
Accion: 0 (mantener Flat)
Retorno del mercado: -0.01 (-1%)

Calculos:
- delta = |0 - 0| = 0
- pnl = 0 * (-0.01) = 0
- trade_cost = 0
- turnover_penalty = 0
- flat_penalty = -0.00002
- hold_bonus = +0.00002

REWARD = 0 - 0 - 0 - 0.00002 + 0.00002 = 0
```

---

## Tracking del Equity

**Archivo:** `env/xauusd_env.py` linea 98

```python
self.equity *= (1.0 + reward)
```

- El equity se inicializa en 1.0 (100%)
- Se multiplica por (1 + reward) en cada paso
- Permite trackear el rendimiento acumulado

---

## Parametros Configurables

| Parametro | Default | Descripcion | Donde Modificar |
|-----------|---------|-------------|-----------------|
| `cost_per_trade` | 0.0001 | Costo por cambio de posicion | Constructor del env |
| `turnover_coef` | 0.0002 | Penalizacion extra por cambio | Constructor del env |
| `flat_penalty` | 0.00002 | Penalizacion por estar Flat | Constructor del env |
| `hold_bonus` | 0.00002 | Bonus por mantener posicion | Constructor del env |
| `window` | 64 | Barras de historia en observacion | Constructor del env |

### Como modificar:

```python
env = XAUUSDTradingEnv(
    features=X,
    returns=r,
    window=64,
    cost_per_trade=0.0002,    # Aumentar si broker cobra mas
    turnover_coef=0.0001,     # Reducir si quieres mas trades
    flat_penalty=0.0,         # Eliminar si mercado es bajista
    hold_bonus=0.00005,       # Aumentar para mas estabilidad
)
```

---

## Espacio de Acciones

**Archivo:** `env/xauusd_env.py` linea 55

```python
self.action_space = spaces.Discrete(2)
```

| Accion | Valor | Significado |
|--------|-------|-------------|
| 0 | Flat | Sin posicion |
| 1 | Long | Comprado |

**Limitacion importante:** No hay opcion de Short (venta en corto).

---

## Espacio de Observaciones

**Archivo:** `env/xauusd_env.py` linea 49

```python
obs_dim = self.window * self.X.shape[1] + 1
```

La observacion incluye:
- Ultimas `window` barras (64 por defecto) de todas las features
- La posicion actual (0 o 1)

Si tienes 152 features:
```
obs_dim = 64 * 152 + 1 = 9,729 valores
```

---

## Diagrama de Flujo del Reward

```
                    +------------------+
                    |  Accion del AI   |
                    |  (0=Flat, 1=Long)|
                    +--------+---------+
                             |
                             v
              +-----------------------------+
              |  Calcular delta (cambio)    |
              |  delta = |new_pos - old_pos||
              +-------------+---------------+
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
    +-------+------+ +------+------+ +------+-------+
    | PnL          | | Trade Cost  | | Penalties/   |
    | old_pos * r  | | cost*delta  | | Bonuses      |
    +-------+------+ +------+------+ +------+-------+
            |               |               |
            +---------------+---------------+
                            |
                            v
                    +-------+-------+
                    |    REWARD     |
                    | (suma total)  |
                    +-------+-------+
                            |
                            v
                    +-------+-------+
                    | Actualizar    |
                    | Equity y Pos  |
                    +---------------+
```

---

## Consideraciones para Modificaciones

### Si quieres permitir Short:
1. Cambiar `action_space` a `Discrete(3)` (0=Flat, 1=Long, 2=Short)
2. Modificar el calculo de `new_pos` para incluir -1
3. El PnL con Short seria: `pnl = self.pos * self.r[self.t]` (automatico si pos=-1)

### Si quieres posiciones fraccionarias:
1. Cambiar `action_space` a `Box(low=0, high=1, shape=(1,))`
2. El `new_pos` seria el valor continuo de la accion
3. El PnL seria proporcional al tamano de posicion

### Si quieres reward basado en Sharpe:
1. Trackear retornos en una ventana
2. Calcular Sharpe rolling: `mean(returns) / std(returns)`
3. Usar Sharpe como reward en lugar de PnL directo

---

## Archivos Relacionados

| Archivo | Contenido |
|---------|-----------|
| `env/xauusd_env.py` | Definicion del entorno y rewards |
| `train/train_ultimate_150.py` | Script de entrenamiento |
| `train/train_ppo.py` | Entrenamiento PPO basico |

---

## Metricas de Evaluacion

Despues del entrenamiento, evalua el modelo con:

```python
python evaluate_model.py --model train/ppo_xauusd_latest.zip
```

Metricas clave:
- **Total Return:** Retorno acumulado (%)
- **Sharpe Ratio:** Retorno ajustado por riesgo (>2 es bueno)
- **Max Drawdown:** Peor caida desde maximo (%)
- **Win Rate:** % de trades ganadores
- **Profit Factor:** Ganancia bruta / Perdida bruta
