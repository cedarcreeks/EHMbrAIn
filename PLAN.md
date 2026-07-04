# EHMbrAIn — EHM con IA vs EHM Tradicional — Gas Path Analysis en CFM56-7B

**Plan maestro v2 (detalle extremo) — 2026-07-03**

> Trabajo de investigación profunda que compara el **Engine Health Monitoring (EHM)** basado en
> IA frente al EHM tradicional, usando **Gas Path Analysis (GPA)** sobre el turbofan
> **CFM56-7B** (Boeing 737NG). No existen datos reales de degradación ni modelo de performance
> disponible: el proyecto **construye ambos** — un gemelo termodinámico en pyCycle y un generador
> de flota sintética degradada con verdad-terreno — y los libera como benchmark abierto.
>
> **Misión (redefinida 2026-07-04): generar conocimiento TRANSFERIBLE a la industria.** El
> objetivo no es solo mover la frontera del conocimiento sino que una organización de ingeniería
> pueda *usar* los resultados: decidir si integrar IA en su práctica de EHM, cuándo, con qué
> sensores y a qué coste. Sin lenguaje que haga el trabajo inaccesible. Doble audiencia
> (industria + investigación) declarada en el report, cap. 1 "Audience and register".

**Decisiones cerradas:** motor CFM56-7B · herramienta pyCycle · alcance IA completo (detección +
diagnóstico + RUL + híbrido physics-informed) · entregable: informe de investigación accesible +
benchmark abierto (NO TFM; redefinido 2026-07-04).
**Defaults asumidos:** sensores de cabina (N1, N2, EGT, WF) como caso base + ablación con set
extendido (P25/T25, PS3/T3) · flota ~100 motores run-to-failure.

## 0-bis. Principio rector: transferibilidad a la industria

Cada fase debe responder, además de su pregunta científica, una pregunta operativa que un
responsable de flota pueda accionar:

| Fase | Pregunta científica | Pregunta operativa (transferencia) |
|---|---|---|
| F1 | ¿Es fiel el gemelo? | ¿Qué se puede construir SOLO con datos públicos certificados (TCDS, EEDB)? — receta replicable para cualquier motor |
| F2 | ¿Es válido el dataset? | ¿Cómo auditar un dataset sintético antes de fiarse de él? (las 3 auditorías son un protocolo reutilizable) |
| F3 | ¿Qué rinde el EHM clásico medido contra verdad-terreno? | ¿Dónde NO hace falta IA? (el tradicional gana eventos a coste casi nulo) |
| F4 | ¿Qué aporta la IA y dónde? | ¿Qué tarea justifica la inversión? (prognosis 3-6×) ¿Cuáles no todavía? |
| F5 | ¿Sobreviven las hipótesis al pre-registro? | Números citables sin escepticismo + mapa sensores×dato×ruido = guía de adquisición |
| F6 | — | Casos narrados en lenguaje de operación (EGT margin, retiradas, lavados) + coste-beneficio parametrizado |

Reglas derivadas (ya operativas, ahora explícitas):
1. **Métricas operativas junto a las estadísticas** — falsas alarmas por 1000 vuelos, ciclos de
   aviso previo, margen EGT recuperado; no solo F1/AUC/RMSE.
2. **Registro de decisiones** (apéndice del report): una organización que replique esto sabe qué
   elección fue medida, cuál literatura, cuál convención y cuál perilla ajustable.
3. **Scoreboard honesto aunque incomode** — "la IA no gana en detección todavía" es información
   de compra tan valiosa como "gana 3-6× en prognosis".
4. **Todo reproducible con hardware de escritorio** (un portátil + GPU de consumo): si solo corre
   en un clúster, no es transferible.
5. **Datasheet + catálogo de fallos versionados**: el generador es adaptable a otro motor
   cambiando el contrato YAML, no reescribiendo código.

---

## 0. Resumen ejecutivo

Se construye un banco de pruebas reproducible en el que **ambas familias de EHM ven exactamente los
mismos datos** y se evalúan con exactamente las mismas métricas. La clave metodológica es que los
datos son sintéticos pero con **verdad-terreno completa** (estado real de salud de cada componente
en cada vuelo), algo imposible con datos reales: permite medir no solo si un método detecta o
pronostica bien, sino si **atribuye el fallo al componente correcto** — el talón de Aquiles del GPA
clásico (smearing) y la caja negra de la IA (falta de explicación física).

El proyecto produce cuatro artefactos publicables:

1. **SynCFM56** — dataset abierto de flota CFM56-7B sintética con verdad-terreno GPA, formato
   snapshots estilo ACARS (más realista que C-MAPSS).
2. **Pipeline tradicional de referencia** implementado con rigor (GPA WLS + Kalman + reglas +
   trending), documentado como baseline honesto.
3. **Suite de IA** con cuantificación de incertidumbre (conformal prediction) y explicabilidad
   anclada a la física (Physics-Consistency Score).
4. **Protocolo de comparación justa pre-registrado** (hipótesis y análisis congelados en git antes
   de ver resultados del test).

---

## 1. Contribuciones académicas (la innovación, explícita)

Un TFM/paper vive de sus *claims*. Estos son los que este plan permite defender, ordenados de más
seguro a más ambicioso:

- **C1. Benchmark abierto SynCFM56.** Primer dataset sintético público de GPA sobre un motor
  comercial concreto (CFM56-7B) con: (a) verdad-terreno de health parameters por vuelo,
  (b) formato de **snapshots ACARS** (reporte de despegue + reporte de crucero por vuelo, no serie
  temporal continua — así trabaja el EHM real, a diferencia de C-MAPSS), (c) eventos operativos
  etiquetados (lavados de compresor, FOD, derivas de sensor). Publicable por sí solo (data paper).
- **C2. Comparación cabeza-a-cabeza con protocolo pre-registrado.** Hipótesis, métricas, splits y
  presupuesto de ajuste **congelados vía git tag antes de evaluar** (pre-registro, práctica
  estándar en medicina/psicología, casi inédita en PHM). Igual presupuesto de tuning para
  tradicional e IA (mismo nº de trials Optuna). Elimina el sesgo típico de "afinamos la IA y
  dejamos el baseline de paja".
- **C3. Híbrido physics-informed con gemelo pyCycle.** Tres mecanismos combinables y ablacionados
  por separado: (i) *features* de residuo contra el gemelo digital, (ii) pérdidas con restricciones
  físicas (monotonicidad de la degradación entre lavados, reset en lavado), (iii) *stacking*
  GPA→ML: los health params estimados por el Kalman-GPA como entrada de la red de RUL.
- **C4. Cuantificación de incertidumbre comparada.** Intervalos de RUL por **conformal prediction**
  (garantía de cobertura sin supuestos distribucionales) frente a la covarianza del filtro de
  Kalman del método tradicional. Métrica: cobertura empírica vs nominal + anchura de intervalo.
  Conformal en PHM es reciente y con hueco claro de literatura.
- **C5. Explicabilidad anclada a física — Physics-Consistency Score (PCS).** Las atribuciones SHAP
  del clasificador de diagnóstico se proyectan al espacio de health parameters vía la matriz de
  coeficientes de influencia (ICM) y se comparan (similitud coseno) con la firma física del fallo.
  Métrica nueva: mide si la IA "razona" de forma físicamente consistente, no solo si acierta.
- **C6. Estudio de determinabilidad sensores × métodos.** Ablación sistemática: set de cabina
  (4 sensores) vs extendido (8) × nivel de ruido × tamaño de dato → mapa de *cuándo gana cada
  enfoque*. Resultado accionable para operadores (¿qué sensor añadir renta más?).
- **C7. Validación sim-to-real.** La metodología completa (no el modelo del motor) se re-ejecuta
  sobre **N-CMAPSS** (DS02/DS03): si el ranking de métodos se mantiene, las conclusiones no son un
  artefacto del simulador propio. Cierra la crítica obvia del revisor ("todo es sintético tuyo").

*Extensión opcional (stretch, no bloqueante):* **C8.** Generación automática de informe de
mantenimiento en lenguaje natural a partir del diagnóstico + evidencia (LLM con plantilla
estructurada y citas a los datos). Vistoso para la demo; en el paper solo como "future work" salvo
que sobre tiempo.

---

## 2. Hipótesis formalizadas (criterios de aceptación cuantitativos)

Cada hipótesis se evalúa sobre el test set congelado, con test estadístico pareado por motor
(Wilcoxon signed-rank, corrección de Holm para multiplicidad, α = 0.05) e intervalos bootstrap
(10 000 remuestreos, por motor).

| ID | Hipótesis | Métrica de decisión | Criterio de confirmación |
|----|-----------|--------------------|--------------------------|
| H1 | La IA detecta antes con menos falsas alarmas | Lead time mediano (ciclos) y FPR a recall fijo 0.9 | Lead time ≥ +20 % vs trending y FPR ≤ 0.5× vs trending, p < 0.05 |
| H2 | La IA aísla mejor fallos con firmas solapadas | Exactitud de aislamiento en el subconjunto "confusable" (pares de fallos con firmas a < 15° en el espacio ICM) | +10 puntos porcentuales vs GPA-WLS, p < 0.05 |
| H3 | La IA pronostica mejor el RUL | RMSE y score NASA PHM08 | Mejora en ambas, p < 0.05 |
| H4 | El híbrido supera a IA pura y GPA puro, sobre todo con poco dato | RMSE de RUL con 10 %, 25 %, 100 % del train | Híbrido ≥ IA pura en 100 % y estrictamente mejor en 10 %/25 %, p < 0.05 |
| H5 | Los intervalos conformal calibran mejor que la covarianza Kalman | \|cobertura empírica − 0.9\| y anchura media | Menor error de cobertura con anchura no peor que +20 % |

La refutación de cualquier hipótesis **también es resultado publicable** (el pre-registro lo hace
creíble). Prohibido reformular hipótesis después del tag `prereg-v1`.

---

## 3. Fundamentos técnicos

### 3.1 Estaciones y arquitectura (SAE ARP755A)

```
  0/1   2      21/25        3      4       45       5       8/18
  ──▶ [FAN] ─┬─ [BOOSTER]─[HPC]─[CÁMARA]─[HPT]───[LPT] ── tobera core
             └───────────── bypass ───────────────────── tobera fan
  2  = entrada fan            3  = salida HPC (PS3/T3)
  25 = entrada HPC (P25/T25)  4  = salida cámara (T4, no medida)
  45 = entre HPT y LPT ≈ EGT (sonda T495 en el CFM56)
```

Carrete LP: fan + booster + LPT (velocidad N1). Carrete HP: HPC + HPT (velocidad N2).
Control primario del -7B: **N1** (FADEC dual-channel).

### 3.2 Parámetros corregidos

Para comparar vuelos en condiciones distintas, todo se corrige a ISA nivel del mar con
θ = T_t2/288.15 K y δ = P_t2/101.325 kPa:

```
N_corr   = N / √θ            (exponente ajustable a ≈ 0.5)
WF_corr  = WF / (δ · √θ)     (exponente de θ típicamente 0.5–0.7, se calibra en F1)
EGT_corr = EGT / θ
```

Los exponentes exactos por parámetro se **derivan del propio modelo pyCycle** (regresión sobre
barridos de la envolvente), no se toman de tablas genéricas — cabo suelto clásico que aquí se
cierra por construcción.

### 3.3 El problema GPA (formulación)

Estado de salud: vector de desviaciones respecto a motor nuevo
`x = [Δη_fan, ΔΓ_fan, Δη_LPC, ΔΓ_LPC, Δη_HPC, ΔΓ_HPC, Δη_HPT, ΔΓ_HPT, Δη_LPT, ΔΓ_LPT]ᵀ` (10×1),
donde Δη = desviación de eficiencia isentrópica y ΔΓ = desviación de capacidad de flujo corregida,
ambas en % relativo.

Medidas: desviaciones de los parámetros corregidos respecto al baseline de motor nuevo en el mismo
punto de operación, `Δz` (4×1 cabina u 8×1 extendido).

Modelo lineal local:

```
Δz = H(u) · Δx + S · b + v
```

- `H(u)` — **ICM** (influence coefficient matrix), Jacobiano ∂z/∂x evaluado en el punto de
  operación `u` (altitud, Mach, N1, ΔT_ISA, sangrado). Se genera en F1 por diferencias centrales
  con perturbaciones de ±0.5 % y se tabula sobre una rejilla de `u` (interpolación en runtime).
- `b` — sesgos de sensor (aumentan el estado si se estiman), `S` matriz de selección.
- `v` — ruido de medida, `v ~ N(0, R)`, R diagonal con las σ de la tabla §5.4.

**Estimación WLS con regularización (snapshot a snapshot):**

```
x̂ = (Hᵀ R⁻¹ H + λ·P₀⁻¹)⁻¹ Hᵀ R⁻¹ Δz
```

Con 4 medidas y 10 incógnitas el sistema es infra-determinado: rank(H) ≤ 4. Se documenta el número
de condición y los valores singulares de H (análisis SVD en F1) y se aplican las dos salidas
clásicas: (a) regularización con prior P₀ (magnitudes de deterioro esperables) y (b) enfoque
**concentrador de Volponi**: hipótesis de fallo simple → se estima solo el subvector del componente
sospechoso y se elige el que minimiza el residuo.

**Kalman-GPA (seguimiento temporal):** el deterioro evoluciona lento → paseo aleatorio.

```
x_k = x_{k-1} + w_k ,   w ~ N(0, Q)      Q = diag(q_i), q calibrado a tasas de F2
z_k = H(u_k) · x_k + v_k                 R de la tabla de sensores
```

Estado aumentado opcional con sesgos de sensor (para el caso de estudio "sensor engaña al GPA").
En eventos etiquetados de lavado se resetea la parte recuperable del estado (fouling) — el
tradicional también recibe esta información, para que la comparación sea justa.

**Smearing (por qué falla el GPA clásico):** con H mal condicionada, el ruido proyecta un fallo
concentrado (p. ej. solo HPT) sobre varios componentes. El subconjunto "confusable" de H2 se define
computando el ángulo entre columnas/firmas de la ICM: pares con ángulo < 15° son los casos duros.

### 3.4 Mecanismos de degradación (física a modelar)

| Mecanismo | Componente | Efecto típico (literatura) | Perfil temporal | Recuperable |
|---|---|---|---|---|
| Fouling (suciedad) | Fan/booster/HPC | ΔΓ −1…−4 %, Δη −0.5…−2.5 % | Exponencial saturante | Sí (lavado, recupera 30–70 %) |
| Erosión | Compresores | Δη −0.5…−1.5 %, ΔΓ −0.5…−1 % | Lineal lento | No |
| Holgura de punta (clearance) | HPC/HPT | Δη −0.5…−1.5 % | Rápido en primeros ~1000 ciclos, luego lento (rodaje) | No |
| Deterioro sección caliente | HPT | Δη −1…−3 %, ΔΓ +1…+2 % (área efectiva ↑) | Lineal/acelerado con ciclos | No |
| Deterioro LPT | LPT | Δη −0.5…−1.5 % | Lineal lento | No |
| FOD / evento | Fan/HPC | Escalón Δη −0.5…−2 % instantáneo | Escalón (Poisson, λ ≈ 0.01–0.05 /motor/1000 ciclos) | No |
| Desajuste VSV/VBV | HPC | Firma pseudo-fallo de flujo | Escalón o deriva | Mantenimiento |
| Deriva de sensor | Cualquier sonda | Rampa de sesgo (p. ej. EGT +1…+5 °C /1000 ciclos) | Rampa/escalón | Recalibración |

Rangos a refinar en F2 con literatura (Diakunchak 1992; Kurz & Brun; estudios Sallee/CF6 de NASA;
documentación C-MAPSS). El efecto macroscópico agregado a validar: **erosión del EGT margin** desde
~70–100 °C (motor nuevo, día ISA+15) hasta ~0 en 15 000–25 000 ciclos con el patrón realista
"rápido al principio, lento después, dientes de sierra por lavados".

---

## 4. Especificación del motor de referencia (CFM56-7B26/27)

Valores públicos aproximados para calibración (verificar contra TCDS EASA E.004 / FAA E00055EN
durante F1 — tarea WP1.2):

| Magnitud | Valor aprox. | Fuente |
|---|---|---|
| Empuje despegue (7B26 / 7B27) | 117.4 / 121.4 kN | TCDS |
| BPR (crucero) | ~5.1–5.3 | dominio público |
| OPR (despegue) | ~32.7 | dominio público |
| Gasto másico total despegue | ~350–360 kg/s | dominio público |
| Diámetro fan | 1.55 m (61 in) | dominio público |
| N1 100 % / redline | 5175 rpm / 104 % | TCDS |
| N2 100 % / redline | 14 460 rpm / 105 % | TCDS |
| EGT redline (despegue, 5 min) | 950 °C | TCDS |
| Etapas | 1 fan + 3 booster + 9 HPC + 1 HPT + 4 LPT | dominio público |
| TSFC crucero | ~0.60–0.65 lb/lbf/h | dominio público |

El modelo se declara **"representativo del CFM56-7B"**, no certificable — así se redacta en el
paper y se esquiva la crítica de exactitud absoluta: lo que importa a las conclusiones es la
estructura de sensibilidades (ICM), no el tercer decimal del TSFC.

---

## 5. Fases y work packages

Convención: **WPx.y** = work package; cada uno lista tareas, entregable y **DoD** (definition of
done, verificable). Duraciones para 1 persona a dedicación alta. Total ~30 semanas.

### Fase 0 — Encuadre, infraestructura y reproducibilidad (2 sem)

**WP0.1 — Repositorio y entorno.**
Estructura:

```
EHMbrAIn/
├── PLAN.md  README.md  LICENSE(MIT código)  CITATION.cff
├── pyproject.toml  uv.lock            # Python 3.11, uv
├── conf/                              # configs Hydra (motor, datagen, modelos, eval)
├── src/ehmbrain/
│   ├── perf/        # F1: modelo pyCycle, calibración, ICM, decks
│   ├── datagen/     # F2: degradación, flota, sensores, snapshots
│   ├── trad/        # F3: baseline, trending, WLS, Kalman, reglas, RUL clásico
│   ├── ai/          # F4: detección, diagnóstico, RUL, híbrido, UQ, XAI
│   ├── eval/        # F5: métricas, tests estadísticos, ablaciones
│   └── common/      # unidades, correcciones, IO, esquema de datos
├── tests/           # pytest; físicos (signos ICM) + numéricos + regresión
├── notebooks/       # exploración; nada de lógica de producción
├── data/            # DVC-tracked: raw/ interim/ processed/
├── dashboard/       # Streamlit
└── paper/           # LaTeX del paper/TFM, figuras generadas por eval/
```

Herramientas: **uv** (entorno), **DVC** (datos + pipeline `dvc.yaml` con stages
datagen→trad→ai→eval), **MLflow** (experimentos, naming `f{fase}/{tarea}/{modelo}/{fecha}`),
**Hydra** (configs), **pre-commit** (ruff + formato), **GitHub Actions** (CI: tests + pipeline
mínimo con flota de 3 motores "smoke fleet"), semillas globales fijadas y registradas.
**DoD:** `make all` regenera pipeline smoke end-to-end en CI verde.

**WP0.2 — Revisión bibliográfica dirigida.**
Cuatro bloques con fichas de 1 página: (a) GPA clásico — Urban 1972, Volponi (tutoriales GPA,
Kalman, fusión), Doel; (b) benchmarks — Saxena C-MAPSS 2008, Arias-Chao N-CMAPSS 2021, PHM08;
(c) IA en PHM de turbinas — surveys recientes + LSTM/TCN/Transformer para RUL; (d) huecos que
justifican C1–C7 (conformal en PHM, XAI física, pre-registro). Salida directa al estado del arte
del TFM.
**DoD:** documento `paper/related_work.md` con ≥40 referencias clasificadas y los huecos marcados.

**WP0.3 — Especificación de datos y fallos (contrato).**
Congelar: esquema de columnas del snapshot, catálogo de fallos con IDs, sets de sensores, unidades
(SI interno, unidades de display aviación), política de splits.
**DoD:** `conf/data_schema.yaml` y `conf/fault_catalog.yaml` versionados; revisados contra §3.4.

**★ H0 (gate):** CI verde con pipeline smoke; pyCycle "hola mundo" (ciclo ejemplo) corre local y en
CI; contrato de datos congelado.

### Fase 1 — Gemelo de performance CFM56-7B en pyCycle (5–6 sem)

**WP1.1 — Ciclo de diseño (design point).**
Turbofan doble carrete, flujo separado, sangrados de refrigeración HPT/LPT y extracción
customer bleed + potencia mecánica. Punto de diseño: crucero M0.78 / 35 000 ft. Elementos pyCycle:
`Inlet, Fan, Splitter, Duct, Compressor(booster), Compressor(HPC), Combustor, Turbine(HPT),
Turbine(LPT), Nozzle(core), Nozzle(bypass), Shaft(LP), Shaft(HP), Bleeds`.
**DoD:** ciclo converge; BPR, OPR, FPR, T4 dentro de rangos públicos.

**WP1.2 — Off-design y calibración multipunto (MDP).**
Mapas genéricos (pyCycle/NPSS) escalados. Calibración simultánea a 4 puntos ancla: despegue SLS
ISA+15, subida, crucero M0.78/35k, ralentí. Variables de ajuste: escalares de mapa, áreas de
tobera, fracciones de sangrado. Recopilar los valores TCDS reales (tabla §4) y puntos publicados.
**DoD (parte del gate H1):** error ≤ **±3 %** en WF, N2, EGT y empuje en los 4 anclas; tendencias
off-design monótonas correctas.

**WP1.3 — Generador de decks baseline.**
Rejilla de envolvente: altitud {0, 10k, 20k, 31k, 35k, 39k ft} × Mach {0, 0.25, 0.5, 0.78, 0.82} ×
N1 {60–104 %} × ΔT_ISA {−20…+30 °C} × bleed {on/off}. Salida: tablas parquet baseline
(interpolador `scipy` empaquetado en `common/`).
**DoD:** interpolador con error de interpolación < 0.1 % validado por leave-one-out en la rejilla.

**WP1.4 — Generación de la ICM.**
Diferencias centrales ±0.5 % sobre los 10 health params, en cada nodo de una sub-rejilla de
operación. Análisis SVD: rank, número de condición, ángulos entre firmas (define el set
"confusable" para H2). Test físico automático de signos (p. ej. Δη_HPT ↓ ⇒ EGT ↑, WF ↑).
**DoD:** ICM tabulada + informe SVD + tests de signos en verde dentro de `tests/test_icm_physics.py`.

**WP1.5 — Validación y documento del modelo.**
Tabla modelo-vs-referencias, curvas off-design, límites declarados (sin transitorios, sin
humedad, sin Reynolds — documentados como fuera de alcance).
**DoD:** informe `paper/model_validation.md` listo para ser sección del TFM.

**★ H1 (gate):** errores WP1.2 dentro de objetivo; ICM físicamente coherente; decks + interpolador
publicados como artefacto DVC versionado.

### Fase 2 — Degradación, fallos y flota sintética SynCFM56 (5 sem)

**WP2.1 — Librería de trayectorias de salud.**
Por mecanismo (§3.4): forma funcional parametrizada — exponencial saturante
`Δ(n) = A·(1−e^(−n/τ))` para fouling, bilineal para rodaje/clearance, lineal para erosión,
escalón para FOD, rampa para deriva de sensor. Lavados: proceso de renovación cada 500–1500 ciclos
con recuperación Beta(α,β) del fouling acumulado. Composición aditiva sobre el vector x.
**DoD:** cada trayectoria unitaria testeada y ploteada; catálogo `fault_catalog.yaml` implementado
al 100 %.

**WP2.2 — Modelo de flota jerárquico.**
Cada motor i muestrea multiplicadores de tasa `m_i ~ LogNormal(0, σ_flota)` (motores "buenos" y
"malos"), calendario de lavados propio, eventos FOD Poisson, severidad de misión (ratings
derateados 22k/24k/26k/27k lbf como sub-flotas). Run-to-failure: fin de vida = EGT margin ≤ 0 en
despegue día caliente **o** fallo discreto sorteado. **100 motores** (70 train / 10 val / 20 test,
**split por motor**, estratificado por tipo de fallo dominante).
**DoD:** distribución de vidas 15 000–25 000 ciclos con cola realista; auditoría anti-fuga
automatizada (ningún motor en dos splits).

**WP2.3 — Modelo de sensores y snapshots ACARS.**
Por vuelo se emiten **2 snapshots** (como los reportes reales de despegue y crucero estable):
condiciones de operación muestreadas de distribuciones realistas (masa, derate, TAT, altitud de
crucero, Mach) + medidas = verdad del modelo + ruido + sesgo + cuantización + huecos.

| Sensor | σ ruido | Cuantización | Deriva posible |
|---|---|---|---|
| N1, N2 | 0.05–0.1 % | 0.05 % | rara |
| EGT | 2–3 °C | 1 °C | +1…+5 °C/1000 ciclos |
| WF | 0.5 % | 8 kg/h | sí |
| P25, PS3 | 0.25–0.5 % | — | sí |
| T25, T3 | 1–2 °C | — | sí |

Huecos: 2–5 % de snapshots perdidos (MCAR) + rachas (ACARS caído, MNAR).
**DoD:** dataset parquet con doble tabla — `measured` (lo que ve el EHM) y `truth` (x real, RUL
real, etiqueta de fallo, eventos) — más diccionario de datos completo.

**WP2.4 — Auditoría de realismo y dificultad.**
(a) Sanity físico: DEGT/DWF/DN2 con signos y magnitudes coherentes con la ICM; (b) dificultad: un
clasificador trivial (logistic sobre snapshot crudo) **no** debe superar ~60 % de aislamiento — si
lo hace, el dataset es demasiado fácil → subir ruido/solape; (c) comparación cualitativa de
tendencias EGTM contra patrones publicados.
**DoD:** informe de auditoría + ajuste iterado; parámetros finales congelados y versionados.

**★ H2 (gate):** SynCFM56 v1.0 congelado (tag DVC + zenodo draft), auditado, sin fuga, con ficha
de dataset (datasheet for datasets) redactada.

### Fase 3 — EHM tradicional de referencia (4 sem)

Regla de oro: implementarlo **tan bien como se pueda** — el paper solo es creíble si el baseline es
fuerte. Presupuesto de tuning idéntico al de la IA (WP5.2).

**WP3.1 — Baseline y desviaciones.** Corrección de parámetros (§3.2) + interpolador de decks →
DEGT, DWF, DN2, DN1 por snapshot. Suavizado exponencial doble (nivel + tendencia) como los
sistemas OEM tipo SAGE/ADEM.
**WP3.2 — Trending y alertas.** Detección por umbral sobre desvío suavizado + reglas de
persistencia (k de n), y detección de escalón (CUSUM) para eventos tipo FOD. Umbrales calibrados en
val a FPR objetivo.
**WP3.3 — GPA WLS snapshot.** §3.3 con regularización y concentrador de Volponi; salida: x̂ y
residuo por hipótesis de fallo.
**WP3.4 — Kalman-GPA.** Filtro con estado x (10) + sesgos opcionales; Q calibrada en val; reset
parcial en lavados. Salida: trayectoria de salud por motor con covarianza.
**WP3.5 — Aislamiento por reglas.** Árbol de decisión experto sobre firmas (dirección y magnitud
relativa de DEGT/DWF/DN2 + salida GPA). Documentar cada regla con su justificación física.
**WP3.6 — RUL tradicional.** Extrapolación de EGTM: ajuste lineal/exponencial robusto (Theil–Sen)
sobre ventana reciente → ciclos hasta EGTM = 0; intervalo por la covarianza del ajuste/Kalman.
**DoD de fase:** pipeline `trad/` corre sobre todo el test en < 1 h y emite el mismo formato de
salida (contrato `eval/`) que la IA.

**★ H3 (gate):** métricas §6 completas para el tradicional; informe de límites observados
(smearing medido con verdad-terreno — figura clave del paper).

### Fase 4 — EHM con IA (6–7 sem)

Contrato común: misma entrada (snapshots medidos), mismo formato de salida que `trad/`, todo
registrado en MLflow, splits idénticos.

**WP4.1 — Ingeniería de features y datamodule.** Ventanas deslizantes (N = 30–100 ciclos) sobre
desvíos suavizados + condiciones de operación + flags de evento (lavado). Normalización por motor
train-only. Dataloaders PyTorch + versión tabular para XGBoost.
**WP4.2 — Detección de anomalías.** Autoencoder (denso y variacional), Isolation Forest, one-class
SVM; score suavizado EWMA; umbral por cuantil en val. Métrica primaria: lead time y FPR (H1).
**WP4.3 — Diagnóstico.** XGBoost, MLP y CNN-1D multiclase (catálogo de fallos + "sano");
desbalanceo con focal loss/pesos; salida probabilística calibrada (temperature scaling).
**WP4.4 — Pronóstico RUL.** LSTM, TCN y Transformer pequeño (comparados); target = ciclos hasta
fin de vida; truco estándar RUL cap (p. ej. 130) evaluado en ablación; pérdida MSE vs pérdida
asimétrica (alineada con score NASA) en ablación.
**WP4.5 — Híbrido physics-informed (C3).** Tres mecanismos ablacionados: (i) residuos del gemelo
como features extra, (ii) pérdida con penalización de no-monotonicidad del health index entre
lavados, (iii) stacking: x̂ del Kalman-GPA como entrada. Combinación completa = modelo "PI-full".
**WP4.6 — Incertidumbre (C4).** Split conformal sobre residuos de RUL en val → intervalos con
cobertura nominal 90 %; comparar con intervalos del WP3.6. También conformal para el clasificador
(conjuntos de predicción).
**WP4.7 — Explicabilidad (C5).** SHAP sobre diagnóstico; proyección a espacio de health params vía
ICM⁺ (pseudo-inversa); **PCS** = similitud coseno entre firma SHAP-implicada y firma ICM del fallo
verdadero; distribución de PCS por clase y correlación PCS-acierto.
**WP4.8 — Robustez.** Estrés: +50 %/+100 % ruido, sesgo de sensor no visto, derates fuera de
distribución, 10 %/25 % del train (para H4).
**DoD de fase:** todos los modelos versionados, reproducibles con `dvc repro`, tarjetas de modelo
(model cards) escritas.

**★ H4 (gate):** la IA supera al tradicional en ≥1 métrica primaria por tarea con p < 0.05
(pre-chequeo en val; la confirmación formal es en F5 sobre test).

### Fase 5 — Evaluación comparativa pre-registrada (3 sem)

**WP5.1 — Pre-registro (C2).** Documento `paper/prereg.md`: hipótesis H1–H5 con umbrales de §2,
métricas exactas, splits (hashes de los ficheros), tests estadísticos, presupuesto de tuning, regla
de parada. **Git tag `prereg-v1` antes de tocar el test set.**
**WP5.2 — Presupuesto de tuning justo.** Optuna, **mismo nº de trials** (p. ej. 100) para
hiperparámetros del tradicional (umbrales, λ, Q, ventanas) y de cada familia IA, solo con
train+val.
**WP5.3 — Evaluación congelada.** Una única pasada sobre test por método. Métricas §6 + tests
(Wilcoxon pareado por motor, Holm, bootstrap BCa 10k). Tamaños de efecto (Cliff's delta) además de
p-valores.
**WP5.4 — Ablaciones (C6).** Matriz: {4 vs 8 sensores} × {ruido ×1, ×1.5, ×2} × {tamaño de dato
10/25/100 %} × {método}. Resultado: mapas de calor "quién gana dónde".
**WP5.5 — Sim-to-real (C7).** Re-ejecutar detección + RUL (tradicional simplificado e IA) sobre
N-CMAPSS DS02/DS03 con adaptación mínima documentada. Se compara el *ranking* de métodos, no los
números absolutos.
**★ H5 (gate):** todas las hipótesis resueltas con números; `make paper-figures` regenera todas
las figuras desde datos crudos; ranking estable (o discrepancia analizada) en N-CMAPSS.

### Fase 6 — Casos de estudio, dashboard y redacción (3–4 sem)

**WP6.1 — Casos de estudio narrados (5).**
(a) Fouling HPC con dientes de sierra y lavado — el tradicional lo maneja bien (honestidad);
(b) deterioro HPT progresivo hasta límite EGTM — comparación de horizonte de pronóstico;
(c) deriva de sensor EGT que induce falso diagnóstico en GPA sin estado de sesgo, detectada por la
IA (y por el Kalman aumentado — matiz importante);
(d) FOD escalón — CUSUM vs detección IA, lead time;
(e) fallo "confusable" (firmas a <15°) — smearing del WLS vs clasificador + PCS.
**WP6.2 — Dashboard Streamlit.** Vista flota (ranking de riesgo), vista motor (tendencias, salud
estimada vs verdad, RUL con intervalos), vista comparación (tradicional vs IA lado a lado), replay
temporal de casos de estudio.
**WP6.3 — Paper/TFM.** Estructura TFM: 1 Introducción y motivación · 2 Estado del arte ·
3 Modelo de performance CFM56-7B · 4 Generación de flota sintética · 5 EHM tradicional ·
6 EHM con IA · 7 Protocolo experimental pre-registrado · 8 Resultados · 9 Discusión y limitaciones
· 10 Conclusiones y trabajo futuro. Paper derivado (formato corto) para venue.
Venues candidatos: **PHM Society Annual Conference**, ASME Turbo Expo (GT), IEEE Aerospace,
*Aerospace Science and Technology*, MDPI *Aerospace* (data paper de SynCFM56 aparte).
**WP6.4 — Publicación de artefactos.** Repo público (MIT), dataset en Zenodo (CC-BY, DOI),
CITATION.cff, README con replicación en un comando.
**★ H6 (gate — cierre):** demo end-to-end; memoria completa revisable; `make all` desde cero
reproduce dataset → modelos → figuras → tablas del paper.

---

## 6. Métricas (definiciones exactas)

**Detección.**
- Lead time = ciclos entre primera alerta sostenida (k de n) y el evento/umbral de verdad-terreno;
  mediana por motor.
- FPR a recall fijo 0.9; F1; ROC-AUC y PR-AUC (PR es la primaria: clases desbalanceadas).
- Falsas alarmas por 1000 vuelos (métrica operativa que entiende un operador).

**Diagnóstico/aislamiento.**
- Exactitud top-1 y top-2, matriz de confusión, macro-F1.
- Exactitud en subconjunto confusable (H2).
- **Smearing index** (solo posible con verdad-terreno): fracción de ‖x̂‖₁ asignada a componentes
  sanos. Se aplica igual a GPA (x̂ directo) y a IA (vía atribución §WP4.7).
- **PCS** (C5): cos(firma_SHAP_proyectada, firma_ICM_fallo_real).

**Salud/GPA.** RMSE de x̂ vs x real por parámetro; sesgo; correlación de trayectoria.

**RUL.**
- RMSE, MAE.
- **Score NASA PHM08**: `S = Σᵢ [exp(−dᵢ/13) − 1]` si dᵢ<0, `[exp(dᵢ/10) − 1]` si dᵢ≥0, con
  d = RUL_pred − RUL_real (penaliza más predecir tarde).
- **α-λ accuracy**: fracción de predicciones dentro de ±α=20 % del RUL real evaluada en
  λ = {0.5, 0.7, 0.9} de la vida.
- Horizonte de pronóstico (primer instante desde el cual la predicción permanece dentro de ±α).
- UQ: cobertura empírica de intervalos 90 % y anchura media normalizada (H5).

**Coste-beneficio (traducción operativa).** Modelo simple parametrizado: coste de retirada no
programada ≫ coste de inspección ≫ coste de falsa alarma; se barren los ratios y se reporta el
ahorro esperado por política de decisión derivada de cada método. Sensibilidad incluida (no un
número mágico).

**No funcionales.** Tiempo de cómputo por snapshot, tamaño de modelo, datos mínimos para
rendimiento aceptable, interpretabilidad (cualitativa, rúbrica).

---

## 7. Cronograma (30 semanas)

| Semanas | Fase | Contenido | Hito |
|---|---|---|---|
| 1–2 | F0 | Repo, CI, DVC/MLflow, biblio, contrato de datos | **H0** |
| 3–5 | F1 | Ciclo diseño + off-design + calibración anclas | |
| 6–8 | F1 | Decks, ICM, SVD, validación, informe modelo | **H1** (s8) |
| 9–11 | F2 | Trayectorias, flota jerárquica, sensores/ACARS | |
| 12–13 | F2 | Auditoría realismo/dificultad, congelado v1.0 | **H2** (s13) |
| 14–17 | F3 | Baseline, trending, WLS, Kalman, reglas, RUL clásico | **H3** (s17) |
| 18–20 | F4 | Features, detección, diagnóstico | |
| 21–23 | F4 | RUL, híbrido PI, conformal, SHAP/PCS, robustez | **H4** (s23) |
| 24 | F5 | **Pre-registro (tag) + tuning justo** | |
| 25–26 | F5 | Evaluación test, ablaciones, N-CMAPSS | **H5** (s26) |
| 27–28 | F6 | Casos de estudio + dashboard | |
| 29–30 | F6 | Redacción final, artefactos públicos, defensa | **H6** (s30) |

Colchón implícito: los rangos de fase (§5) suman hasta 32 semanas si algo se tuerce; el camino
crítico es F1→F2 (todo depende del gemelo y del dataset).

---

## 8. Riesgos y mitigaciones

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Calibración pyCycle no llega a ±3 % | Media | Alto | Relajar a ±5 % con justificación; lo crítico es la ICM (sensibilidades), no el absoluto; fallback GSP |
| Curva de aprendizaje pyCycle/OpenMDAO | Media | Medio | Empezar por ejemplos oficiales (H0); comunidad OpenMDAO activa; reservar 1 sem extra en F1 |
| Dataset demasiado fácil → IA gana trivialmente | Media | Alto | Auditoría de dificultad WP2.4 con umbral explícito; ruido/solape realistas; revisor externo del datasheet |
| Baseline tradicional débil → comparación de paja | Media | Alto | WP3 con mismo presupuesto de tuning; reglas documentadas con física; smearing medido, no asumido |
| Fuga train/test | Baja | Alto | Split por motor automatizado + test de fuga en CI |
| Sobre-alcance (C1–C7 es mucho) | Alta | Medio | C1–C5 son el núcleo; C6 recortable a 2 celdas; C7 recortable a solo RUL; C8 fuera por defecto |
| Resultados refutan hipótesis | Media | Bajo | El pre-registro convierte la refutación en resultado publicable |
| Cómputo insuficiente para Transformers | Baja | Bajo | Modelos pequeños bastan (datos tabulares/ventana); Colab/GPU puntual si hace falta |

---

## 9. Cabos sueltos cerrados (decisiones menores, fijadas ya)

1. **Unidades:** SI interno; display aviación (lbf, °C, kg/h) solo en dashboard/figuras.
2. **Humedad, Reynolds, transitorios:** fuera de alcance, declarados como limitación.
3. **Exponentes de corrección:** derivados del propio modelo (WP1.3), no genéricos.
4. **ICM dependiente del punto de operación:** tabulada e interpolada, no una única matriz.
5. **Información de lavados:** disponible para ambos métodos (registro de mantenimiento realista).
6. **RUL cap:** decidido en ablación (WP4.4), mismo valor para todos los métodos.
7. **Semillas:** 5 semillas por modelo IA; se reporta media ± desviación; el tradicional es
   determinista (ventaja que se comenta en la discusión).
8. **Formato de intercambio:** parquet + esquema versionado; contrato de salida único para
   `trad/` y `ai/` de modo que `eval/` sea ciego al origen.
9. **Licencias:** código MIT, dataset CC-BY 4.0, pyCycle es Apache-2.0 (compatible).
10. **Hardware:** todo el pipeline debe correr en un portátil (CPU) salvo entrenamiento de redes
    (GPU pequeña); presupuesto de cómputo se registra y reporta.
11. **Idioma:** memoria TFM en español, paper y repo en inglés.
12. **Criterio de fin de vida:** EGTM ≤ 0 en despegue día caliente (ISA+15 derate real) o fallo
    discreto — el mismo para verdad-terreno, tradicional e IA.

---

## 10. Acciones inmediatas (arranque H0)

1. `uv init` + esqueleto de repo (§WP0.1) + primer commit.
2. Instalar pyCycle y correr su ejemplo de turbofan (verificación de entorno).
3. Redactar `conf/fault_catalog.yaml` y `conf/data_schema.yaml` desde §3.4 y §WP2.3.
4. Descargar TCDS EASA E.004 (CFM56-7B) y volcar la tabla §4 con valores verificados.
5. Abrir `paper/related_work.md` con las 4 fichas de WP0.2.


---

## F7 (propuesta 2026-07-04) — Tomografía por puntos de operación: MOPA oportunista aprendido

**Idea**: H2 dijo "compra sensores"; F7 explora la salida sin hardware — H(u) varía con la
condición de vuelo, luego una secuencia de snapshots ordinarios a condiciones dispersas son
proyecciones distintas del mismo estado de salud. Fusionarlas = tomografía del gas path.
Factibilidad YA medida en nuestra ICM: apilar los 6 puntos restaura rango 10/10 y duplica la
separación de pares u-rompibles (η_HPT~Γ_HPT 6.0°→13.1°); η_HPC~η_HPT casi no se mueve
(1.32°→1.66°) → existe un mapa u-rompible vs fundamental, y ese mapa es contribución.

**Novedad honesta** (WP7.0 la endurece): MOPA clásico existe (puntos deliberados, salud
constante, estimación clásica); los huecos = scatter OPORTUNISTA de servicio + salud que
DERIVA dentro de la ventana de fusión (el modelo temporal aprendido separa deriva de fallo
mientras fusiona — claim técnico central) + adjudicación con verdad-terreno + conjuntos de
aislamiento conformal como salida de ambigüedad calibrada.

Hipótesis H7.1–H7.4, WPs 7.0–7.6, prereg-v2, misma disciplina de gates: detalle completo en
`docs/f7-proposal.md`.


---

## F8 — Programa de superación de limitaciones (directriz 2026-07-04)

**Directriz del usuario**: las limitaciones declaradas no son notas al pie — son la frontera.
Cada una se convierte en línea de investigación con hipótesis de descubrimiento.

| # | Limitación declarada | Vía de superación | Posible descubrimiento |
|---|---|---|---|
| L1 | Generación linealizada (baseline+ICM·x) | Sustituto neuronal DIFERENCIABLE de pyCycle (entrenado con solves reales; error < auditoría actual) → generación no lineal a coste ~0 | El sustituto habilita análisis-por-síntesis (diagnóstico = inversión del twin) y pérdidas físicas exactas — el mecanismo híbrido no probado de H4 |
| L2 | 2 condiciones de snapshot + scatter | Generador de mix de misión (múltiples reportes/vuelo, FL/Mach variados) — sinergia directa con F7 (calendario diseñado) | Cuantificar el valor diagnóstico marginal de CADA reporte del calendario (curva valor-vs-coste de datos) |
| L3 | EGT proxy en estación 4.5 | Añadir canal T49.5 (interetapa LPT) del propio modelo + shunt de display como sesgo conocido | ¿Cambia el mapa de confusabilidad con la estación real? (la geometría ICM depende de dónde mides) |
| L4 | Etiquetas crónicas = proxy de edad | Tareas a nivel de MECANISMO usando las contribuciones por-mecanismo ya almacenadas | Separabilidad fouling-vs-erosión: ¿distinguible el lavable de lo permanente sin evento de lavado? |
| L5 | Una arquitectura por tarea | Barrido TCN/Transformer bajo presupuesto F5-style | ¿La ventaja RUL es del aprendizaje o del GRU? (robustez de conclusión) |
| L6 | H4 = solo mecanismo stacking | Los 2 restantes: residuos-del-twin (vía L1) y pérdida con restricción física | ¿Existe ALGÚN híbrido que gane? Si no: resultado fuerte contra la etiqueta "physics-informed" |
| L7 | Deriva de sensor sin estimar | Kalman de estado aumentado + tarea IA de detección de sesgo (cross-channel) | ¿Quién detecta antes el termopar mentiroso? (Caso C resuelto) |
| L8 | Sim-to-real solo FD001 | N-CMAPSS DS02 (pipeline de descarga/HDF5 dedicado) | Transferencia con física de vuelo real por ciclo |
| L9 | PCS nulo (clasificador débil) | Re-evaluar PCS sobre el learner F7 (competente) | ¿La métrica separa razonamiento físico de ruido cuando hay señal? (validación de C5) |
| L10 | Mapas genéricos escalados | Calibrar parámetros de FORMA de mapa contra la curva EEDB completa (4 puntos + working line) | ¿Cuánta forma de mapa es recuperable solo de datos públicos? |

Orden sugerido por sinergia: L1 (sustituto diferenciable — desbloquea L2, L6, F7-drift) → L2+F7
juntos → L4, L7 (tareas nuevas sobre flota v2) → L3, L5, L9 → L8, L10. Cada línea con
hipótesis pre-registrable y gate propio, misma disciplina H0–H6.


---

## F9 — Estándar handbook/tesis doctoral para TODO el report (directriz 2026-07-04, ampliada)

**Directriz ampliada del usuario**: las reglas de completitud no son solo para conceptos
sueltos — el documento entero debe ser handbook completo Y tesis doctoral: todos los
ingredientes que intervienen bien presentados y caracterizados (origen de los datos, qué
representan, cómo se generan, qué es synthetic data, metodologías explicadas, cada técnica
de IA detallada). Norma permanente N7.

**Checklist de elevación por capítulo** (pasada retroactiva):
- [x] ch. nuevo "Data: origin, meaning, and the synthetic-data method" — ecosistema real de
  datos EHM (ACARS/QAR/shop), significado físico de cada canal, qué es dato sintético +
  taxonomía + defensas, cadena de procedencia TikZ, anatomía de una fila
- [x] ch3 maquinaria estadística completa (parte 1 hecha)
- [x] ch6/ch7/ch10 matemática de cada técnica (parte 1 hecha)
- [ ] ch7: autoencoder con ecuaciones, SHAP/Shapley fórmula, tabla de caracterización de
  TODOS los ingredientes IA (modelo/tarea/entradas/parámetros/entrenamiento/modos de fallo)
- [ ] ch3: sección "diseño de un benchmark justo como método" (axiomas de equidad,
  jerarquía de evidencia exploratorio/confirmatorio, filosofía de gates)
- [ ] ch2: revisar GPA/termodinámica al estándar (derivaciones guiadas, ejemplos)
- [ ] ch4: revisar calibración (qué es TCDS/EEDB en detalle — ahora en ch. datos, enlazar)
- [ ] ch5: enlazar con ch. datos; caracterizar cada mecanismo de degradación con su modelo
- [ ] ch8-11: revisar que cada resultado remite a la maquinaria definida
- [ ] Gate F9: relectura completa — ningún término antes de definirse; notación al día

## F9 (original) — Completitud conceptual del report (directriz 2026-07-04)

**Directriz del usuario**: desarrollar TODOS los conceptos planteados vagamente — sin modelo
matemático ni explicación guía — hasta el estándar del resto del documento.

**Procedimiento** (pasada sistemática, capítulo a capítulo):
1. Inventario: barrer el report buscando conceptos usados sin (a) definición formal,
   (b) modelo matemático, (c) explicación guía para el lector novato. Candidatos conocidos:
   conformal prediction / APS (se usa, nunca se deriva: cuantil conforme, garantía de
   cobertura, construcción del set adaptativo), test de McNemar (b/c discordantes, binomial
   exacta), Wilcoxon signed-rank, corrección de Holm, bootstrap BCa, delta de Cliff,
   distancia de Mahalanobis + shrinkage Ledoit-Wolf, Theil-Sen (mediana de pendientes),
   CUSUM (derivación log-likelihood), suavizado Holt (ecuaciones de nivel/tendencia),
   Kalman-GPA (predicción/actualización completas), PCS (proyección H+ y coseno),
   GRU (ecuaciones de puertas), Optuna/TPE (qué optimiza), gradiente descendente/backprop
   (está en primer ML pero sin ecuación), SVD/rango/ángulos de firma (parcial en ch4).
2. Para cada concepto: recuadro o subsección con — modelo matemático completo con notación
   del documento · explicación guía en 1-2 párrafos (por qué funciona, qué rompe si falta) ·
   ejemplo numérico trabajado cuando quepa · referencia canónica.
3. Gate F9: relectura completa — ningún término técnico usado antes de definirse; verificar
   con el índice de notación (ch. 0) actualizado.
4. Norma nueva N6: todo concepto nuevo que entre al report a partir de ahora entra CON su
   modelo matemático y explicación guía (no se acumula deuda conceptual).
