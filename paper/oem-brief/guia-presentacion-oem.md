# Guía para presentar el deck OEM

Para un jefe que **sabe de performance a alto nivel** y es **muy crítico con la IA**.
Las diapositivas están en inglés; esta guía es lo que dices tú, en llano.

---

## Antes de empezar: la postura

**Tu ventaja es que no vienes a vender IA.** Vienes a contar que la mediste y que casi
todo se cayó. Eso desarma al escéptico en el primer minuto, y a partir de ahí te escucha.

**Regla de oro:** cada vez que digas algo bueno de la IA, di antes el número que lo limita.
Nunca al revés.

**Tres cosas que NO debes decir:**
- "La IA revoluciona el mantenimiento" — pierdes la sala.
- "Nuestro modelo aprende patrones complejos" — suena a folleto.
- Cualquier cifra sin decir contra qué se compara.

**Frase de arranque sugerida:**
> "Monté un banco de pruebas para medir si lo que se publica sobre IA en performance
> se sostiene. Te adelanto el resultado: la mayoría no. Incluido lo mío. Lo que sí
> queda es pequeño y, curiosamente, no es software."

---

## 1. Portada

Solo léela y pasa. No te detengas.

---

## 2. *Start from the skeptical position*

**Qué decir:**
> "La pregunta era simple: ¿la IA gana al método clásico de análisis de gas path, o no?
> La trampa habitual es comparar una IA bien ajustada contra un método clásico mal
> ajustado. Aquí les di a los dos el mismo dato, la misma métrica y el mismo presupuesto
> de ajuste. Y escribí las hipótesis **antes** de correr nada, para no poder cambiarlas
> después.
>
> A la derecha tienes lo que salió. Un paper publicado cuyo titular resulta ser ruido.
> Dos titulares míos que se caen a la tercera parte. Y la tarea donde más se vende la
> IA acaba en empate."

**Idea en una frase:** no vengo a convencerte, vengo a enseñarte una auditoría.

---

## 3. *The rig, in one slide*

**Qué decir:**
> "Construí un motor virtual del CFM56-7B calibrado contra datos de certificación
> públicos, y con él una flota de cien motores que viven hasta que salen de ala. La
> clave es esta: **de cada motor sé la verdad**. Sé exactamente cuánto se ha degradado
> cada componente en cada vuelo.
>
> Eso con datos reales no lo puedes hacer. Con datos reales solo puedes comparar una
> estimación contra otra estimación. Aquí puedo preguntar lo que de verdad importa:
> ¿el diagnóstico **acertó**?"

**Si pregunta "¿y eso es realista?":**
> "El modelo está anclado a medidas de certificación reales que nunca usé para
> ajustarlo. Los números absolutos no los defiendo. Lo que se transfiere son los
> **órdenes de magnitud y quién gana a quién**, que es de lo que va todo esto."

---

## 4. *A peer-reviewed deep-learning result, replicated exactly*

Esta es la diapositiva importante. Tómate tiempo.

**Qué decir:**
> "Cogí un paper de 2026, publicado en Springer, revisado por pares. Dice que una red
> neuronal concreta es la mejor de quince arquitecturas para predecir vida remanente.
> Lo reimplementé exactamente como está escrito.
>
> Primero, lo justo: **replica**. Me sale 14,56 donde ellos publican 14,12. No estoy
> montando un hombre de paja.
>
> Ahora el problema. Ellos ordenan quince arquitecturas con diferencias de entre 0,12 y
> 0,60. Pero si entrenas **el mismo modelo dos veces**, cambiando solo el número
> aleatorio de arranque, el resultado se mueve 2,94. La diferencia entre modelos es
> cinco veces más pequeña que el ruido de repetir el mismo modelo. **Ese ranking no
> mide nada.** Y ellos publican una sola corrida por modelo.
>
> Y hay algo peor, los puntos rojos: en cinco de diez corridas el modelo **no llega a
> entrenar**. Se queda prediciendo cero. Y la causa está en sus propias tablas: una
> configuración que, si el modelo se atasca al principio, no puede recuperarse."

**Analogía si hace falta:**
> "Es como cronometrar quince motores en banco, con diferencias de una décima entre
> ellos, cuando repetir el ensayo del mismo motor te da tres décimas de dispersión."

---

## 5. *The evaluation question to ask anyone*

**Qué decir:**
> "De todo esto, si te llevas una sola cosa, que sea esta pregunta: **¿cuál es la
> dispersión entre repeticiones?** Antes de mirar el titular.
>
> Sirve para un proveedor que te enseña un piloto. Sirve para un equipo interno.
> Y sirve para mí — de hecho me la apliqué, y es la siguiente diapositiva."

**Idea en una frase:** un ranking dentro del ruido no es un resultado.

---

## 6. *We applied it to our own claims, and both shrank*

**Qué decir:**
> "Me apliqué el mismo criterio y perdí dos titulares.
>
> El primero: yo decía que la IA daba intervalos de incertidumbre seis veces más
> estrechos. Falso tal como estaba dicho. Mi intervalo estaba **calibrado** — ajustado
> para que cuando digo noventa por ciento, sea de verdad noventa. El del método clásico
> no lo estaba, y por eso salía anchísimo. Calibras los dos igual y la ventaja se queda
> en 1,34. Tres cuartas partes de aquel seis era la calibración, no el modelo. Y la
> calibración es una técnica que el lado clásico puede coger gratis.
>
> El segundo: 'la IA predice la vida remanente entre 2,3 y 4,4 veces mejor'. Cierto,
> pero solo contra la extrapolación lineal que se usa hoy en operación. Contra un
> método clásico bueno es 1,34 veces al noventa por ciento de la vida, y **empate** al
> setenta."

**Idea en una frase:** cuando la línea base es competente, la ventaja es del 30 %, no
de varias veces.

---

## 7. *Where AI lost outright*

**Qué decir:**
> "Aquí la IA no gana: pierde el empate. Distinguir entre averías parecidas de gas
> path: 31 % el método clásico, 31 % la IA. Idéntico.
>
> Y esto no es que el algoritmo sea malo. Es que **la información no está en el dato**.
> Tienes tres medidas de cabina útiles y diez parámetros de salud que estimar. Es como
> pedirme tres números que sumen diez: hay infinitas respuestas y ninguna manera de
> elegir. Encima, dos averías distintas dejan huellas separadas 1,3 grados: casi la
> misma firma.
>
> Lo comprobé por cuatro caminos independientes. Reduje a la mitad el ruido de todos
> los sensores: no cambia. Metí un 'sensor virtual', que es lo que venden algunos
> proveedores: **cero, exactamente cero**. Barrí todo el envolvente de operación:
> sigue en 1,38 grados. Y ningún método aprendido lo movió."

**Remate:**
> "Ningún algoritmo arregla que falte información."

---

## 8. *And most prognostic error is not a method gap*

**Qué decir:**
> "Y una última mala noticia, que en realidad es dinero. El 87 % de la incertidumbre en
> vida remanente a mitad de vida es **irreducible**. Dos motores que miden exactamente
> igual fallan de verdad con cientos de ciclos de diferencia. No es que no sepamos
> medirlo: es que el futuro no está escrito en la medida de hoy.
>
> Esto lo puedo afirmar porque conozco la verdad de cada motor. Y lo que significa para
> presupuesto es directo: si financias un algoritmo mejor para predecir a mitad de vida,
> estás persiguiendo ruido. Al final de la vida sí se abre un margen de cuatro veces:
> **ahí** sí paga."

---

## 9–10. *An instrumentation design tool* / *Why this one is ours to act on*

Aquí cambias de tono. Has pasado media presentación diciendo que no. Ahora el sí.

**Qué decir:**
> "Lo más sólido que salió de todo esto **no es IA**. Es teoría de estimación clásica
> aplicada a través del modelo del motor.
>
> La idea: para un motor y sus condiciones reales de vuelo, puedo calcular cuánta
> información sobre cada componente llega de verdad a los sensores. Y de ahí sale un
> límite: esto lo puedes saber, esto no lo puede saber **nadie** con esos sensores.
>
> Mira la tabla. Si añades sondas de estación, la eficiencia del compresor de alta pasa
> a estimarse 45 veces mejor, y distinguir las averías confusas pasa de 0,15 a 0,92.
> Si en cambio reduces el ruido de las sondas que ya tienes: no cambia nada. Si compras
> software con 'sensores virtuales': no cambia nada.
>
> Y aquí está la consecuencia para nosotros: cualquier servicio de EHM montado **solo
> sobre datos de cabina** tiene un techo que ningún roadmap de software levanta. La
> salida es instrumentación. Y la instrumentación la decidimos nosotros, no el cliente."

**Idea en una frase:** esto pone precio a una decisión de hardware **antes** de
comprometer el hardware.

---

## 11. *The one AI result that held up*

**Qué decir:**
> "Y ahora sí, lo único de IA que aguantó, y es más estrecho y más útil que lo que se
> suele contar.
>
> Meter física dentro del modelo — darle al aprendiz la estimación del estado del motor
> — **empeora** la predicción de vida remanente. Lo comprobé dos veces y las dos falló.
> Pero esa misma inyección **mejora mucho** decir qué mecanismo se está comiendo el
> motor.
>
> Y lo monté con un brazo de control, para que no me pudiera engañar a mí mismo: si el
> control también hubiera ganado, entonces la mejora sería solo 'más datos de entrada'.
>
> La explicación es mecánica y tiene sentido: la vida remanente ya va en el margen de
> EGT que el modelo está viendo, así que el estado físico no le añade nada. Pero **qué
> módulo** se está degradando es justo lo que ese estado codifica.
>
> Conclusión: 'physics-informed' no es ni magia ni humo. **Depende de la tarea.** Y
> ahora sé de cuál."

---

## 12. *What this is, and what it is not*

**Di esta diapositiva entera, sin acelerar.** Es la que te da credibilidad.

**Qué decir:**
> "Qué **no** es esto. No es un producto. No es una herramienta validada para uso
> operativo. Y los números absolutos no los defiendo: la flota es sintética y el modelo
> está calibrado contra datos públicos, no contra nuestro banco.
>
> Qué **sí** es: un método reutilizable para decidir qué afirmaciones sobre IA en
> performance financiamos y cuáles rechazamos. Con la disciplina que hace que las
> respuestas valgan algo — hipótesis congeladas antes de correr, mismo presupuesto de
> ajuste para los dos lados, los resultados negativos con el mismo peso que los
> positivos, y ninguna cifra escrita a mano: todas salen del código."

---

## 13. *Proposed next step*

**Qué decir:**
> "La propuesta es una sola: repetir el análisis de instrumentación sobre **nuestros**
> modelos de motor y sobre las configuraciones de sensores que estemos considerando.
>
> No necesita IA. No necesita recoger datos nuevos. Y responde a una pregunta que ya
> estamos pagando por responder de otra manera."

**Y cállate.** Deja que hable él.

---

## 14–15. Backup

No las pases salvo que pregunte. Están para:
- **Disciplina de método** — si te pregunta "¿y cómo sé que no te has engañado?"
- **El muro es instantáneo** — si te pregunta "¿entonces no se puede diagnosticar nada?"

---

# Objeciones probables, y qué contestar

**"Esto es todo simulado, no me sirve."**
> "Correcto, y por eso no defiendo ningún número absoluto. Lo que se transfiere es la
> geometría: cuántas incógnitas hay contra cuántas medidas. Eso es igual en nuestro
> motor real, y es lo que manda. Además, el análisis de instrumentación corre sobre
> nuestros modelos, donde es más fuerte que aquí."

**"La IA es humo."**
> "En tres de las cuatro tareas que medí, te doy la razón con números. En la cuarta hay
> un 30 % de mejora, que es real y es pequeño. Lo que te propongo no depende de que la
> IA funcione."

**"¿Cuánto ha costado esto?"**
> "Corre entero en un portátil en minutos. Y todo viene de datos públicos de
> certificación, así que no hay problema de propiedad intelectual para circularlo
> internamente."

**"¿Por qué no lo ha hecho ya el fabricante de sensores / un proveedor?"**
> "Porque hace falta conocer la verdad del motor para validarlo, y eso solo lo tienes
> en simulación. Un estudio con datos reales no puede comprobar si su propia estimación
> de incertidumbre es honesta. Nosotros sí."

**"¿Y si el proveedor X dice que su modelo da un 95 %?"**
> "Primera pregunta: ¿contra qué línea base, y estaba ajustada igual? Segunda: ¿cuál es
> la dispersión entre repeticiones? Con esas dos preguntas se cae la mayoría."

---

# Si solo tienes cinco minutos

Diapositivas **4, 6, 7, 9, 13**. En ese orden.

Un paper publicado que no se sostiene → mis propios números corregidos → dónde la IA
pierde y por qué → lo que sí sirve, que es instrumentación → la propuesta.
