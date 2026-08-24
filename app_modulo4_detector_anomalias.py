import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Detector de Anomalías — Módulo 4",
    layout="wide"
)

st.title("🚨 Detector de Anomalías: Lógica + Big-O + NumPy")
st.caption(
    "Módulo 4 — Matemáticas Discretas y Complejidad. "
    "La misma decisión lógica, evaluada de forma ingenua vs. vectorizada."
)

tab1, tab2, tab3 = st.tabs(
    ["🔎 Simulación de alarma", "📈 Notación Big-O", "⚡ Benchmark en vivo"]
)


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------
def generar_datos(n, seed=42):
    rng = np.random.default_rng(seed)

    temperaturas = rng.uniform(15, 40, n)
    humedades = rng.uniform(20, 80, n)
    velocidades_viento = rng.uniform(0, 100, n)

    return temperaturas, humedades, velocidades_viento


# ---------------------------------------------------------------------------
# Versión con loop
#
# Condición:
# temperatura > umbral
# AND
# humedad < umbral
# AND
# velocidad del viento > umbral
# ---------------------------------------------------------------------------
def alarma_logica_loop(
    temperaturas,
    humedades,
    velocidades_viento,
    temp_umbral,
    hum_umbral,
    viento_umbral
):
    resultados = []

    for temp, hum, viento in zip(
        temperaturas,
        humedades,
        velocidades_viento
    ):
        resultados.append(
            temp > temp_umbral
            and hum < hum_umbral
            and viento > viento_umbral
        )

    return np.array(resultados)


# ---------------------------------------------------------------------------
# Versión vectorizada con NumPy
# ---------------------------------------------------------------------------
def alarma_logica_vectorizada(
    temperaturas,
    humedades,
    velocidades_viento,
    temp_umbral,
    hum_umbral,
    viento_umbral
):
    return (
        (temperaturas > temp_umbral)
        & (humedades < hum_umbral)
        & (velocidades_viento > viento_umbral)
    )


# ---------------------------------------------------------------------------
# Tab 1: Simulación de alarma
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Alarma por regla lógica")

    st.write(
        "La alarma dispara cuando se cumplen simultáneamente tres "
        "proposiciones conectadas mediante un **AND lógico**:"
    )

    st.markdown(
        """
        **P1:** Temperatura > umbral  
        **P2:** Humedad < umbral  
        **P3:** Velocidad del viento > umbral  

        **Alarma = P1 AND P2 AND P3**
        """
    )

    col_cfg, col_data = st.columns([1, 2])

    with col_cfg:
        n = st.slider(
            "Número de lecturas (n)",
            50,
            5000,
            500,
            step=50
        )

        temp_umbral = st.slider(
            "Umbral temperatura (°C) — mayor que",
            15,
            40,
            30
        )

        hum_umbral = st.slider(
            "Umbral humedad (%) — menor que",
            20,
            80,
            40
        )

        viento_umbral = st.slider(
            "Umbral velocidad del viento (km/h) — mayor que",
            0,
            100,
            60
        )

    temps, hums, velocidades_viento = generar_datos(n)

    with col_cfg:
        alarmas = alarma_logica_vectorizada(
            temps,
            hums,
            velocidades_viento,
            temp_umbral,
            hum_umbral,
            viento_umbral
        )

        st.metric(
            "Alarmas detectadas",
            f"{alarmas.sum()} / {n}"
        )

    with col_data:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        ax.scatter(
            temps[~alarmas],
            hums[~alarmas],
            c="steelblue",
            alpha=0.5,
            label="Normal",
            s=15,
        )

        ax.scatter(
            temps[alarmas],
            hums[alarmas],
            c="crimson",
            alpha=0.8,
            label="Alarma / anomalía",
            s=25,
        )

        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel("Humedad (%)")
        ax.set_title("Temperatura vs. Humedad")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    with st.expander("Ver datos y lógica aplicada"):
        df = pd.DataFrame({
            "temperatura": temps.round(2),
            "humedad": hums.round(2),
            "velocidad_viento_kmh": velocidades_viento.round(2),
            "alarma": alarmas,
        })

        st.dataframe(
            df,
            use_container_width=True,
            height=250
        )


# ---------------------------------------------------------------------------
# Tab 2: Notación Big-O
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("¿Por qué importa la complejidad?")

    st.write(
        "El detector recorre las n lecturas una vez. Aunque ahora evalúa "
        "tres proposiciones por lectura, sigue siendo un algoritmo **O(n)**. "
        "Aquí puedes observar cómo crece el número de operaciones frente "
        "a otras complejidades comunes."
    )

    n_max = st.slider(
        "Tamaño máximo de n para la gráfica",
        10,
        200,
        50
    )

    n_valores = np.arange(1, n_max + 1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.plot(
        n_valores,
        np.ones_like(n_valores),
        label="O(1) — constante"
    )

    ax2.plot(
        n_valores,
        n_valores,
        label="O(n) — lineal (nuestro detector)"
    )

    ax2.plot(
        n_valores,
        n_valores * np.log2(np.maximum(n_valores, 2)),
        label="O(n log n)"
    )

    ax2.plot(
        n_valores,
        n_valores ** 2,
        label="O(n²) — cuadrática"
    )

    ax2.set_xlabel("Tamaño de los datos (n)")
    ax2.set_ylabel("Operaciones (teórico)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

    st.info(
        "Nuestro detector de alarmas es O(n) tanto con loop como con NumPy. "
        "Agregar una tercera proposición no cambia la notación Big-O porque "
        "cada lectura sigue evaluándose una sola vez. Lo que cambia es la "
        "cantidad de operaciones constantes realizadas en cada lectura."
    )


# ---------------------------------------------------------------------------
# Tab 3: Benchmark en vivo
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(
        "Loop vs. NumPy: misma lógica, distinta velocidad real"
    )

    st.write(
        "Ejecuta la misma condición lógica de tres proposiciones sobre "
        "datos sintéticos, una vez con un loop de Python puro y otra vez "
        "vectorizada con NumPy, y compara el tiempo real."
    )

    n_bench = st.select_slider(
        "Tamaño de datos para el benchmark",
        options=[
            1_000,
            10_000,
            100_000,
            500_000,
            1_000_000
        ],
        value=100_000,
    )

    temp_umbral_b = st.slider(
        "Umbral temperatura (°C)",
        15,
        40,
        30,
        key="temp_bench"
    )

    hum_umbral_b = st.slider(
        "Umbral humedad (%)",
        20,
        80,
        40,
        key="hum_bench"
    )

    viento_umbral_b = st.slider(
        "Umbral velocidad del viento (km/h)",
        0,
        100,
        60,
        key="viento_bench"
    )

    if st.button(
        "▶️ Ejecutar benchmark",
        type="primary"
    ):
        temps_b, hums_b, velocidades_viento_b = generar_datos(
            n_bench
        )

        # Número de repeticiones
        repeticiones_loop = 1
        repeticiones_vec = 20

        # ---------------------------------------------------------------
        # Benchmark del loop
        # ---------------------------------------------------------------
        inicio = time.perf_counter()

        for _ in range(repeticiones_loop):
            alarma_logica_loop(
                temps_b,
                hums_b,
                velocidades_viento_b,
                temp_umbral_b,
                hum_umbral_b,
                viento_umbral_b
            )

        t_loop = (
            time.perf_counter() - inicio
        ) / repeticiones_loop

        # ---------------------------------------------------------------
        # Benchmark de NumPy
        # ---------------------------------------------------------------
        inicio = time.perf_counter()

        for _ in range(repeticiones_vec):
            alarma_logica_vectorizada(
                temps_b,
                hums_b,
                velocidades_viento_b,
                temp_umbral_b,
                hum_umbral_b,
                viento_umbral_b
            )

        t_vec = (
            time.perf_counter() - inicio
        ) / repeticiones_vec

        # ---------------------------------------------------------------
        # Resultados
        # ---------------------------------------------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Tiempo con loop",
            f"{t_loop * 1000:.3f} ms"
        )

        col2.metric(
            "Tiempo con NumPy",
            f"{t_vec * 1000:.3f} ms"
        )

        if t_vec > 0:
            speedup = t_loop / t_vec

            col3.metric(
                "NumPy es más rápido por",
                f"{speedup:,.0f}x"
            )
        else:
            col3.metric(
                "NumPy es más rápido por",
                "Demasiado rápido para medir"
            )

        st.caption(
            f"Tiempo con loop promediado sobre "
            f"{repeticiones_loop} corrida(s); "
            f"tiempo con NumPy promediado sobre "
            f"{repeticiones_vec} corridas, "
            "para reducir el ruido de la medición."
        )

        # ---------------------------------------------------------------
        # Gráfica del benchmark
        # ---------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))

        ax3.bar(
            ["Loop (Python)", "NumPy (vectorizado)"],
            [t_loop * 1000, t_vec * 1000],
            color=["indianred", "seagreen"]
        )

        ax3.set_ylabel("Tiempo (milisegundos)")
        ax3.set_title("Comparación de rendimiento")
        ax3.grid(
            alpha=0.3,
            axis="y"
        )

        st.pyplot(fig3)

        # ---------------------------------------------------------------
        # Verificación de resultados
        # ---------------------------------------------------------------
        resultado_loop = alarma_logica_loop(
            temps_b,
            hums_b,
            velocidades_viento_b,
            temp_umbral_b,
            hum_umbral_b,
            viento_umbral_b
        )

        resultado_vectorizado = alarma_logica_vectorizada(
            temps_b,
            hums_b,
            velocidades_viento_b,
            temp_umbral_b,
            hum_umbral_b,
            viento_umbral_b
        )

        if np.array_equal(
            resultado_loop,
            resultado_vectorizado
        ):
            st.success(
                "✓ Ambas versiones producen exactamente el mismo resultado."
            )
        else:
            st.error(
                "✗ Las versiones producen resultados diferentes."
            )

    else:
        st.caption(
            "Ajusta los parámetros y presiona "
            "**Ejecutar benchmark** para ver el resultado."
        )
