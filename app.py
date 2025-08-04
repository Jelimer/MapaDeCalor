import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard de Activos",
    page_icon="📈",
    layout="wide"
)

# --- Estilo CSS ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A4E69;
    }
    .news-item {
        border-bottom: 1px solid #444;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- FUNCIONES DE CACHÉ Y PROCESAMIENTO ---

@st.cache_data(ttl="1h")
def cargar_datos_historicos(ticker):
    try:
        return yf.Ticker(ticker).history(period="max", auto_adjust=True)
    except Exception:
        return None

@st.cache_data(ttl="6h")
def cargar_info_ticker(ticker_str):
    try:
        ticker_obj = yf.Ticker(ticker_str)
        info = ticker_obj.info
        if info and 'longName' in info:
            # Precargar datos financieros y noticias para evitar múltiples llamadas
            info['_financials'] = ticker_obj.financials
            info['_quarterly_financials'] = ticker_obj.quarterly_financials
            info['_balance_sheet'] = ticker_obj.balance_sheet
            info['_quarterly_balance_sheet'] = ticker_obj.quarterly_balance_sheet
            info['_cashflow'] = ticker_obj.cashflow
            info['_quarterly_cashflow'] = ticker_obj.quarterly_cashflow
            info['_news'] = ticker_obj.news
            return info
        return None
    except Exception:
        return None

# --- FUNCIONES DE VISUALIZACIÓN ---

def plot_binary_heatmap(ax, data, annot=True, fmt='.2%', show_yticklabels=False):
    annot_kws = {"fontsize": 9, "fontweight": "bold", "color": "white"}
    sns.heatmap(data, mask=(data >= 0), cmap=['#d9534f'], annot=annot, fmt=fmt, linewidths=0.5, linecolor='black', cbar=False, ax=ax, yticklabels=show_yticklabels, annot_kws=annot_kws)
    sns.heatmap(data, mask=(data < 0), cmap=['#5cb85c'], annot=annot, fmt=fmt, linewidths=0.5, linecolor='black', cbar=False, ax=ax, yticklabels=show_yticklabels, annot_kws=annot_kws)
    if show_yticklabels:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, style='italic', fontsize=10)

def generar_mapa_calor(datos, anios_a_mostrar):
    precios_ajustados = datos['Close']
    rendimientos_mensuales = precios_ajustados.resample('ME').last().pct_change()
    df_rendimientos = rendimientos_mensuales.to_frame(name='rendimiento').dropna()
    if df_rendimientos.empty: return None, None
    df_rendimientos['Año'] = df_rendimientos.index.year
    df_rendimientos['Mes'] = df_rendimientos.index.month
    mapa_calor_mensual_full = df_rendimientos.pivot_table(values='rendimiento', index='Año', columns='Mes')
    anios_disponibles = sorted([i for i in mapa_calor_mensual_full.index if isinstance(i, int)], reverse=True)
    if not anios_disponibles: return None, None
    if anios_a_mostrar > len(anios_disponibles): anios_a_mostrar = len(anios_disponibles)
    anios_seleccionados = anios_disponibles[:anios_a_mostrar]
    mapa_calor_mensual = mapa_calor_mensual_full.loc[sorted(anios_seleccionados)]
    
    # --- Cálculo de Métricas ---
    # Extraer la primera y última fecha del índice del dataframe de rendimientos filtrado por los años seleccionados
    fecha_inicio_periodo = df_rendimientos[df_rendimientos['Año'].isin(anios_seleccionados)].index.min().strftime('%d-%m-%Y')
    fecha_fin_periodo = df_rendimientos[df_rendimientos['Año'].isin(anios_seleccionados)].index.max().strftime('%d-%m-%Y')
    
    rendimiento_total = (mapa_calor_mensual + 1).prod().prod() - 1
    volatilidad_anual = mapa_calor_mensual.stack().std() * np.sqrt(12)
    mejor_mes_val = mapa_calor_mensual.stack().max()
    peor_mes_val = mapa_calor_mensual.stack().min()
    mejor_mes_idx = mapa_calor_mensual.stack().idxmax()
    peor_mes_idx = mapa_calor_mensual.stack().idxmin()
    
    metricas = {
        "fecha_inicio": fecha_inicio_periodo,
        "fecha_fin": fecha_fin_periodo,
        "rendimiento_total": rendimiento_total,
        "volatilidad_anual": volatilidad_anual,
        "mejor_mes": f"{mejor_mes_val:.2%} ({pd.to_datetime(str(mejor_mes_idx[1]), format='%m').strftime('%b')} {mejor_mes_idx[0]})",
        "peor_mes": f"{peor_mes_val:.2%} ({pd.to_datetime(str(peor_mes_idx[1]), format='%m').strftime('%b')} {peor_mes_idx[0]})"
    }
    # --- Generación del Gráfico ---
    rendimiento_anual = (mapa_calor_mensual + 1).prod(axis=1) - 1
    mensual_anios_df = mapa_calor_mensual.copy()
    mensual_anios_df.columns = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    anual_anios_df = rendimiento_anual.to_frame(name='Anual')
    mensual_metricas_df = pd.DataFrame({'Promedio': mensual_anios_df.mean(), 'Mediana': mensual_anios_df.median()}).T
    anual_metricas_df = pd.DataFrame({'Promedio': {'Anual': anual_anios_df['Anual'].mean()},'Mediana': {'Anual': anual_anios_df['Anual'].median()}}).T

    # Ajustamos el tamaño de la figura para dar espacio al texto de las métricas abajo
    fig_height = max(5, len(mensual_anios_df) * 0.55)
    fig, axes = plt.subplots(
        2, 2, figsize=(16, fig_height + 2), # +2 para el espacio extra
        gridspec_kw={
            'height_ratios': [len(mensual_anios_df), 2], 'hspace': 0.08,
            'width_ratios': [12, 1.2], 'wspace': 0.05,
            'bottom': 0.2 # Ajustar el fondo para que quepa el texto
        }
    )
    fig.patch.set_facecolor('#1e1e1e') # Color de fondo para la figura completa

    ax_tl, ax_tr, ax_bl, ax_br = axes.flatten()

    sns.heatmap(mensual_anios_df, annot=True, fmt='.2%', cmap='RdYlGn', center=0, linewidths=0.5, linecolor='black', cbar=False, ax=ax_tl, annot_kws={"fontsize": 9})
    ax_tl.set_ylabel('Año', fontsize=11); ax_tl.tick_params(axis='y', labelsize=9, colors='white'); ax_tl.xaxis.tick_top(); ax_tl.xaxis.set_label_position('top'); ax_tl.set_xlabel(''); ax_tl.set_xticks(np.arange(len(mensual_anios_df.columns)) + 0.5); ax_tl.set_xticklabels(mensual_anios_df.columns, fontsize=10, rotation=0, ha="center", color='white'); ax_tl.tick_params(axis='x', length=0)
    
    plot_binary_heatmap(ax=ax_tr, data=anual_anios_df, show_yticklabels=False); ax_tr.xaxis.tick_top(); ax_tr.xaxis.set_label_position('top'); ax_tr.set_xlabel(''); ax_tr.tick_params(axis='x', length=0, colors='white'); ax_tr.set_ylabel(''); ax_tr.set_yticklabels([]); ax_tr.set_ylim(ax_tl.get_ylim())
    
    plot_binary_heatmap(ax=ax_bl, data=mensual_metricas_df, show_yticklabels=True); ax_bl.set_ylabel('Métricas', fontsize=11, color='white'); ax_bl.set_xlabel(''); ax_bl.tick_params(axis='x', bottom=False, labelbottom=False, colors='white'); ax_bl.tick_params(axis='y', colors='white')
    
    plot_binary_heatmap(ax=ax_br, data=anual_metricas_df, show_yticklabels=False); ax_br.set_xlabel(''); ax_br.set_ylabel(''); ax_br.set_yticklabels([]); ax_br.tick_params(axis='x', bottom=False, labelbottom=False); ax_br.set_ylim(ax_bl.get_ylim())
    
    mappable = ax_tl.collections[0]; cbar = fig.colorbar(mappable, ax=axes, shrink=0.8, pad=0.02, aspect=30); cbar.set_label('Rendimiento Mensual', rotation=270, labelpad=18, fontsize=10, color='white'); cbar.ax.tick_params(labelsize=8, colors='white')
    
    fig.suptitle(f'Mapa de Calor de Rendimientos para {st.session_state.ticker.upper()}', fontsize=16, y=0.95, fontweight='bold', color='white')

    # --- Añadir el texto de las métricas a la figura ---
    stats_text = (
        f"Estadísticas Clave del Período\n"
        f"------------------------------------\n"
        f"Fecha de Inicio: {metricas['fecha_inicio']}   |   "
        f"Fecha Final: {metricas['fecha_fin']}\n"
        f"Rendimiento Total: {metricas['rendimiento_total']:.2%}   |   "
        f"Volatilidad Anualizada: {metricas['volatilidad_anual']:.2%}\n"
        f"Mejor Mes: {metricas['mejor_mes']}   |   "
        f"Peor Mes: {metricas['peor_mes']}"
    )
    fig.text(0.5, 0.05, stats_text, ha='center', va='bottom', fontsize=10, color='white', bbox=dict(boxstyle="round,pad=0.5", fc="#292929", ec="#444", lw=1))

    plt.close(fig)
    return fig, metricas

def crear_grafico_precios(datos, ticker, sma_short, sma_long):
    """Crea un gráfico de velas (candlestick) y volumen con Plotly, incluyendo Medias Móviles Simples."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Gráfico de Velas
    fig.add_trace(go.Candlestick(x=datos.index,
                                   open=datos['Open'],
                                   high=datos['High'],
                                   low=datos['Low'],
                                   close=datos['Close'],
                                   name='Precio'), row=1, col=1)

    # Medias Móviles
    if sma_short > 0:
        datos[f'SMA{sma_short}'] = datos['Close'].rolling(window=sma_short).mean()
        fig.add_trace(go.Scatter(x=datos.index, y=datos[f'SMA{sma_short}'], name=f'SMA {sma_short}',
                                 line=dict(color='#fca311', width=1.5, dash='dot')), row=1, col=1)
    if sma_long > 0:
        datos[f'SMA{sma_long}'] = datos['Close'].rolling(window=sma_long).mean()
        fig.add_trace(go.Scatter(x=datos.index, y=datos[f'SMA{sma_long}'], name=f'SMA {sma_long}',
                                 line=dict(color='#e5383b', width=1.5, dash='dot')), row=1, col=1)

    # Gráfico de Volumen
    fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen',
                         marker_color='#4a4e69'), row=2, col=1)

    fig.update_layout(
        title_text=f'Precio Histórico y Volumen de {ticker.upper()}',
        title_x=0.5,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Precio", row=1, col=1); fig.update_yaxes(title_text="Volumen", row=2, col=1)
    
    return fig

# --- INTERFAZ DE USUARIO ---
st.sidebar.header("⚙️ Controles")
st.session_state.ticker = st.sidebar.text_input("Ticker (ej: GGAL, AAPL, MELI.BA)", value="GGAL").upper()

info_ticker = cargar_info_ticker(st.session_state.ticker)
nombre_empresa = info_ticker.get('longName', st.session_state.ticker) if info_ticker else st.session_state.ticker
st.title(f"📊 Dashboard de Activo: {nombre_empresa}")

if st.session_state.ticker:
    datos_historicos = cargar_datos_historicos(st.session_state.ticker)
    if datos_historicos is not None and info_ticker is not None:
        tab_rend, tab_info, tab_fund, tab_finan = st.tabs(["📈 Rendimientos", "ℹ️ Info & Precios", "🏦 Fundamental", "📰 Finanzas & Noticias"])

        with tab_rend:
            st.subheader("Análisis de Rentabilidad")
            anios_slider = st.slider("Años a Mostrar", 1, 30, 10, 1, key="anios_heatmap")
            fig_heatmap, metricas = generar_mapa_calor(datos_historicos, anios_slider)
            if fig_heatmap and metricas:
                st.subheader("Estadísticas Clave del Período")
                c1, c2, c3 = st.columns(3)
                c1.metric("Fecha de Inicio", metricas['fecha_inicio'])
                c2.metric("Fecha Final", metricas['fecha_fin'])
                c3.metric("Rendimiento Total", f"{metricas['rendimiento_total']:.2%}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Volatilidad Anualizada", f"{metricas['volatilidad_anual']:.2%}")
                c2.metric("Mejor Mes", metricas['mejor_mes'])
                c3.metric("Peor Mes", metricas['peor_mes'])

                st.pyplot(fig_heatmap, use_container_width=True)
                buf = BytesIO()
                fig_heatmap.savefig(buf, format="png", dpi=300, bbox_inches='tight', facecolor=fig_heatmap.get_facecolor())
                st.sidebar.download_button("📥 Descargar Mapa de Calor", buf.getvalue(), f"mapa_calor_{st.session_state.ticker}_{anios_slider}a.png", "image/png")

        with tab_info:
            st.subheader("Información de la Compañía")
            c1, c2 = st.columns(2)
            c1.metric("Sector", info_ticker.get('sector', 'N/A')); c2.metric("Industria", info_ticker.get('industry', 'N/A'))
            st.markdown(f"**Resumen:** {info_ticker.get('longBusinessSummary', 'No hay resumen disponible.')}")
            st.markdown("---")
            st.subheader("Gráfico de Precios y Volumen")
            c1, c2 = st.columns(2)
            sma_short = c1.number_input("Media Móvil Corta (días)", 0, 100, 50, 1)
            sma_long = c2.number_input("Media Móvil Larga (días)", 0, 200, 0, 1)
            fig_precios = crear_grafico_precios(datos_historicos, st.session_state.ticker, sma_short, sma_long)
            st.plotly_chart(fig_precios, use_container_width=True)

        with tab_fund:
            st.subheader("Ratios y Datos Fundamentales")
            def format_value(v, t='n'):
                if v is None or v == 'N/A': return "N/A"
                if t == 'p': return f"{v:.2%}"
                if t == 'm': return f"${v:,.0f}"
                return f"{v:,}"
            c1, c2, c3 = st.columns(3)
            c1.metric("Capitalización de Mercado", format_value(info_ticker.get('marketCap'), 'm'))
            c2.metric("Ratio Precio/Beneficio (P/E)", format_value(info_ticker.get('trailingPE')))
            c3.metric("Ratio Precio/Ventas (P/S)", format_value(info_ticker.get('priceToSalesTrailing12Months')))
            c1, c2, c3 = st.columns(3)
            c1.metric("Ratio Precio/Libro (P/B)", format_value(info_ticker.get('priceToBook')))
            c2.metric("Beneficio por Acción (EPS)", format_value(info_ticker.get('trailingEps')))
            c3.metric("Rentabilidad por Dividendo", format_value(info_ticker.get('dividendYield'), 'p'))
            st.markdown("---")
            st.write("**Nota:** La disponibilidad de los datos puede variar.")

        with tab_finan:
            st.subheader("Estados Financieros")
            freq = st.radio("Frecuencia", ["Anual", "Trimestral"], horizontal=True)
            financial_statement = st.selectbox("Seleccionar Estado Financiero", ["Estado de Resultados", "Balance General", "Flujo de Caja"])
            if financial_statement == "Estado de Resultados":
                data = info_ticker['_financials'] if freq == "Anual" else info_ticker['_quarterly_financials']
            elif financial_statement == "Balance General":
                data = info_ticker['_balance_sheet'] if freq == "Anual" else info_ticker['_quarterly_balance_sheet']
            else:
                data = info_ticker['_cashflow'] if freq == "Anual" else info_ticker['_quarterly_cashflow']
            st.dataframe(data.style.format("{:,.0f}"), use_container_width=True)
            st.markdown("---")
            st.subheader("Últimas Noticias")
            news = info_ticker.get('_news', [])
            if news:
                for item in news:
                    title = item.get('title', 'No hay título')
                    link = item.get('link')
                    publisher = item.get('publisher', 'N/A')
                    publish_time = pd.to_datetime(item.get('providerPublishTime'), unit='s').strftime('%Y-%m-%d') if item.get('providerPublishTime') else 'N/A'
                    
                    if link:
                        st.markdown(f"<div class='news-item'><b><a href='{link}' target='_blank'>{title}</a></b><br><small>{publisher} - {publish_time}</small></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='news-item'><b>{title}</b><br><small>{publisher} - {publish_time}</small></div>", unsafe_allow_html=True)
            else:
                st.info("No se encontraron noticias recientes para este activo.")

    else:
        st.error(f"No se pudieron cargar datos para '{st.session_state.ticker}'. Verifica el ticker.")
else:
    st.info("Introduce un ticker en la barra lateral para comenzar.")

st.sidebar.markdown("---")
st.sidebar.info("Aplicación creada por Gemini.")