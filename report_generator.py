import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN DE ESTILO ---
COLOR_PRIMARY = (23, 43, 77)      # Azul Oscuro
COLOR_SECONDARY = (0, 190, 255)   # Azul Cian
FONT_FAMILY = 'Helvetica'

class ProfessionalPadelReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=False) 

    def header(self):
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, 210, 20, 'F')
        self.set_font(FONT_FAMILY, 'B', 12)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 5)
        self.cell(0, 10, 'PADEL ANALISIS TFG - INFORME TÉCNICO', ln=True, align='L')

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_FAMILY, 'I', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Creado por PADEL ANALISIS TFG Luis Miguel Sanz Fernandez', align='C')

    def draw_cover_page(self, video_name="Análisis de Partido"):
        self.add_page()
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, 210, 297, 'F')
        
        self.set_y(110)
        self.set_font(FONT_FAMILY, 'B', 30)
        self.set_text_color(255, 255, 255)
        self.multi_cell(0, 15, f"INFORME DE RENDIMIENTO\n{video_name}", align='C')
        
        self.set_y(150)
        self.set_font(FONT_FAMILY, '', 14)
        date_str = datetime.now().strftime("%d / %m / %Y")
        self.cell(0, 10, f"Fecha: {date_str}", align='C', ln=True)
        
        self.set_y(260)
        self.set_font(FONT_FAMILY, 'B', 10)
        self.cell(0, 10, "TFG - Luis Miguel Sanz Fernandez", align='C')

# --- GRÁFICOS MATPLOTLIB ---

def draw_court_lines(ax):
    """
    Dibuja la pista centrada.
    """
    line_color = 'black' 
    line_width = 1.5
    z = 10 
    
    # Contorno (-5 a 5 en X, -10 a 10 en Y)
    ax.plot([-5, 5], [-10, -10], color=line_color, lw=line_width, zorder=z)
    ax.plot([-5, 5], [10, 10], color=line_color, lw=line_width, zorder=z)
    ax.plot([-5, -5], [-10, 10], color=line_color, lw=line_width, zorder=z)
    ax.plot([5, 5], [-10, 10], color=line_color, lw=line_width, zorder=z)
    
    # Red (en Y=0)
    ax.plot([-5, 5], [0, 0], color='black', lw=3, linestyle='-', zorder=z) 
    
    # Líneas de saque (+/- 7m)
    ax.plot([-5, 5], [-7, -7], color=line_color, lw=line_width, zorder=z)
    ax.plot([-5, 5], [7, 7], color=line_color, lw=line_width, zorder=z)
    
    # Línea central
    ax.plot([0, 0], [-10, -7], color=line_color, lw=line_width, zorder=z)
    ax.plot([0, 0], [7, 10], color=line_color, lw=line_width, zorder=z)

def generate_player_heatmap(df, player_id, filename):
    fig, ax = plt.subplots(figsize=(4, 8))
    
    # Fondo Claro
    ax.set_facecolor('#F8F9FA') 
    
    # Límites
    ax.set_xlim(-6, 6)
    ax.set_ylim(-11, 11)
    
    # --- CORRECCIÓN CLAVE: INVERTIR EJE Y ---
    # Esto hace que los valores positivos (Jugadores 1 y 2) se pinten ABAJO
    # y los negativos (Jugadores 3 y 4) se pinten ARRIBA.
    ax.invert_yaxis() 
    # ----------------------------------------
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    col_x = f"player{player_id}_x"
    col_y = f"player{player_id}_y"
    
    if col_x in df.columns:
        # Filtrar datos dentro de pista
        data = df[(df[col_x] >= -6) & (df[col_x] <= 6) & (df[col_y] >= -11) & (df[col_y] <= 11)]
        
        if len(data) > 10:
            sns.kdeplot(
                x=data[col_x], 
                y=data[col_y], 
                fill=True, 
                cmap="YlOrRd", # Mapa de calor Rojo/Naranja
                alpha=0.75,    
                thresh=0.1, 
                levels=10,
                ax=ax,
                zorder=1
            )
            # Puntos para referencia visual
            ax.scatter(data[col_x], data[col_y], color='black', s=2, alpha=0.05, zorder=2)
    
    draw_court_lines(ax)
    
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_comparison_chart(stats_data, filename):
    players = [f"J{d['id']}" for d in stats_data]
    values = [d['distancia'] for d in stats_data]
    
    plt.figure(figsize=(8, 3))
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
    
    bars = plt.barh(players, values, color=colors)
    plt.title("Distancia Recorrida (Metros)", fontweight='bold')
    plt.xlabel("Metros")
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}m', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def generate_timeline(df, filename):
    plt.figure(figsize=(10, 3))
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
    for i in range(1, 5):
        col = f'player{i}_Vnorm4'
        if col in df.columns:
            y = df[col].abs().rolling(window=45, min_periods=1).mean() * 3.6
            plt.plot(df['time'], y, label=f'J{i}', color=colors[i-1], lw=1.5)
            
    plt.legend(loc='upper right', fontsize='small')
    plt.title("Evolución de Intensidad (Velocidad km/h)", fontweight='bold')
    plt.ylabel("km/h")
    plt.xlabel("Tiempo (s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# --- GENERADOR DEL REPORTE ---

def create_full_report(csv_path, output_pdf="Informe_Partido.pdf"):
    try:
        df = pd.read_csv(csv_path)
    except:
        return

    # Preparar Datos
    stats = []
    for i in range(1, 5):
        d_col = f'player{i}_distance'
        v_col = f'player{i}_Vnorm4'
        
        dist = df[d_col].sum() if d_col in df.columns else 0
        v_max = (df[v_col].abs().max() * 3.6) if v_col in df.columns else 0
        v_avg = (df[v_col].abs().mean() * 3.6) if v_col in df.columns else 0
        
        stats.append({'id': i, 'distancia': dist, 'v_max': v_max, 'v_avg': v_avg})

    # Generar Imágenes
    generate_comparison_chart(stats, "chart_bars.png")
    generate_timeline(df, "chart_time.png")
    
    heat_imgs = []
    for i in range(1, 5):
        fname = f"heat_p{i}.png"
        generate_player_heatmap(df, i, fname)
        heat_imgs.append(fname)

    # --- PDF ---
    pdf = ProfessionalPadelReport()
    
    # P1: PORTADA
    pdf.draw_cover_page()
    
    # P2: DATOS
    pdf.add_page()
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, 'B', 16)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.cell(0, 10, "1. Análisis Físico y Métricas", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Tabla
    pdf.set_font(FONT_FAMILY, 'B', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(240, 240, 240)
    
    cols = ["Jugador", "Distancia (m)", "Vel. Máx (km/h)", "Vel. Media (km/h)"]
    widths = [30, 40, 45, 45]
    x_start = (210 - sum(widths)) / 2
    
    pdf.set_x(x_start)
    for w, c in zip(widths, cols):
        pdf.cell(w, 8, c, border=1, fill=True, align='C')
    pdf.ln()
    
    pdf.set_font(FONT_FAMILY, '', 10)
    for s in stats:
        pdf.set_x(x_start)
        pdf.cell(widths[0], 8, f"Jugador {s['id']}", border=1, align='C')
        pdf.cell(widths[1], 8, f"{s['distancia']:.1f}", border=1, align='C')
        pdf.cell(widths[2], 8, f"{s['v_max']:.1f}", border=1, align='C')
        pdf.cell(widths[3], 8, f"{s['v_avg']:.1f}", border=1, align='C')
        pdf.ln()
    
    pdf.ln(10)
    pdf.image("chart_bars.png", x=20, w=170)
    pdf.ln(5)
    pdf.image("chart_time.png", x=15, w=180)
    
    # P3: HEATMAPS (GRID FIJO)
    pdf.add_page()
    pdf.ln(15)
    pdf.set_font(FONT_FAMILY, 'B', 16)
    pdf.set_text_color(*COLOR_PRIMARY)
    pdf.cell(0, 10, "2. Ocupación de Pista (Mapas de Calor)", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    
    # GRID 2x2
    # Hemos invertido el eje Y, así que visualmente en el PDF:
    # J1 y J2 (Abajo en video) -> Saldrán abajo en su gráfica individual
    # J3 y J4 (Arriba en video) -> Saldrán arriba en su gráfica individual
    
    y_row1 = 50
    y_row2 = 160
    
    # Fila 1: Jugadores del fondo (en teoría J3/J4 si estuvieran ordenados por posición)
    # Pero mantenemos orden numérico: J1, J2, J3, J4
    
    # J1 (Abajo Izquierda)
    pdf.set_xy(30, y_row1)
    pdf.set_font(FONT_FAMILY, 'B', 12)
    pdf.cell(60, 10, "Jugador 1", align='C')
    pdf.image(heat_imgs[0], x=35, y=y_row1+10, w=50)
    
    # J2 (Abajo Derecha)
    pdf.set_xy(120, y_row1)
    pdf.cell(60, 10, "Jugador 2", align='C')
    pdf.image(heat_imgs[1], x=125, y=y_row1+10, w=50)
    
    # J3 (Arriba Izquierda)
    pdf.set_xy(30, y_row2)
    pdf.cell(60, 10, "Jugador 3", align='C')
    pdf.image(heat_imgs[2], x=35, y=y_row2+10, w=50)
    
    # J4 (Arriba Derecha)
    pdf.set_xy(120, y_row2)
    pdf.cell(60, 10, "Jugador 4", align='C')
    pdf.image(heat_imgs[3], x=125, y=y_row2+10, w=50)
    
    pdf.output(output_pdf)
    
    for f in ["chart_bars.png", "chart_time.png"] + heat_imgs:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    create_full_report("padel_analytics_report.csv")
