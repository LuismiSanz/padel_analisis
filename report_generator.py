import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

class PadelReport(FPDF):
    def header(self):
        # Cabecera con Logo y Título en todas las páginas
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Informe de Rendimiento - Padel Analysis', border=False, ln=True, align='C')
        self.ln(5)

    def footer(self):
        # Pie de página con número
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

def draw_padel_court(ax):
    # Dibuja las líneas de la pista de pádel (20x10 metros)
    # Fondo azul típico de WPT
    ax.set_facecolor('#4A90E2')
    
    # Líneas blancas
    line_color = 'white'
    line_width = 2
    
    # Contorno
    ax.plot([0, 10], [0, 0], color=line_color, lw=line_width) # Fondo abajo
    ax.plot([0, 10], [20, 20], color=line_color, lw=line_width) # Fondo arriba
    ax.plot([0, 0], [0, 20], color=line_color, lw=line_width) # Lateral Izq
    ax.plot([10, 10], [0, 20], color=line_color, lw=line_width) # Lateral Der
    
    # Red (Mitad)
    ax.plot([0, 10], [10, 10], color='white', lw=4, linestyle='-') # Red
    
    # Líneas de saque
    ax.plot([0, 10], [3, 3], color=line_color, lw=line_width) # Línea saque abajo
    ax.plot([0, 10], [17, 17], color=line_color, lw=line_width) # Línea saque arriba
    ax.plot([5, 5], [3, 17], color=line_color, lw=line_width) # Línea central
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 21)
    ax.set_aspect('equal')
    ax.axis('off') # Ocultar ejes numéricos

def generate_heatmap(df, player_id, output_filename="heatmap_temp.png"):
    """Genera un mapa de calor sobre la pista de pádel"""
    fig, ax = plt.subplots(figsize=(6, 10))
    
    # 1. Dibujar Pista
    draw_padel_court(ax)
    
    # 2. Filtrar datos del jugador y dibujar KDE (Heatmap)
    # Asumimos que tu CSV tiene columnas 'player_x' y 'player_y' en metros (0-10, 0-20)
    player_data = df[df['player_id'] == player_id]
    
    if not player_data.empty:
        sns.kdeplot(
            x=player_data['x'], 
            y=player_data['y'], 
            fill=True, 
            cmap="OrRd", # Colores de fuego (Rojo/Naranja)
            alpha=0.6,   # Transparencia para ver las líneas debajo
            thresh=0.1,  # Limpiar el fondo
            ax=ax
        )
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_full_report(csv_path, output_pdf="Reporte_Partido.pdf"):
    # 1. Cargar Datos
    df = pd.read_csv(csv_path)
    
    # 2. Instanciar PDF
    pdf = PadelReport()
    pdf.add_page()
    
    # --- SECCIÓN 1: RESUMEN FÍSICO ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, '1. Análisis Físico y Cobertura de Pista', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "A continuación se muestra el mapa de calor que indica las zonas de mayor ocupación del jugador durante el partido. Las zonas rojas indican mayor permanencia.")
    pdf.ln(5)
    
    # Generar Heatmap (Ejemplo para Jugador 1)
    # NOTA: Asegúrate de que tu CSV tenga las columnas correctas o renómbralas aquí
    # Simulamos columnas x e y si no existen para que el código no falle al probar
    if 'x' not in df.columns: 
        # Mapeo rápido basado en tu CSV ('player1_x', 'player1_y')
        # Aquí deberías filtrar por frame para "aplanar" tu CSV si tiene columnas anchas
        # O usar una columna 'player_active' si ya lo tienes procesado.
        pass 

    # IMPORTANTE: Aquí asumo que transformas tu CSV "ancho" (player1_x, player2_x...) 
    # a un formato "largo" o eliges uno para pintar.
    # Para el ejemplo, usaremos player1_x como 'x'
    heatmap_df = pd.DataFrame({
        'player_id': 1,
        'x': df['player1_x'],
        'y': df['player1_y']
    })
    
    generate_heatmap(heatmap_df, player_id=1, output_filename="heatmap_p1.png")
    
    # Insertar Imagen en PDF
    # (x, y, w, h) -> Centrado
    pdf.image("heatmap_p1.png", x=60, y=50, w=90)
    
    # --- SECCIÓN 2: ESTADÍSTICAS DE GOLPES ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, '2. Clasificación de Golpes', ln=True)
    
    # Tabla simple de golpes (Si tienes la columna 'shot_type')
    if 'shot_type' in df.columns:
        shot_counts = df['shot_type'].value_counts()
        
        pdf.set_font('Helvetica', '', 10)
        pdf.ln(5)
        
        # Cabecera Tabla
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(80, 10, 'Tipo de Golpe', border=1, fill=True)
        pdf.cell(40, 10, 'Cantidad', border=1, fill=True, ln=True)
        
        # Filas
        for golpe, count in shot_counts.items():
            pdf.cell(80, 10, str(golpe), border=1)
            pdf.cell(40, 10, str(count), border=1, ln=True)

    # Guardar
    pdf.output(output_pdf)
    
    # Limpieza
    if os.path.exists("heatmap_p1.png"):
        os.remove("heatmap_p1.png")
    
    print(f"✅ Reporte generado: {output_pdf}")

# Para probarlo directamente si ejecutas el script
if __name__ == "__main__":
    # Usa tu archivo real aquí
    create_full_report("padel_analytics_report.csv")
