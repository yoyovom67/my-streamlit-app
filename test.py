
import streamlit as st
import numpy as np
from numpy.linalg import eig
import plotly.graph_objects as go


theta_deg = 0

def main():
    # Configuration de la page pour utiliser toute la largeur
    st.set_page_config(layout="wide")

    # Titre de l'application
    st.title("Calcul en temps réel du cercle de Mohr avec Visualisation des Tenseurs")

    # Afficher les graphiques côte à côte en haut de page
    st.write("## Visualisations")
    col1, col2 = st.columns(2)

    # Interface utilisateur pour entrer les éléments indépendants de la matrice
    stress_matrix, is_2d = get_user_input()

    # Afficher la matrice en LaTeX
    display_stress_matrix(stress_matrix)

    # Calcul des contraintes principales
    principal_stresses, eigenvectors = calculate_principal_stresses(stress_matrix)
    #print(eigenvectors)
    # Tracer le cercle de Mohr
    fig_mohr = plot_mohr_circles(principal_stresses, stress_matrix, is_2d)

    if is_2d:
        fig_graph = create_2d_visualization()
        
    else :
        fig_graph = create_cube_visualization(eigenvectors)



    # Afficher les graphiques dans les colonnes du haut
    with col1:
        st.plotly_chart(fig_mohr, use_container_width=True)
    with col2:
        st.plotly_chart(fig_graph, use_container_width=True)

def get_user_input():
    st.write("### Entrez les éléments indépendants de votre matrice des contraintes symétrique (3x3) :")

    col_input1, col_input2, col_input3 = st.columns(3)
    sigma_xx = col_input1.number_input(r"$\sigma_{xx}$ (MPa)", value=75.0, step=1.0)	
    sigma_xy = col_input2.number_input(r"$\tau_{xy}$ (MPa)", value=-43.3, step=1.0)
    sigma_xz = col_input3.number_input(r"$\tau_{xz}$ (MPa)", value=0.0, step=1.0)
    sigma_yy = col_input2.number_input(r"$\sigma_{yy}$ (MPa)", value=25.0, step=1.0)
    sigma_yz = col_input3.number_input(r"$\tau_{yz}$ (MPa)", value=0.0, step=1.0)
    sigma_zz = col_input3.number_input(r"$\sigma_{zz}$ (MPa)", value=-100.0, step=1.0)

    # Construction de la matrice symétrique
    stress_matrix = np.array([
        [sigma_xx, sigma_xy, sigma_xz],
        [sigma_xy, sigma_yy, sigma_yz],
        [sigma_xz, sigma_yz, sigma_zz]
    ])

    # Vérifier si on est en 2D (troisième ligne et colonne nulles)
    is_2d = np.allclose(stress_matrix[2], 0) and np.allclose(stress_matrix[:, 2], 0)

    return stress_matrix, is_2d

def display_stress_matrix(stress_matrix):
    st.write("### Matrice des contraintes (symétrique) :")
    latex_matrix = r"""
    \begin{bmatrix}
    %.2f & %.2f & %.2f \\ 
    %.2f & %.2f & %.2f \\ 
    %.2f & %.2f & %.2f
    \end{bmatrix}
    """ % (
        stress_matrix[0,0], stress_matrix[0,1], stress_matrix[0,2],
        stress_matrix[1,0], stress_matrix[1,1], stress_matrix[1,2],
        stress_matrix[2,0], stress_matrix[2,1], stress_matrix[2,2]
    )
    st.latex(latex_matrix)

def calculate_principal_stresses(stress_matrix):
    # Calcul des contraintes principales (valeurs propres) et vecteurs propres
    principal_stresses, eigenvectors = eig(stress_matrix)
    
    # Trier les contraintes principales par ordre décroissant
    #sorted_indices = np.argsort(principal_stresses)[::-1]  # Indices triés
    #principal_stresses = principal_stresses[sorted_indices]
    #eigenvectors = eigenvectors[:, sorted_indices]  # Réordonner les vecteurs propres
    
    # Ajuster l'orientation des vecteurs propres pour rester cohérent avec le repère principal
    principal_axes = np.eye(3)  # Repère principal standard
    for i in range(3):
        if np.dot(eigenvectors[:, i], principal_axes[:, i]) < 0:  # Orientation incorrecte
            eigenvectors[:, i] *= -1  # Inverser le vecteur propre
            #print(f"Vecteur propre {i} inversé.")

    return principal_stresses, eigenvectors


def plot_mohr_circles(principal_stresses, stress_matrix, is_2d):
    sigma_1, sigma_2, sigma_3 = principal_stresses
    
    st.write(f"### Contraintes principales :")#
    #ecrire en gras
    st.latex(f"\\text{{σ₁}} = {sigma_1:.2f} \, \\text{{MPa}}")

    if is_2d:
        if sigma_2:
            st.latex(f"\\text{{σ₂}} = {sigma_2:.2f} \, \\text{{MPa}}")
        else:
            st.latex(f"\\text{{σ₂}} = {sigma_3:.2f} \, \\text{{MPa}}")

        return plot_mohr_circle_2d(stress_matrix)
        
    else:
        st.latex(f"\\text{{σ₂}} = {sigma_2:.2f} \, \\text{{MPa}}")
        st.latex(f"\\text{{σ₃}} = {sigma_3:.2f} \, \\text{{MPa}}")
        return plot_mohr_circles_3d(principal_stresses, stress_matrix)

def plot_mohr_circle_2d(stress_matrix):
    # Extraire les contraintes du tenseur de contrainte
    sigma_xx = stress_matrix[0, 0]
    sigma_yy = stress_matrix[1, 1]
    tau_xy = stress_matrix[0, 1]

    st.write("### Problème en 2D : Tracé du cercle de Mohr avec angle.")

    # Calculer le centre et le rayon du cercle de Mohr
    center = (sigma_xx + sigma_yy) / 2
    radius = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy ** 2)

    st.write(f"Centre du cercle de Mohr :")
    # \( c = {center:.2f} \, \text{{MPa}} \)
    st.latex(f"c = {center:.2f} \, \\text{{MPa}}")
    st.write(f"Rayon du cercle de Mohr :")
    #dire r=tau_max = la valeur du rayon
    st.latex(f"r = \\tau_{{max}} = {radius:.2f} \, \\text{{MPa}}")
    # Créer le cercle de Mohr
    fig_mohr = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 360)
    sigma_n = center + radius * np.cos(theta)
    tau_n = radius * np.sin(theta)
    fig_mohr.add_trace(go.Scatter(x=sigma_n, y=tau_n, mode='lines', name="Cercle de Mohr"))

    # Ajouter les contraintes principales sur le cercle
    sigma_1 = center + radius
    sigma_2 = center - radius
    fig_mohr.add_trace(go.Scatter(
        x=[sigma_1, sigma_2],
        y=[0, 0],
        mode='markers+text',
        text=["σ₁", "σ₂"],
        textposition="bottom center",
        marker=dict(size=10, color='red'),
        name="Contraintes principales"
    ))

    # Ajouter le point représentant l'état de contrainte initial
    plot_additional_elements_2d(fig_mohr, stress_matrix)

    # Configuration du graphique du cercle de Mohr
    configure_mohr_plot(fig_mohr)

    return fig_mohr

def plot_mohr_circles_3d(principal_stresses, stress_matrix):
    #sort principal stresses dans l'ordre croissant
    sigma_1, sigma_2, sigma_3 = np.sort(principal_stresses)    
    # Calcul des centres et rayons des cercles à partir des contraintes principales
    circles = [
        {'center': (sigma_1 + sigma_3) / 2, 'radius': abs(sigma_1 - sigma_3) / 2, 'name': "Cercle σ₁-σ₃"},
        {'center': (sigma_1 + sigma_2) / 2, 'radius': abs(sigma_1 - sigma_2) / 2, 'name': "Cercle σ₁-σ₂"},
        {'center': (sigma_2 + sigma_3) / 2, 'radius': abs(sigma_2 - sigma_3) / 2, 'name': "Cercle σ₂-σ₃"},
    ]

    # Tracer les cercles
    fig_mohr = go.Figure()
    for circle in circles:
        add_mohr_circle(fig_mohr, circle['center'], circle['radius'], circle['name'])

    # Ajouter les contraintes principales
    fig_mohr.add_trace(go.Scatter(
        x=[sigma_1, sigma_2, sigma_3],
        y=[0, 0, 0],
        mode='markers+text',
        text=["σ₁", "σ₂", "σ₃"],
        textposition="top center",
        marker=dict(size=10, color='red'),
        name="Contraintes principales"
    ))

    # Ajouter le point τ_max pour le plus grand cercle
    tau_max = circles[0]['radius']
    center = circles[0]['center']
    fig_mohr.add_trace(go.Scatter(
        x=[center],
        y=[tau_max],
        mode='markers+text',
        text="τ_max",
        textposition="top center",
        marker=dict(size=10, color='blue'),
        name="τ_max"
    ))

    # Vérifier si les contraintes de cisaillement τ_xz et τ_yz sont nulles
    tau_xz = stress_matrix[0, 2]
    tau_yz = stress_matrix[1, 2]

    if tau_xz == 0 and tau_yz == 0:
        # Reproduire le tracé 2D sur le cercle correspondant
        fig_mohr = plot_additional_elements_2d(fig_mohr, stress_matrix)

    # Configuration du graphique du cercle de Mohr
    configure_mohr_plot(fig_mohr)

    return fig_mohr

def plot_additional_elements_2d(fig_mohr, stress_matrix):
    # Fonction pour ajouter les éléments supplémentaires en cas de contraintes de cisaillement nulles sur τ_xz et τ_yz
    sigma_xx = stress_matrix[0, 0]
    sigma_yy = stress_matrix[1, 1]
    tau_xy = stress_matrix[0, 1]
    global theta_deg
    center = (sigma_xx + sigma_yy) / 2
    radius = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + tau_xy ** 2)

    # Ajouter le point représentant l'état de contrainte initial
    fig_mohr.add_trace(go.Scatter(
        x=[sigma_xx],
        y=[tau_xy],
        mode='markers+text',
        text=["X'"],
        textposition="top center",
        marker=dict(size=8, color='green'),
        name="État initial"
    ))

    fig_mohr.add_trace(go.Scatter(
        x=[sigma_yy],
        y=[-tau_xy],
        mode='markers+text',
        text=["Y'"],
        textposition="bottom center",
        marker=dict(size=8, color='green'),
        showlegend=False
    ))

    # Tracer le segment de (σₓₓ, τ_xy) à (σ_yy, -τ_xy)
    if tau_xy != 0:
        fig_mohr.add_trace(go.Scatter(
            x=[sigma_xx, sigma_yy],
            y=[tau_xy, -tau_xy],
            mode='lines+markers',
            line=dict(color='orange', width=2),
            marker=dict(size=6, color='orange'),
            name="Repère initial"
        ))

    # Calculer l'angle 2θ
    two_theta = np.arctan2(2 * tau_xy, sigma_xx - sigma_yy)
    theta_deg = np.degrees(two_theta) / 2  # θ en degrés

    st.write(f"L'angle θ correspondant à l'orientation des contraintes principales est :")
    st.latex(f"\\theta = {theta_deg:.2f}^\circ")

    # Tracer l'arc représentant l'angle 2θ et remplir l'aire
    scale_factor = 3
    arc_theta = np.linspace(0, two_theta, 100)
    arc_sigma = center + radius / scale_factor * np.cos(arc_theta)
    arc_tau = radius / scale_factor * np.sin(arc_theta)

    # Remplir l'aire sous l'arc
    fig_mohr.add_trace(go.Scatter(
        x=np.concatenate([[center], arc_sigma, [center]]),
        y=np.concatenate([[0], arc_tau, [0]]),
        fill='toself',
        fillcolor='rgba(128, 0, 128, 0.5)',  # Couleur violette avec opacité 0.5
        line=dict(color='purple', width=0),
        showlegend=False
    ))

    # Ajouter une annotation pour l'angle 2θ
    angle_annotation_x = center + radius * np.cos(two_theta / 2) * 0.7
    angle_annotation_y = radius * np.sin(two_theta / 2) * 0.7
    fig_mohr.add_annotation(
        x=angle_annotation_x,
        y=angle_annotation_y,
        text=f"2θ = {np.degrees(two_theta):.2f}°",
        showarrow=False,
        font=dict(color='purple')
    )
    #ajouter tau_max , sachant qe tau_max = radius et center est con
    fig_mohr.add_trace(go.Scatter(
        x=[center],
        y=[radius],
        mode='markers+text',
        text="τ_max",
        textposition="top center",
        marker=dict(size=10, color='blue'),
        name="τ_max"
    ))
    return fig_mohr

def add_mohr_circle(fig, center, radius, name):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center + radius * np.cos(theta)
    y = radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))

def configure_mohr_plot(fig_mohr):
    fig_mohr.update_layout(
        title="Cercle de Mohr",
        xaxis_title="Contrainte normale (σ) [MPa]",
        yaxis_title="Contrainte de cisaillement (τ) [MPa]",
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=True, zeroline=True),
        yaxis=dict(scaleanchor="x", scaleratio=1, showgrid=True, zeroline=True),
        template="plotly_white"
    )
def create_2d_visualization():
    # Paramètres
    theta_rad = np.radians(theta_deg)  # Conversion en radians
    arc_scale = .5  # Facteur d'échelle pour l'arc

    # Matrice de rotation pour le repère de base
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    # Repère principal (bleu, fixe)
    x_principal, y_principal = np.array([1, 0]), np.array([0, 1])

    # Repère de base (rouge)
    x_base = rotation_matrix @ x_principal
    y_base = rotation_matrix @ y_principal

    # Création de la figure
    fig_2d = go.Figure()

    # Carré aligné avec le repère principal (bleu)
    square = np.array([
        [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]
    ]).T
    fig_2d.add_trace(go.Scatter(
        x=square[0, :],
        y=square[1, :],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Carré principal'
    ))

    # Ajouter les axes principaux (bleu) avec annotations
    fig_2d.add_trace(go.Scatter(
        x=[0, x_principal[0]],
        y=[0, x_principal[1]],
        mode='lines',
        line=dict(color='blue', width=3),
        name='X principal'
    ))
    fig_2d.add_annotation(
        x=x_principal[0] * 1.2,
        y=x_principal[1] * 1.2,
        text="X principal",
        showarrow=False,
        font=dict(color='blue', size=12)
    )
    fig_2d.add_trace(go.Scatter(
        x=[0, y_principal[0]],
        y=[0, y_principal[1]],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Y principal'
    ))
    fig_2d.add_annotation(
        x=y_principal[0] * 1.2,
        y=y_principal[1] * 1.2,
        text="Y principal",
        showarrow=False,
        font=dict(color='blue', size=12)
    )

    # Ajouter un arc de cercle (violet, plein avec opacité)
    arc_angle = np.linspace(0, theta_rad, 100)
    arc_x = arc_scale * np.cos(arc_angle)
    arc_y = arc_scale * np.sin(arc_angle)
    fig_2d.add_trace(go.Scatter(
        x=np.append(arc_x, 0),  # Fermer l'arc
        y=np.append(arc_y, 0),
        fill='toself',
        fillcolor='rgba(128, 0, 128, 0.5)',  # Violet avec opacité
        mode='lines',
        line=dict(color='purple', width=2),
        name='Arc (θ)'
    ))

    # Ajouter \( \theta \) avec sa valeur absolue arrondie
    theta_text = f"&#952; = {abs(round(theta_deg, 2))}°"
    fig_2d.add_annotation(
        x=arc_scale * 0.7,
        y=arc_scale * 0.4,
        text=theta_text,
        showarrow=False,
        font=dict(color='purple', size=16)
    )

    # Carré aligné avec le repère de base (rouge)
    rotated_square = rotation_matrix @ square
    fig_2d.add_trace(go.Scatter(
        x=rotated_square[0, :],
        y=rotated_square[1, :],
        mode='lines',
        line=dict(color='red', width=2),
        name='Carré base'
    ))

    # Ajouter les axes de base (rouge) avec annotations
    fig_2d.add_trace(go.Scatter(
        x=[0, x_base[0]],
        y=[0, x_base[1]],
        mode='lines',
        line=dict(color='red', width=3),
        name='X initial'
    ))
    fig_2d.add_annotation(
        x=x_base[0] * 1.2,
        y=x_base[1] * 1.2,
        text="X initial",
        showarrow=False,
        font=dict(color='red', size=12)
    )
    fig_2d.add_trace(go.Scatter(
        x=[0, y_base[0]],
        y=[0, y_base[1]],
        mode='lines',
        line=dict(color='red', width=3),
        name='Y initial'
    ))
    fig_2d.add_annotation(
        x=y_base[0] * 1.2,
        y=y_base[1] * 1.2,
        text="Y initial",
        showarrow=False,
        font=dict(color='red', size=12)
    )

    # Configuration de la mise en page
    fig_2d.update_layout(
        title=f"Visualisation 2D : Repère principal et repère de base (θ = {theta_deg}°)",
        xaxis=dict(scaleanchor="y", title="X"),
        yaxis=dict(title="Y"),
        showlegend=False
    )

    return fig_2d

def create_cube_visualization(eigenvectors):
    # Définir les coordonnées du cube
    cube_coords = np.array([
        [0, 0, 0], [1, 0, 0],
        [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1],
        [1, 1, 1], [0, 1, 1]
    ]).T

    # Calculer le centre du cube
    center_cube = np.array([0.5, 0.5, 0.5])

    # Créer la figure 3D
    fig_cubes = go.Figure()

    # Créer le cube bleu (repère de base)
    blue_cube = create_cube(cube_coords, 'blue')
    for trace in blue_cube:
        fig_cubes.add_trace(trace)

    # Ajouter les axes du cube bleu
    add_axes(fig_cubes, center_cube, np.identity(3), 'blue')

    # Calculer la rotation à partir des vecteurs propres (eigenvectors)
    rotation_matrix = eigenvectors

    # Appliquer la rotation au cube rouge (repère principal)
    rotated_coords = rotation_matrix @ (cube_coords - center_cube[:, np.newaxis])
    rotated_coords += center_cube[:, np.newaxis]

    # Créer le cube rouge (roté)
    red_cube = create_cube(rotated_coords, 'red')
    for trace in red_cube:
        fig_cubes.add_trace(trace)

    # Ajouter les axes du cube rouge
    add_axes(fig_cubes, center_cube, rotation_matrix, 'red')

    # Configurer la disposition
    fig_cubes.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[-1, 2]),
            yaxis=dict(range=[-1, 2]),
            zaxis=dict(range=[-1, 2]),
        ),
        title="Visualisation des Tenseurs de Contrainte",
    )

    return fig_cubes

def create_cube(coords, color):
    # Vérifier les dimensions de coords
    assert coords.shape == (3, 8), "Les coordonnées doivent avoir la forme (3, 8)"
    
    # Liste des arêtes du cube
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # base inférieure
        [4, 5], [5, 6], [6, 7], [7, 4],  # base supérieure
        [0, 4], [1, 5], [2, 6], [3, 7]   # arêtes verticales
    ]

    lines = []

    for edge in edges:
        lines.append(go.Scatter3d(
            x=coords[0, edge],  # Axe X
            y=coords[1, edge],  # Axe Y
            z=coords[2, edge],  # Axe Z
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False
        ))

    return lines

def add_axes(fig, center, rotation_matrix, color):
    axes_length = 1.0
    axes = np.array([
        [axes_length, 0, 0],
        [0, axes_length, 0],
        [0, 0, axes_length]
    ]).T

    labels = ['X', 'Y', 'Z']
    axis_colors = [color] * 3

    # Appliquer la rotation
    rotated_axes = rotation_matrix @ axes

    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0] + rotated_axes[0, i]],
            y=[center[1], center[1] + rotated_axes[1, i]],
            z=[center[2], center[2] + rotated_axes[2, i]],
            mode='lines+text',
            line=dict(color=axis_colors[i], width=4),
            text=[None, labels[i]],
            textfont=dict(color=axis_colors[i], size=12),
            textposition='middle center',
            showlegend=False
        ))

if __name__ == "__main__":
    main()
