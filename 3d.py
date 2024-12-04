import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Cube 3D avec Plotly", layout="centered")

# Titre de l'application
st.title("Visualisation d'un cube en 3D avec Plotly")

# Définir les coordonnées du cube
x = np.array([0, 1, 1, 0, 0, 1, 1, 0])
y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
z = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Calculer le centre du cube bleu
center_x = (x.max() + x.min()) / 2
center_y = (y.max() + y.min()) / 2
center_z = (z.max() + z.min()) / 2

# Fonction pour effectuer la rotation d'un cube autour d'un point
def rotate_cube(x, y, z, angle_x, angle_y, angle_z, cx, cy, cz):
    # Translater les points pour centrer le cube sur l'origine
    x = x - cx
    y = y - cy
    z = z - cz

    # Matrices de rotation
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Appliquer les rotations
    coords = np.dot(rz, np.dot(ry, np.dot(rx, np.vstack([x, y, z]))))

    # Re-translater les points pour revenir au centre initial
    x_rot, y_rot, z_rot = coords[0] + cx, coords[1] + cy, coords[2] + cz
    return x_rot, y_rot, z_rot

# Fonction pour générer une base orthogonale
def generate_base(cx, cy, cz, angles):
    base_vectors = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1])
    }

    # Matrices de rotation
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    # Appliquer les rotations aux vecteurs de base
    rotation_matrix = np.dot(rz, np.dot(ry, rx))
    rotated_base = {key: np.dot(rotation_matrix, vec) + np.array([cx, cy, cz]) for key, vec in base_vectors.items()}
    return rotated_base

# Curseurs pour les angles de rotation
angle_x = st.slider("Rotation autour de X (degrés)", 0, 360, 0)
angle_y = st.slider("Rotation autour de Y (degrés)", 0, 360, 0)
angle_z = st.slider("Rotation autour de Z (degrés)", 0, 360, 0)

# Convertir les angles en radians
angle_x_rad = np.radians(angle_x)
angle_y_rad = np.radians(angle_y)
angle_z_rad = np.radians(angle_z)

# Calculer les nouvelles coordonnées pour le cube rouge
x_red, y_red, z_red = rotate_cube(x, y, z, angle_x_rad, angle_y_rad, angle_z_rad, center_x, center_y, center_z)

# Définir les arêtes du cube (listes des segments)
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # base inférieure
    [4, 5], [5, 6], [6, 7], [7, 4],  # base supérieure
    [0, 4], [1, 5], [2, 6], [3, 7]   # arêtes verticales
]

# Créer les segments des deux cubes
lines = []
for edge in edges:
    # Cube bleu
    lines.append(go.Scatter3d(
        x=[x[edge[0]], x[edge[1]]],
        y=[y[edge[0]], y[edge[1]]],
        z=[z[edge[0]], z[edge[1]]],
        mode='lines',
        line=dict(color='blue', width=4),
        showlegend=True
    ))
    # Cube rouge
    lines.append(go.Scatter3d(
        x=[x_red[edge[0]], x_red[edge[1]]],
        y=[y_red[edge[0]], y_red[edge[1]]],
        z=[z_red[edge[0]], z_red[edge[1]]],
        mode='lines',
        line=dict(color='red', width=4),
        showlegend=True
    ))

# Générer les bases orthogonales
blue_base = generate_base(center_x, center_y, center_z, (0, 0, 0))
red_base = generate_base(center_x, center_y, center_z, (angle_x_rad, angle_y_rad, angle_z_rad))

# Ajouter les vecteurs des bases orthogonales
for color, base in zip(['blue', 'red'], [blue_base, red_base]):
    for vec_name, vec in base.items():
        lines.append(go.Scatter3d(
            x=[center_x, vec[0]],
            y=[center_y, vec[1]],
            z=[center_z, vec[2]],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            name=f"Base {color} - {vec_name}"
        ))

# Créer la figure 3D
fig = go.Figure(data=lines)

# Configurer la disposition
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectratio=dict(x=1, y=1, z=1),
        xaxis=dict(range=[-1.5, 1.5]),
        yaxis=dict(range=[-1.5, 1.5]),
        zaxis=dict(range=[-1.5, 1.5]),
    ),
    title="Cube 3D avec Rotation",
)

# Afficher la figure dans Streamlit
st.plotly_chart(fig)