import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_colors(image, num_colors):
    # Convert image to RGB and reshape it
    image = np.array(image)
    image = image.reshape((-1, 3))
    
    # Use KMeans clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def plot_colors(colors):
    # Create a color palette visualization
    palette = np.zeros((100, 500, 3), dtype=int)
    step = 500 // len(colors)
    for i, color in enumerate(colors):
        palette[:, i * step:(i + 1) * step, :] = color
    return palette

st.title("Color Palette Picker from Real-World Images")
st.write("Upload an image to generate a harmonious color palette.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    num_colors = st.slider("Number of colors in the palette", 2, 10, 5)
    
    if st.button("Generate Palette"):
        colors = extract_colors(image, num_colors)
        palette = plot_colors(colors)
        
        st.image(palette, caption="Extracted Color Palette", use_column_width=True)
        st.write("Colors in HEX format:")
        for color in colors:
            st.write(f"#{''.join([f'{c:02X}' for c in color])}")

st.write("---")
st.write("Built for creatives who want inspiration from the real world. 🎨")
