import streamlit as st
from gallery.image_gallery import ImageGallery
from PIL import Image
import glob
import altair as alt
import pandas as pd

st.set_page_config(layout="wide")
st.title("Aircraft Classification")

# Add a selectbox to the sidebar
type_selectbox = st.sidebar.selectbox(
    "Classification Type", ("Manufacturer", "Family", "Variant")
)

images = glob.glob("../fgvc-aircraft-2013b/data/images/*.jpg")

# Initialize the index in session state if it is not already set
if "index" not in st.session_state:
    st.session_state.index = 0


# Update function to increment index
def increment_index():
    if st.session_state.index < len(images) - 1:
        st.session_state.index += 1


# Update function to decrement index
def decrement_index():
    if st.session_state.index > 0:
        st.session_state.index -= 1


# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = 1


# Update function to increment page
def increment_page():
    if st.session_state.page < len(images) / 16:
        st.session_state.page += 1


# Update function to decrement page
def decrement_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1


col_original, col_heatmap = st.columns([1, 1])

if images:
    original = Image.open(images[st.session_state.index]).resize((800, 500))
    heatmap = Image.open("./heatmap.png").resize((800, 500))
    col_original.header("Original Image")
    col_original.image(original, use_column_width=True)
    col_heatmap.header("Heatmap")
    col_heatmap.image(heatmap, use_column_width=True)

# Add control buttons to switch images
col_upload, space1, col_prev, space2, col_next, space3, col_classify = st.columns(
    [2, 2, 1, 4, 1, 3, 1.5]
)

col_upload.button("Upload")
if col_prev.button(":arrow_left:", on_click=decrement_index):
    pass
if col_next.button(":arrow_right:", on_click=increment_index):
    pass
col_classify.button("Classify")

# Ensure index remains within the bounds
st.session_state.index = max(0, min(st.session_state.index, len(images) - 1))

# Add bar chart to show the classification scores
chart_data = pd.DataFrame(
    [0.4, 0.3, 0.15, 0.1, 0.05],
    index=["Airbus", "Boeing", "Bombardier", "Cessna", "Embraer"],
)

chart_data = pd.melt(chart_data.reset_index(), id_vars=["index"])

chart = (
    alt.Chart(chart_data, title="Classification Scores", height=300)
    .mark_bar()
    .encode(
        x=alt.X(
            "value", type="quantitative", title="Score", axis=alt.Axis(labelFontSize=16)
        ),
        y=alt.Y(
            "index",
            type="nominal",
            title="Manufacturer",
            axis=alt.Axis(labelFontSize=16),
        ),
        color=alt.Color("index", type="nominal", title="", legend=None),
        order=alt.Order("index", sort="descending"),
    )
)
# Add some space
st.text("")
st.text("")
st.text("")
st.altair_chart(chart, use_container_width=True)

with st.expander("Dataset Gallery"):
    # Add image gallery
    default_gallery = ImageGallery(
        directory="../fgvc-aircraft-2013b/data/images",
        number_of_columns=4,
        page=st.session_state.page,
    )
    # Add control buttons to switch pages
    spacer, col_prev, spacer, col_next, spacer, col_info = st.columns(
        [4, 1, 1, 1, 2, 1]
    )
    if col_prev.button("<", on_click=decrement_page):
        pass
    if col_next.button("\>", on_click=increment_page):
        pass
    col_info.write(f"Page {st.session_state.page}/{len(images) // 16}")
