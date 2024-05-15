import streamlit as st
from gallery.image_gallery import ImageGallery
from PIL import Image
import glob
import altair as alt
import pandas as pd


st.set_page_config(layout="wide")
st.title("Aircraft Classification")
type_selectbox = st.sidebar.selectbox(  # Add selectbox to the sidebar
    "Classification Type", ("Manufacturer", "Family", "Variant")
)

images = glob.glob("../fgvc-aircraft-2013b/data/images/*.jpg")  # Load images
page_size = 16  # Gallery page size
page_max = len(images) // page_size  # Maximum number of pages
# Create an empty placeholder for the images that can be updated later
placeholder = st.empty()

# Initialize index state
if "index" not in st.session_state:
    st.session_state.index = 0

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = 1


def increment_index():
    """Increment classification image index"""
    if st.session_state.index < len(images) - 1:
        st.session_state.index += 1
    elif st.session_state.index == len(images) - 1:
        st.session_state.index = 0
    set_images()


def decrement_index():
    """Decrement classification image index"""
    if st.session_state.index > 0:
        st.session_state.index -= 1
    elif st.session_state.index == 0:
        st.session_state.index = len(images) - 1
    set_images()


def increment_page():
    """Increment image gallery page"""
    if st.session_state.page < len(images) / page_size:
        st.session_state.page += 1
    elif st.session_state.page == len(images) / page_size:
        st.session_state.page = 1


def decrement_page():
    """Decrement image gallery page"""
    if st.session_state.page > 1:
        st.session_state.page -= 1
    elif st.session_state.page == 1:
        st.session_state.page = page_max


def set_images(img_custom: Image.Image = None):
    """Set original image and heatmap in a placeholder to make it easier to update

    Args:
        img_custom (Image.Image, optional): Use to set an image uploaded by the user. Defaults to None.
    """
    placeholder.empty()  # Clear the placeholder
    with placeholder.container():
        # Add columns to display images
        col_original, col_heatmap = st.columns([1, 1])

        # Display images if available
        if images:
            if img_custom:
                img = img_custom.resize((800, 500))
            else:
                img = Image.open(images[st.session_state.index]).resize((800, 500))
            heatmap = Image.open("./heatmap.png").resize((800, 500))
            col_original.header("Original Image")
            col_original.image(img, use_column_width=True)
            col_heatmap.header("Heatmap")
            col_heatmap.image(heatmap, use_column_width=True)


set_images()

# Add columns to display buttons
col_upload, space1, col_prev, space2, col_next, space3, col_classify = st.columns(
    [3, 2, 1, 4, 1, 3, 1.5]
)

# Button to move to previous image
if col_prev.button(
    "<",
    key="btn_prev_img",
    on_click=decrement_index,
    disabled=st.session_state.get("disabled", False),
):
    pass

# Button to move to next image
if col_next.button(
    "\>",
    key="btn_next_img",
    on_click=increment_index,
):
    pass

col_classify.button("Classify")

# Add file uploader
uploaded_file = st.file_uploader(
    "Upload custom image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    uploaded_image = Image.open(uploaded_file).resize((800, 500))
    # Replace the original image with the uploaded image
    set_images(uploaded_image)


# Add bar chart to show classification scores
chart_data = pd.DataFrame(
    [0.4, 0.3, 0.15, 0.1, 0.05],
    index=["Airbus", "Boeing", "Bombardier", "Cessna", "Embraer"],
)

# Convert DataFrame to a format suitable for Altair
chart_data = pd.melt(chart_data.reset_index(), id_vars=["index"])

# Create bar chart
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
    col_info.write(f"Page {st.session_state.page}/{len(images) // page_size}")
