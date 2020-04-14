import streamlit as st


def plot_with_streamlit(plot_function, inputs):
    info = plot_function.__doc__
    lines = info.split("\n")
    title = lines[0]
    description = "\n".join(lines[1:])
    figure = plot_function(inputs)
    figure.update_layout(title_text=title, title_font_size=20)
    st.write(figure)
    show_info = st.checkbox(f"Show Info: {title}")
    if show_info:
        st.write(description)
