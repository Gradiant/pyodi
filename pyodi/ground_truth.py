import json
import re

import streamlit as st

def m
st.title("Object Detection Insights")

st.header("Ground Truth")

ground_truth_file = st.sidebar.file_uploader("COCO Ground Truth File", type="json")

string_to_match = st.sidebar.text_input("String to match", value="drone_racing")

if ground_truth_file is not None:
    pass