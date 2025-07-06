import streamlit as st
from typing import Set

def create_source_links(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    
    sources_list = sorted(source_urls)
    sources_string = "\n ### Sources:\n"
    
    for i, source in enumerate(sources_list, 1):
        cleaned_source = source.replace("https:/\\", "Context Used: ").replace(".html", "")
        sources_string += f"> {i}. {cleaned_source}\n"

    return sources_string


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)