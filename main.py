import streamlit as st

from server.core import get_retrieval_chain
from utils import create_source_links, load_css

st.set_page_config(
    page_title="LangChain Docs Helper Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = "assets/style.css"
load_css(css_path)

st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.50.0/files/dark/langchain.png" width="100">
    </div>
    <h1>LangChain Helper Bot</h1>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **LLM Model:** `nvidia/llama-3.3-nemotron-super-49b-v1`  
    **Temperature:** `0.3`  
    **Embedding Model:** `BAAI/bge-large-en-v1.5`  
    **Vector Store:** `Pinecone`
    """
)

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
    and "selected_suggestion" not in st.session_state
    and "generating" not in st.session_state
):
    st.session_state.chat_answers_history = []
    st.session_state.user_prompt_history = []
    st.session_state.chat_history = []
    st.session_state.selected_suggestion = ""
    st.session_state.generating = False

st.markdown(
    """
    <div class="chat-header">
        <h1 class="chat-title">
            LangBot: Your LangChain Guide
        </h1>
        <div class="model-selector">
            nvidia/llama-3.3
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if not st.session_state["chat_answers_history"]:
    suggestions = [
        "How do I use a RecursiveUrlLoader to load content from a page?",
        "How can I define the state schema for my LangGraph graph?",
        "How can I run a model locally on my laptop with Ollama?",
        "Explain RAG techniques, how LangGraph can implement them."
    ]
    
    st.markdown('<div class="suggestions-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with col1 if i % 2 == 0 else col2:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.selected_suggestion = suggestion
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state["chat_answers_history"]:
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([10, 1])
with col1:
    input_key = f"chat_input_{len(st.session_state.get('chat_answers_history', []))}"
    default_value = st.session_state.get('selected_suggestion', '')
    prompt = st.text_area(
        "Message",
        placeholder="How can I...",
        height=68,
        key=input_key,
        label_visibility="collapsed",
        value=default_value
    )
with col2:
    send_button = st.button(
        "➤", 
        key="send_button", 
        use_container_width=True, 
        disabled=st.session_state.generating
    )

if send_button and not st.session_state.generating:
    if not prompt.strip(): 
        st.markdown("""
            <span style="
                background-color: #f8d7da;  
                color: #721c24;            
                border: 1px solid #f5c6cb;
                border-radius: 4px;
                margin-left: 30px;
                margin-bottom: 12px;
                padding: 10px;
                font-weight: 500;">
                ⚠️ Please enter a message before submitting.
            </span>
            """, unsafe_allow_html=True)
    else:
        st.session_state.generating = True
        st.rerun()

if st.session_state.generating:
    with st.spinner("Generating Response...", show_time=True):
        response_generator = get_retrieval_chain(
            query=prompt, 
            chat_history=st.session_state["chat_history"]
        )

        response_placeholder = st.empty()
        full_answer = ""
        source_documents = None

        for chunk in response_generator:
            token = chunk.get("answer", "")
            full_answer += token
            response_placeholder.markdown(full_answer)

            if "context" in chunk:
                source_documents = chunk["context"]

        sources = set(
            doc.metadata["source"] for doc in source_documents  
        ) if source_documents else set()

        formatted_response = create_source_links(sources)

        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(full_answer + formatted_response)
        st.session_state.chat_history.append(("user", prompt))  
        st.session_state.chat_history.append(("assistant", full_answer))
        st.session_state.selected_suggestion = ""
        st.session_state.generating = False

        st.rerun()
