import streamlit as st

from Rag_pipeline.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Marburg Data Science Assistant",
    #page_icon="🎓",
    layout="wide"
)

st.title("Marburg Data Science Admission Assistant")

st.markdown(
    "Ask questions about the **MSc Data Science program, admission requirements, deadlines, scholarships, and studying in Marburg.**"
)

# Load RAG system
@st.cache_resource
def load_rag():
    return RAGPipeline()

rag = load_rag()


# Initializing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
query = st.chat_input("Ask a question about the MSc Data Science program...")


if query:

    # show user message
    st.chat_message("user").markdown(query)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("assistant"):

        with st.spinner("Searching and generating answer..."):

            answer, chunks = rag.generate_answer(query)

        st.markdown(answer)

        # Sources
        st.markdown("---")
        st.markdown("**Sources**")

        seen_urls = set()

        for chunk in chunks:

            url = chunk["metadata"]["url"]
            title = chunk["metadata"]["title"]

            if url not in seen_urls:
                st.markdown(f"- [{title}]({url})")
                seen_urls.add(url)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )