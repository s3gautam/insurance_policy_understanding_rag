import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS

# ------------------------
# PAGE SETUP
# ------------------------

st.set_page_config(
    page_title="Insurance Policy Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Insurance Policy Assistant")
st.write("Upload an insurance policy PDF and ask questions about it.")

# ------------------------
# FILE UPLOAD
# ------------------------

uploaded_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type=["pdf"]
)

# ------------------------
# PDF PROCESSING
# ------------------------

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Reading and indexing policy..."):

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    st.success(
        f"Policy loaded successfully ({len(chunks)} chunks indexed)"
    )

    # ------------------------
    # QUESTION INPUT
    # ------------------------

    question = st.text_input(
        "Ask a question about the policy"
    )

    if question:

        with st.spinner("Searching policy..."):

            results = vectorstore.similarity_search(
                question,
                k=4
            )

            context = ""
            pages = []

            for doc in results:

                page = doc.metadata.get("page", 0)

                pages.append(page + 1)

                context += (
                    f"\n[Page {page + 1}]\n"
                    f"{doc.page_content}\n"
                )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0
            )

            prompt = f"""
You are an expert insurance policy assistant.

The uploaded document is an insurance policy.

Answer the customer's question using ONLY information found in the policy.

Rules:
- Give a clear and direct answer.
- Be concise and customer-friendly.
- Explain insurance terms simply.
- Do not mention chunks, retrieval, context, embeddings, or internal processing.
- Do not mention page numbers in the answer.
- Do not make up information.
- If the answer is not present in the policy, respond exactly:
"I could not find this information in the policy."

Policy Content:
{context}

Customer Question:
{question}
"""

            response = llm.invoke(prompt)

            unique_pages = sorted(set(pages))

            st.markdown("## Answer")
            st.write(response.content)

            st.markdown("### 📄 Source Pages")
            st.write(
                ", ".join(
                    [f"Page {p}" for p in unique_pages]
                )
            )
