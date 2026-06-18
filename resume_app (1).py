import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")

# ---------------------------------
# GROQ MODEL
# ---------------------------------

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ---------------------------------
# PROMPT
# ---------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are an Expert ATS Resume Screening Assistant.

Resume:
{context}

Job Description:
{question}

Provide:

1. ATS Score (0-100)
2. Matching Skills
3. Missing Skills
4. Strengths
5. Weaknesses
6. Project Analysis
7. Education Analysis
8. Hiring Recommendation

Rules:

ATS Score >= 80: STRONGLY RECOMMENDED
ATS Score 60-79: RECOMMENDED
ATS Score 40-59: CONSIDER
ATS Score < 40: REJECTED

Give professional output.
"""
)

output_parser = StrOutputParser()

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------
# JOB DESCRIPTION INPUT
# ---------------------------------

job_description = st.text_area(
    "Enter Job Description",
    height=250,
    placeholder="""Example:

Python Developer

Required Skills:
- Python
- Machine Learning
- SQL
- Flask
- GitHub

Experience:
2+ Years
"""
)

# ---------------------------------
# ANALYSIS
# ---------------------------------

if st.button("Analyze Resumes"):

    if not uploaded_files:
        st.warning("Upload at least one Resume PDF")
        st.stop()

    if job_description.strip() == "":
        st.warning("Enter Job Description")
        st.stop()

    results = []

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    jd_embedding = embeddings.embed_query(job_description)

    required_skills = [
        "python", "machine learning", "sql",
        "flask", "github", "java",
        "html", "css", "javascript"
    ]

    for uploaded_file in uploaded_files:

        with st.spinner(f"Analyzing {uploaded_file.name}..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            docs = splitter.split_documents(documents)

            vector_db = FAISS.from_documents(docs, embeddings)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(job_description)

            resume_text = "\n".join([doc.page_content for doc in docs])
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # ATS SCORE
            resume_embedding = embeddings.embed_query(resume_text[:5000])
            score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
            ats_score = round(score * 100, 2)

            # MATCHING / MISSING SKILLS
            resume_lower = resume_text.lower()
            matching_skills = [s for s in required_skills if s in resume_lower]
            missing_skills = [s for s in required_skills if s not in resume_lower]

            if ats_score >= 40:
                status = "✅ SHORTLISTED"
            elif ats_score >= 30:
                status = "⚠️ CONSIDER"
            else:
                status = "❌ REJECTED"

            chain = prompt | llm | output_parser
            llm_result = chain.invoke({
                "context": context,
                "question": job_description
            })

            st.markdown("---")
            st.subheader(uploaded_file.name)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("ATS Score", f"{ats_score}%")
            with col2:
                st.metric("Status", status)

            st.write("### Matching Skills")
            st.write(matching_skills)

            st.write("### Missing Skills")
            st.write(missing_skills)

            st.write("### AI Analysis")
            st.markdown(llm_result)

            results.append({
                "Resume": uploaded_file.name,
                "Score": ats_score,
                "Status": status
            })

            # Clean up temp file
            os.unlink(pdf_path)

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)

    st.markdown("---")
    st.subheader("🏆 Resume Ranking")
    st.dataframe(df, use_container_width=True)
