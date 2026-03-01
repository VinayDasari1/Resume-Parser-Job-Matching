import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Parser & Matcher", layout="centered")

st.title("📄 Resume Parser & Job-Role Matching AI Engine")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

SKILL_DATABASE = [
    "Python", "Java", "C", "Spring Boot", "React",
    "MySQL", "PostgreSQL", "Machine Learning",
    "Deep Learning", "NLP", "Cloud Computing",
    "HTML", "CSS", "JWT", "REST API"
]


def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        if len(line.split()) <= 4 and "@" not in line:
            return line.strip()
    return "Not Found"

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    match = re.search(r"\b\d{10}\b", text)
    return match.group(0) if match else "Not Found"

def extract_skills(text):
    found_skills = []
    for skill in SKILL_DATABASE:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here")

if st.button("Analyze Resume"):

    if uploaded_resume is None or job_description.strip() == "":
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Parsing and Matching..."):

            resume_text = extract_text_from_pdf(uploaded_resume)

            name = extract_name(resume_text)
            email = extract_email(resume_text)
            phone = extract_phone(resume_text)
            resume_skills = extract_skills(resume_text)

            job_skills = extract_skills(job_description)
            matched_skills = list(set(resume_skills) & set(job_skills))

            skill_ratio = f"{len(matched_skills)} / {len(job_skills)}" if job_skills else "0"

            resume_embedding = model.encode([resume_text])
            job_embedding = model.encode([job_description])

            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
            score = round(similarity * 100, 2)

        st.success("Analysis Complete!")

        
        st.subheader("📌 Extracted Resume Information")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")
        st.write(f"**Skills Found:** {', '.join(resume_skills) if resume_skills else 'None'}")

        
        st.subheader("🎯 Matched Skills with Job Description")
        if matched_skills:
            st.write(", ".join(matched_skills))
            st.write(f"**Skill Match Ratio:** {skill_ratio}")
        else:
            st.write("No direct skill matches found.")

    
        st.subheader("📊 Semantic Match Score")
        st.write(f"### {score}%")

        if score > 75:
            st.write("🟢 Strong Match")
        elif score > 50:
            st.write("🟡 Moderate Match")
        else:
            st.write("🔴 Weak Match")

        st.subheader("📄 Resume Preview")
        st.write(resume_text[:1000])
