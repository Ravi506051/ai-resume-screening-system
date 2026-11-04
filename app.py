import streamlit as st, pandas as pd, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("AI Resume Screening Assistant"); st.caption("Local text similarity only; a human must make all hiring decisions.")
job=st.text_area("Job description",height=180); files=st.file_uploader("Upload .txt resumes",type="txt",accept_multiple_files=True); manual=st.text_area("Or paste a resume")
if st.button("Match resumes"):
 docs=[(f.name,f.getvalue().decode("utf-8",errors="ignore")) for f in (files or [])]
 if manual.strip(): docs.append(("Pasted resume",manual))
 if not job.strip() or not docs: st.error("Provide a job description and at least one resume.")
 else:
  vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),max_features=5000); mat=vec.fit_transform([job]+[x[1] for x in docs]); sims=cosine_similarity(mat[:1],mat[1:]).ravel()
  rows=[]
  for (name,txt),s in zip(docs,sims):
   terms=sorted(set(re.findall(r"[a-zA-Z]{3,}",job.lower())) & set(re.findall(r"[a-zA-Z]{3,}",txt.lower())))
   rows.append({"resume":name,"similarity":round(float(s),3),"matched_terms":", ".join(terms[:20])})
  out=pd.DataFrame(rows).sort_values("similarity",ascending=False); st.dataframe(out,use_container_width=True); st.download_button("Download report",out.to_csv(index=False),"resume_matches.csv")
