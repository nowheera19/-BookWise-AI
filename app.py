
import streamlit as st
import torch
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./BookWiseAI_Adapter"

st.set_page_config(
    page_title="BookWise AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 BookWise AI")
st.write("Your personal college study assistant.")
st.info("Upload a PDF and ask questions about its content.")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        fix_mistral_regex=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()
    return tokenizer, model


def generate_answer(context, question, tokenizer, model):
    prompt = f"""
You are BookWise AI, a college study assistant.
Answer using simple English.
Use the PDF content to answer the question.
If the answer is not present in the PDF, say so honestly.

PDF CONTENT:
{context[:7000]}

QUESTION:
{question}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[-1]

    answer = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return answer.strip()


uploaded_file = st.file_uploader(
    "Upload your study notes",
    type=["pdf"]
)

pdf_text = ""

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)

        pdf_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        if pdf_text.strip():
            st.success(
                f"PDF read successfully! Pages: {len(reader.pages)}"
            )

            with st.expander("Preview PDF text"):
                st.text(pdf_text[:2000])
        else:
            st.warning(
                "No readable text found in this PDF."
            )

    except Exception as error:
        st.error(f"PDF reading failed: {error}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

question = st.chat_input("Ask something about your PDF...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    st.chat_message("user").write(question)

    if not pdf_text.strip():
        answer = "Please upload a readable PDF first."
        st.chat_message("assistant").write(answer)
    else:
        with st.chat_message("assistant"):
            with st.spinner(
                "Loading AI model and generating your answer..."
            ):
                try:
                    tokenizer, model = load_model()
                    answer = generate_answer(
                        pdf_text,
                        question,
                        tokenizer,
                        model
                    )
                    st.write(answer)

                except Exception as error:
                    answer = f"Something went wrong: {error}"
                    st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
