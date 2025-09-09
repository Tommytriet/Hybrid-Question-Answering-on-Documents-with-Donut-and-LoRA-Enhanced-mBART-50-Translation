import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
# -----------------------------
# Load Donut DocVQA
# -----------------------------
@st.cache_resource
def load_donut():
    donut_model_id = "Tommynguyen02/donut-base-finetune"
    processor = DonutProcessor.from_pretrained(donut_model_id)
    model = VisionEncoderDecoderModel.from_pretrained(donut_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device

donut_processor, donut_model, device = load_donut()

def donut_qa(image, question_en):
    img = image.convert("RGB")
    task_prompt = f"<s_docvqa><s_question>{question_en}</s_question><s_answer>"
    pixel_values = donut_processor(img, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = donut_processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = donut_model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        eos_token_id=donut_processor.tokenizer.eos_token_id,
        pad_token_id=donut_processor.tokenizer.pad_token_id,
    )
    return donut_processor.batch_decode(outputs, skip_special_tokens=True)[0]

# -----------------------------
# Load mBART50 fine-tuned (EN-VI)
# -----------------------------
@st.cache_resource
def load_mbart():
    base_model_id = "facebook/mbart-large-50-many-to-many-mmt"   # model gốc
    lora_model_id = "Tommynguyen02/mbart50-LoRA"                 # repo LoRA adapter của bạn

    # Load base model
    model = MBartForConditionalGeneration.from_pretrained(base_model_id)

    # Gắn LoRA adapter
    model = PeftModel.from_pretrained(model, lora_model_id)

    # Load tokenizer từ base model
    tokenizer = MBart50TokenizerFast.from_pretrained(base_model_id)

    # Chọn thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device

# Load model + tokenizer
trans_tokenizer, trans_model, device = load_mbart()

def vi_to_en(text):
    trans_tokenizer.src_lang = "vi_VN"
    inputs = trans_tokenizer(text, return_tensors="pt").to(device)
    gen_ids = trans_model.generate(
        **inputs,
        forced_bos_token_id=trans_tokenizer.lang_code_to_id["en_XX"]
    )
    return trans_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

def en_to_vi(text):
    trans_tokenizer.src_lang = "en_XX"
    inputs = trans_tokenizer(text, return_tensors="pt").to(device)
    gen_ids = trans_model.generate(
        **inputs,
        forced_bos_token_id=trans_tokenizer.lang_code_to_id["vi_VN"]
    )
    return trans_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Hệ thống Dịch Hóa Đơn (Donut + mBART50)")
st.write("Upload ảnh hóa đơn + đặt câu hỏi bằng tiếng Việt.")

uploaded_file = st.file_uploader("Chọn ảnh hóa đơn", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # 👉 Hiển thị ảnh trước
    st.image(image, caption="Ảnh hóa đơn đã upload", use_container_width=True)

    # 👉 Sau đó mới nhập câu hỏi
    question_vi = st.text_input("Nhập câu hỏi bằng TIẾNG VIỆT")

    if question_vi:
        # Step 1: dịch câu hỏi sang tiếng Anh
        question_en = vi_to_en(question_vi)

        # Step 2: hỏi Donut (bằng tiếng Anh)
        answer_en = donut_qa(image, question_en)

        # Step 3: dịch câu trả lời sang tiếng Việt
        answer_vi = en_to_vi(answer_en)

        st.subheader("📝 Kết quả")
        st.write(answer_vi)

        with st.expander("Chi tiết (debug)"):
            st.json({
                "Câu hỏi (VI)": question_vi,
                "Question (EN)": question_en,
                "Answer (EN)": answer_en,
                "Trả lời (VI)": answer_vi
            })



