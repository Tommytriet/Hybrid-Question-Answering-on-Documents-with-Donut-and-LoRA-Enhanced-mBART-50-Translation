import streamlit as st
import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel, DonutProcessor,
    MBartForConditionalGeneration, MBart50TokenizerFast, BitsAndBytesConfig
)
from peft import PeftModel
import gc

# -----------------------------
# Helper: Clear model để giảm RAM
# -----------------------------
def clear_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------
# Load Donut DocVQA (lazy, quantized)
# -----------------------------
@st.cache_resource
def load_donut():
    donut_model_id = "naver-clova-ix/donut-base-finetuned-docvqa"

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = DonutProcessor.from_pretrained(donut_model_id)
    model = VisionEncoderDecoderModel.from_pretrained(
        donut_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    return processor, model

def donut_qa(image, question_en, processor, model):
    img = image.convert("RGB")
    task_prompt = f"<s_docvqa><s_question>{question_en}</s_question><s_answer>"

    pixel_values = processor(img, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt", legacy=False
    ).input_ids

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

# -----------------------------
# Load mBART50 fine-tuned (EN-VI) (lazy, quantized)
# -----------------------------
@st.cache_resource
def load_mbart():
    base_model_id = "facebook/mbart-large-50-many-to-many-mmt"
    lora_model_id = "Tommynguyen02/mbart50-LoRA"

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = MBartForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_model_id)

    tokenizer = MBart50TokenizerFast.from_pretrained(base_model_id)
    return tokenizer, model

def vi_to_en(text, tokenizer, model):
    tokenizer.src_lang = "vi_VN"
    inputs = tokenizer(text, return_tensors="pt", legacy=False)
    gen_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

def en_to_vi(text, tokenizer, model):
    tokenizer.src_lang = "en_XX"
    inputs = tokenizer(text, return_tensors="pt", legacy=False)
    gen_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"]
    )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Hệ thống Dịch Hóa Đơn (Donut + mBART50)")
st.write("Upload ảnh hóa đơn + đặt câu hỏi bằng tiếng Việt.")

uploaded_file = st.file_uploader("Chọn ảnh hóa đơn", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Hiển thị ảnh
    st.image(image, caption="Ảnh hóa đơn đã upload", width="stretch")

    # Nhập câu hỏi
    question_vi = st.text_input("Nhập câu hỏi bằng TIẾNG VIỆT")

    if question_vi:
        # Lazy load Donut + mBART
        with st.spinner("Đang load model Donut..."):
            donut_processor, donut_model = load_donut()
        with st.spinner("Đang load model mBART..."):
            trans_tokenizer, trans_model = load_mbart()

        # Step 1: dịch câu hỏi sang tiếng Anh
        question_en = vi_to_en(question_vi, trans_tokenizer, trans_model)

        # Step 2: hỏi Donut (bằng tiếng Anh)
        answer_en = donut_qa(image, question_en, donut_processor, donut_model)

        # Step 3: dịch câu trả lời sang tiếng Việt
        answer_vi = en_to_vi(answer_en, trans_tokenizer, trans_model)

        st.subheader("📝 Kết quả")
        st.write(answer_vi)

        with st.expander("Chi tiết (debug)"):
            st.json({
                "Câu hỏi (VI)": question_vi,
                "Question (EN)": question_en,
                "Answer (EN)": answer_en,
                "Trả lời (VI)": answer_vi
            })

        # Optional: clear models sau khi dùng để giảm RAM
        clear_model(donut_model)
        clear_model(trans_model)
