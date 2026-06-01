# streamlit run final_project/travel_lens_ai.py
import base64
import io
import os
import re
from datetime import date

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont


load_dotenv(override=True)

MISTRAL_MODEL = "pixtral-12b-latest"

STYLE_GUIDES = {
    "文青感": "文字有畫面感、溫柔、細膩，但不要過度浮誇。",
    "旅遊部落格": "像旅遊部落客介紹景點，清楚、有資訊感、自然好讀。",
    "幽默口吻": "輕鬆、有趣、像朋友分享旅行小插曲。",
    "極簡質感": "句子短、乾淨、有留白感，像高質感相簿文案。",
    "Instagram 風": "適合社群分享，節奏明快，帶一點生活感。",
}

FILTERS = {
    "原圖": "保留照片原始色彩。",
    "暖色旅行": "提升暖色與亮度，適合陽光、街景與度假照片。",
    "復古底片": "加入底片感色調、淡淡暗角與懷舊氛圍。",
    "黑白文藝": "轉成高對比黑白，適合街拍、人像與建築。",
    "柔和明信片": "降低對比、增加柔光，讓照片更像旅行卡片。",
    "城市高對比": "提升清晰度、飽和度與對比，適合夜景與城市感照片。",
}


def get_mistral_client():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("MISTRAL_API_KEY")
        except (FileNotFoundError, KeyError):
            api_key = None

    if not api_key:
        st.error("系統尚未完成 API 設定，請聯絡管理者。")
        st.stop()

    return OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
    )


def uploaded_file_to_bgr(uploaded_file):
    image_array = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        st.error("無法讀取這張圖片，請改用 JPG、PNG 或 WEBP。")
        st.stop()
    return image


def bgr_to_pil(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def pil_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_to_data_url(image_bytes, mime_type="image/jpeg"):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def apply_filter(image_bgr, filter_name):
    if filter_name == "原圖":
        return image_bgr.copy()

    image = image_bgr.astype(np.float32)

    if filter_name == "暖色旅行":
        image[:, :, 2] *= 1.12
        image[:, :, 1] *= 1.04
        image[:, :, 0] *= 0.92
        image = cv2.convertScaleAbs(np.clip(image, 0, 255), alpha=1.06, beta=8)
        return image

    if filter_name == "復古底片":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        sepia = np.array(
            [
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ]
        )
        vintage_rgb = np.clip(rgb @ sepia.T, 0, 255).astype(np.uint8)
        vintage_bgr = cv2.cvtColor(vintage_rgb, cv2.COLOR_RGB2BGR)

        rows, cols = vintage_bgr.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 1.6)
        kernel_y = cv2.getGaussianKernel(rows, rows / 1.6)
        vignette = kernel_y @ kernel_x.T
        vignette = vignette / vignette.max()
        vintage = vintage_bgr.astype(np.float32)
        vintage *= vignette[:, :, np.newaxis] * 0.45 + 0.65
        return np.clip(vintage, 0, 255).astype(np.uint8)

    if filter_name == "黑白文藝":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.convertScaleAbs(gray, alpha=1.12, beta=4)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if filter_name == "柔和明信片":
        blur = cv2.GaussianBlur(image_bgr, (0, 0), 8)
        soft = cv2.addWeighted(image_bgr, 0.72, blur, 0.28, 0)
        soft = cv2.convertScaleAbs(soft, alpha=0.94, beta=18)
        return soft

    if filter_name == "城市高對比":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge((l_channel, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.18
        hsv[:, :, 2] *= 1.03
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image_bgr.copy()


def analyze_image_with_cv(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    b_mean, g_mean, r_mean = cv2.mean(image_bgr)[:3]
    height, width = image_bgr.shape[:2]

    if r_mean - b_mean > 10:
        color_tone = "暖色調"
    elif b_mean - r_mean > 10:
        color_tone = "冷色調"
    else:
        color_tone = "中性色調"

    if brightness < 85:
        brightness_label = "偏暗"
    elif brightness > 175:
        brightness_label = "偏亮"
    else:
        brightness_label = "亮度適中"

    if contrast < 38:
        contrast_label = "柔和低對比"
    elif contrast > 72:
        contrast_label = "高對比"
    else:
        contrast_label = "自然對比"

    return {
        "尺寸": f"{width} x {height}",
        "亮度": brightness_label,
        "對比": contrast_label,
        "色調": color_tone,
        "平均亮度": f"{brightness:.1f}",
        "對比數值": f"{contrast:.1f}",
    }


def build_prompt(location, travel_date, style, platforms, extra_context, filter_name, cv_summary):
    platform_text = "、".join(platforms)
    location_text = location.strip() if location.strip() else "未提供，請根據照片推測即可"
    context_text = extra_context.strip() if extra_context.strip() else "無"
    cv_text = "；".join(f"{key}: {value}" for key, value in cv_summary.items())

    return f"""
你是一位旅行攝影策展助理，請根據使用者上傳的旅行照片產生內容。

請使用繁體中文，並以 Markdown 格式輸出。請不要編造過於確定的地點資訊；如果照片無法判斷，就用「可能」、「看起來像」描述。

使用者資訊：
- 地點：{location_text}
- 日期：{travel_date}
- 文字風格：{style}
- 風格要求：{STYLE_GUIDES[style]}
- 使用者想輸出的平台：{platform_text}
- 選擇的照片濾鏡：{filter_name}
- OpenCV 影像分析結果：{cv_text}
- 補充背景：{context_text}

請依照以下格式回答：

## 照片內容分析
- 場景：
- 主要元素：
- 色調與氛圍：
- 可能的旅行亮點：

## 旅行短文
請寫 100 到 150 字，像是可以放在個人旅行相簿或作品集裡的短文。

## Instagram Caption
請寫一段自然、適合社群分享的 caption。

## Facebook 貼文
請寫一段比 Instagram 稍微完整的 Facebook 貼文。

## Hashtags
請提供 8 到 12 個 hashtag，混合中文與英文。

## 明信片短句
請給一句 20 字以內、適合放在照片上的文字。
"""


def generate_travel_content(client, prompt, image_data_url):
    response = client.chat.completions.create(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def extract_postcard_sentence(text):
    match = re.search(r"##\s*明信片短句\s*(.+)", text, re.S)
    if not match:
        return "把這一刻，收藏成旅行的光"

    sentence = match.group(1).strip().splitlines()[0]
    sentence = re.sub(r"^[\-*\d.\s]+", "", sentence).strip()
    return sentence[:28] or "把這一刻，收藏成旅行的光"


def get_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def wrap_text(text, max_chars):
    lines = []
    current = ""
    for char in text:
        current += char
        if len(current) >= max_chars:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines


def make_postcard(image, title, location, travel_date):
    postcard_width = 1080
    margin = 72
    photo_width = postcard_width - margin * 2

    image = image.convert("RGB")
    ratio = photo_width / image.width
    photo_height = int(image.height * ratio)
    resized_photo = image.resize((photo_width, photo_height), Image.LANCZOS)

    title_font = get_font(46, bold=True)
    meta_font = get_font(26)
    footer_font = get_font(22)

    title_lines = wrap_text(title, 16)
    text_height = 72 + len(title_lines) * 58 + 72
    postcard_height = photo_height + margin * 2 + text_height

    canvas = Image.new("RGB", (postcard_width, postcard_height), "#f8f5ef")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(resized_photo, (margin, margin))
    border_color = "#ffffff"
    draw.rectangle(
        [margin - 14, margin - 14, margin + photo_width + 14, margin + photo_height + 14],
        outline=border_color,
        width=14,
    )

    text_y = margin + photo_height + 56
    for line in title_lines:
        draw.text((margin, text_y), line, fill="#1f2933", font=title_font)
        text_y += 58

    meta = " / ".join(part for part in [location.strip(), str(travel_date)] if part)
    if meta:
        draw.text((margin, text_y + 10), meta, fill="#64748b", font=meta_font)

    draw.text(
        (margin, postcard_height - 52),
        "Travel Lens AI",
        fill="#94a3b8",
        font=footer_font,
    )

    return canvas


st.set_page_config(
    page_title="Travel Lens AI",
    page_icon="📷",
    layout="wide",
)

st.title("Travel Lens AI：旅行照片策展助理")
st.caption("上傳旅行照片，套用照片濾鏡，讓 AI 幫你生成旅行故事與明信片。")

with st.sidebar:
    st.header("創作設定")
    filter_name = st.selectbox("照片濾鏡", list(FILTERS.keys()))
    st.caption(FILTERS[filter_name])
    style = st.selectbox("文案風格", list(STYLE_GUIDES.keys()))
    location = st.text_input("地點", placeholder="例如：京都、台南、巴黎")
    travel_date = st.date_input("日期", value=date.today())
    platforms = st.multiselect(
        "輸出用途",
        ["Instagram", "Facebook", "旅行相簿", "個人網站"],
        default=["Instagram", "Facebook", "旅行相簿"],
    )
    extra_context = st.text_area(
        "補充背景",
        placeholder="例如：這是畢業旅行、第一次自助旅行、和朋友一起去等",
        height=120,
    )

uploaded_file = st.file_uploader(
    "請上傳一張旅行照片",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    original_bgr = uploaded_file_to_bgr(uploaded_file)
    filtered_bgr = apply_filter(original_bgr, filter_name)
    filtered_image = bgr_to_pil(filtered_bgr)
    filtered_png = pil_to_png_bytes(filtered_image)
    cv_summary = analyze_image_with_cv(filtered_bgr)

    left_col, right_col = st.columns([1.05, 0.95])

    with left_col:
        st.subheader("照片預覽")
        tab_original, tab_filtered = st.tabs(["原圖", "濾鏡後"])
        with tab_original:
            st.image(uploaded_file, caption=uploaded_file.name, width="stretch")
        with tab_filtered:
            st.image(filtered_image, caption=f"{filter_name} 濾鏡", width="stretch")
            st.download_button(
                "下載濾鏡照片",
                data=filtered_png,
                file_name="travel_lens_filtered.png",
                mime="image/png",
            )

    with right_col:
        st.subheader("OpenCV 影像分析")
        metric_cols = st.columns(3)
        metric_cols[0].metric("亮度", cv_summary["亮度"])
        metric_cols[1].metric("對比", cv_summary["對比"])
        metric_cols[2].metric("色調", cv_summary["色調"])
        st.caption(f"圖片尺寸：{cv_summary['尺寸']}；平均亮度：{cv_summary['平均亮度']}；對比數值：{cv_summary['對比數值']}")

        st.subheader("產生旅行內容")
        st.write("系統會結合照片、OpenCV 分析結果與你的設定，產生旅行故事和社群文案。")

        if st.button("開始生成", type="primary"):
            prompt = build_prompt(
                location=location,
                travel_date=travel_date,
                style=style,
                platforms=platforms,
                extra_context=extra_context,
                filter_name=filter_name,
                cv_summary=cv_summary,
            )

            image_data_url = bytes_to_data_url(uploaded_file.getvalue(), uploaded_file.type or "image/jpeg")
            client = get_mistral_client()

            with st.spinner("AI 正在閱讀照片並撰寫旅行故事..."):
                result = generate_travel_content(client, prompt, image_data_url)

            st.session_state["travel_result"] = result
            st.session_state["postcard_image"] = filtered_png
            st.session_state["postcard_title"] = extract_postcard_sentence(result)

if st.session_state.get("travel_result"):
    st.divider()
    st.markdown(st.session_state["travel_result"])

    st.divider()
    st.subheader("明信片輸出")
    postcard_title = st.text_input(
        "明信片文字",
        value=st.session_state.get("postcard_title", "把這一刻，收藏成旅行的光"),
    )
    postcard_image = Image.open(io.BytesIO(st.session_state["postcard_image"]))
    postcard = make_postcard(
        image=postcard_image,
        title=postcard_title,
        location=location,
        travel_date=travel_date,
    )
    postcard_png = pil_to_png_bytes(postcard)

    preview_col, download_col = st.columns([1, 1])
    with preview_col:
        st.image(postcard, caption="AI 旅行明信片", width="stretch")
    with download_col:
        st.write("你可以下載濾鏡後的旅行明信片，放進簡報、作品集或個人網站。")
        st.download_button(
            "下載明信片 PNG",
            data=postcard_png,
            file_name="travel_lens_postcard.png",
            mime="image/png",
        )
        st.download_button(
            "下載文字結果",
            data=st.session_state["travel_result"],
            file_name="travel_lens_ai_result.md",
            mime="text/markdown",
        )
else:
    st.info("先上傳一張照片，選擇濾鏡與文案風格，再按「開始生成」。")
