# Travel Lens AI

Travel Lens AI 是一個旅行照片策展與明信片生成工具。

使用者可以上傳旅行照片、套用 OpenCV 濾鏡、讓 Mistral 多模態模型生成旅行短文與社群文案，最後下載旅行明信片 PNG。

目前支援旅行心情選擇與多種明信片模板，使用者可以用不同情緒與版型輸出更有個人風格的旅行卡片。

## Tools

- OpenCV / CV2：照片濾鏡、亮度分析、對比分析、色調判斷。
- Mistral API：圖片理解與旅行文案生成。
- Streamlit：互動式網頁、照片上傳、預覽與下載。

## Features

- 旅行心情：放鬆、冒險、浪漫、懷舊、熱鬧、療癒。
- 明信片樣式：極簡白框、拍立得、復古郵戳、雜誌封面。

## Streamlit Cloud Secrets

Deploy 時請在 Streamlit Cloud 的 Secrets 欄位加入：

```toml
MISTRAL_API_KEY = "your_mistral_api_key"
```
