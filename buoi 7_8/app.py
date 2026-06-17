import os
import pandas as pd
from dotenv import load_dotenv
from google import genai 
from google.genai import types
import streamlit as st

# =============================
# Load menu của quán
# ===========================
menu_df = pd.read_csv("menu.csv")

# =============================
# Cấu hình chatbot (Gemini API)
# =============================
MODEL_VERSION = "gemini-2.5-flash"

# =============================
# Prompt
# ==========================
SYSTEM_INTRO = f"""
    Bạn tên là PhoBot, một trợ lý AI có nhiệm vụ hỗ trợ giải đáp thông tin cho khách hàng đến nhà hàng Viet Cuisine.
    Các chức năng mà bạn hỗ trợ gồm:
    1. Giới thiệu nhà hàng Viet Cuisine: là một nhà hàng thành lập bởi người Việt, ở địa chỉ 329 Scottmouth, Georgia, USA
    2. Giới thiệu menu của nhà hàng, gồm các món: {', '.join(menu_df['name'].to_list())}.
    3. Hỏi đáp về món ăn trong menu, ví dụ như thành phần món ăn, cách chế biến (dựa trên thông tin trong menu: {menu_df.to_dict(orient='records')})
    Ngoài ba chức năng trên, bạn không hỗ trợ chức năng nào khác. Đối với các câu hỏi ngoài chức năng mà bạn hỗ trợ, trả lời bằng 'Tôi đang không hỗ trợ chức năng này. Xin liên hệ nhân viên nhà hàng qua hotline 318-237-3870 để được trợ giúp.'
    Hãy có thái độ thân thiện và lịch sự khi nói chuyện với khác hàng, vì khách hàng là thượng đế.
"""

INITIAL_MESS = """
Xin chào!

Tôi là PhoBot, trợ lý trực tuyến của nhà hàng Việt Cusine

Tôi có thể hộ trợ:
- Giới thiệu nhà hàng
- Xem menu món ăn
- Hỏi đáp về món ăn trong menu

Bạn cần tới giúp gì?
"""

# =============================
#  Hàm tạo chatbot
# =============================
def restaurant_chatbot():
    # config cho trang web
    st.set_page_config(
        page_title="PhoBot - Trợ lý nhà hàng Viet Cuisine",
        page_icon="🍜",
        layout="wide",
    )

    st.title("PhoBot - Viet Cuisine")

    # slidebar (1 trang nhap API key, 1 trang chat)
    with st.sidebar:
        st.header("API key config")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Nhập API key của bạn tại đây",
        )

        st.divider()


        st.markdown("""
            ### Chức năng hỗ trợ
            - Giới thiệu nhà hàng
            - Xem menu món ăn
            - Hỏi đáp về món ăn trong menu
        """)

    if not api_key:
        st.warning("Vui lòng nhập Gemini API key để bắt đầu sử dụng.")
        st.stop()

    # có api key -> tạo chat
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Lỗi API key: {e}")
        st.stop()

        # tạo session chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": INITIAL_MESS}]
        
    # hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # nhập câu hỏi
    prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

    if prompt:
        # lưu vào lịch sử chat
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        # hiển thị tin nhắn lên màn hình
        with st.chat_message("user"):
            st.markdown(prompt)

        # tối ưu hóa (nếu cần menu thì trả về luôn mà không cần gọi API)
        chat_response = ""

        if any(word in prompt.lower() for word in ["menu", "thực đơn"]):
            chat_response = "## Dưới đây là menu món ăn của nhà hàng Viet Cuisine:\n\n"
            for _, row in menu_df.iterrows():
                chat_response += f'### {row['name']}\n{row['description']}\n'
            else:
                # Gọi API để trả lời
                try:
                    response = client.models.generate_content(
                        model = MODEL_VERSION,
                        contents = prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INTRO,
                            temperature = 0.2,
                            max_output_tokens = 2000,
                        )
                    )
                    chat_response = response.output[0].content[0].text
                except Exception as e:
                    st.error(f"Lỗi khi gọi API: {e}")

        # hiển thị lịch sử chat
        with st.chat_message("assistant"):
                    st.markdown(chat_response)

        # lưu lịch sử chat
        st.session_state.chat_history.append(
            {"role": "assistant", "content": chat_response}
        )


# =============================
# Chạy app
# =============================
if __name__ == "__main__":
    restaurant_chatbot()