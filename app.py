import streamlit as st
import base64

from langgraph_backend import chatbot, ingest_pdf
from langchain_core.messages import HumanMessage


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)


CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}


# ---------------- Sidebar ----------------

with st.sidebar:

    st.title("🤖 AI Document Assistant")

    st.write(
        "Chat with PDF + Screenshot using RAG + LangGraph"
    )

    st.divider()


    # PDF Upload

    uploaded_files = st.file_uploader(
        "📂 Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )


    if uploaded_files:

        if st.button("🚀 Process Documents"):

            for file in uploaded_files:

                file_bytes = file.read()

                ingest_pdf(
                    file_bytes,
                    thread_id="thread-1",
                    filename=file.name
                )


            st.session_state["pdf_uploaded"] = True

            st.success(
                "Documents processed successfully ✅"
            )


    st.divider()


    # Screenshot Upload Feature

    st.subheader("🖼 Upload Screenshot")


    uploaded_image = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"]
    )


    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Screenshot",
            use_container_width=True
        )


        st.session_state["uploaded_image"] = uploaded_image


    st.divider()


    # New Chat

    if st.button("🆕 New Chat"):

        st.session_state["message_history"] = []

        st.rerun()


    # Clear Chat

    if st.button("🗑 Clear Chat"):

        st.session_state["message_history"] = []

        st.rerun()



# ---------------- Main Chat ----------------


st.title("🤖 AI Document Chatbot")

st.caption(
    "Ask questions from PDF or analyze screenshots"
)


st.divider()



if "message_history" not in st.session_state:

    st.session_state["message_history"] = []



# Show previous messages

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])



# Chat Input

user_input = st.chat_input(
    "Ask something..."
)



if user_input:


    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )


    with st.chat_message("user"):

        st.write(user_input)



    with st.chat_message("assistant"):


        with st.spinner("Thinking..."):


            # -------- IMAGE MESSAGE ---------

            if "uploaded_image" in st.session_state:


                image_file = st.session_state["uploaded_image"]


                image_bytes = image_file.getvalue()


                image_base64 = base64.b64encode(
                    image_bytes
                ).decode()



                message = HumanMessage(

                    content=[

                        {
                            "type": "text",
                            "text": user_input
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                f"data:image/jpeg;base64,{image_base64}"
                            }
                        }

                    ]

                )


            # -------- TEXT MESSAGE ---------

            else:


                message = HumanMessage(
                    content=user_input
                )



            response = chatbot.invoke(

                {
                    "messages":[message]
                },

                config=CONFIG

            )



            ai_message = response["messages"][-1].content



            st.write(ai_message)



    st.session_state["message_history"].append(

        {
            "role":"assistant",
            "content":ai_message
        }

    )