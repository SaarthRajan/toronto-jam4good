import streamlit as st

# Set Page Config -> title, logo, layout (centred by default)
st.set_page_config(
    page_title="InsurEase", 
    page_icon="⛑️",
    initial_sidebar_state="collapsed" # hiding the sidebar
)

# Used to remove the streamlit branding - and format other stuff (also check configure.toml for theme)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")


# Constants

def title_ui():

    # might remove icon and add a logo
    # might even remove the title - just have the image

    # # to center the image
    # cols = st.columns(4) # change accordingly
    # with cols[1]:
        # InsurEase logo here
        # st.image("./space_apps_logo.png", width=300)

    st.markdown("""
        <h1 style='text-align: center; color: #FF3333; font-family: sans-serif;'>
            InsurEase ⛑️
        </h1>
        <p style='text-align: center; color: #FF3333; font-family: sans-serif;'>Navigate the Complexity of Insurance with Ease</p>
    """, unsafe_allow_html=True)

def upload_ui():
    # Upload Your Insurance Information PDF - Upload Button
    uploaded_pdf = st.file_uploader("Upload Your Insurance Information PDF", type="pdf")

    if st.button("Generate Benefits"):
        if uploaded_pdf is not None:
            st.success("Processing PDF...")
            # Add your PDF processing logic here - with st
            # e.g., extract_text(uploaded_pdf)
        else:
            st.warning("Please upload a PDF before generating.")

def chat_ui():
    st.header("Chat", anchor=False)
    messages = st.container(border=True, height=600)
    if prompt := st.chat_input("Ask Questions about your Insurance", key="prompt"):
        print("Message Sent")

def benefits_ui():
    st.header("Available Benefits", anchor=False)

def main():

    title_ui()

    upload_ui()

    col1, col2 = st.columns(2, gap="large") #, vertical_alignment="center"

    with col2 :
        chat_ui()
    
    with col1 :
        benefits_ui()


if __name__ == "__main__" :
    main()