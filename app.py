import streamlit as st
import speech_recognition as sr
import pandas as pd
import os
from translator import analyze_hindi_sentence
from translator import extract_token_info
from translator import plot_dependency_tree
from translator import get_cleaned_word_tags
from translator import change_dict_order
from translator import reorder_word_signs
from translator import reorder_neg_wq_words
from translator import remove_stopwords
from translator import extract_sign_words
from translator import load_isl_dictionary
from translator import get_lemmatizer_and_iwn
from translator import get_special_video_dict
from translator import get_synonym_substituted_list
from translator import init_google_translator
from translator import get_isl_hindi_english_dict
from translator import search_videos
from translator import merge_videos_opencv
from translator import merge_videos_moviepy

st.markdown(
    """
    <style>
    /* Change entire page background */
    .stApp {
        background-color: #f0f8ff;  /* light blue, you can use any hex color */
    }

    body, 
    .css-18e3th9,  /* Main app container */
    .css-1v3fvcr,   /* Markdown text */
    .css-1d391kg,   /* Headers */
    .css-1nwbyjy,   /* Subheaders */
    .stTextInput>div>div>input, /* Input box text */
    .css-1q8dd3e,   /* Buttons and labels */
    .css-1lcbmhc,   /* Other text elements */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #000000 !important;  /* Navy Blue */
    }
    
    /* Change background and style of text input box */
    .stTextInput>div>div>input {
        background-color: #E6F2FF;  /* lighter blue input bg */
        border: 2px solid #3399ff;
        border-radius: 8px;
        padding: 12px;
        font-size: 18px;
        # color: #000000;
    }
    
    /* Change placeholder text color */
    .stTextInput>div>div>input::placeholder {
        # color: #666666;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.set_page_config(page_title="Hindi to ISL System", layout="centered")

st.title("Hindi to Indian Sign Language (ISL) Translation System")

# hindi_sentence = st.text_input("Enter the Hindi sentence:")

def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="hi-IN")  # Recognize Hindi speech
        return text
    except sr.UnknownValueError:
        return "❌ Sorry, could not understand audio"
    except sr.RequestError:
        return "⚠️ Could not connect to the speech recognition service"

# --- UI ---
st.markdown("### 🗣️ Enter the Hindi sentence (Type or Speak):")

# Initialize session state to hold spoken input
if 'spoken_text' not in st.session_state:
    st.session_state.spoken_text = ""

# Input from text box (defaults to spoken text if available)
hindi_sentence = st.text_input("📝 Type here or use voice:", value=st.session_state.spoken_text)

# Speech button
if st.button("🎤 Speak"):
    spoken = recognize_speech()
    st.session_state.spoken_text = spoken
    st.rerun()
 # Rerun to update text input with spoken text

# Use hindi_sentence in your logic below this line
st.markdown(f"🗣️ **Hindi Sentence Provided:** {hindi_sentence}")


# Step 2: Proceed only if input is provided
if hindi_sentence.strip():
    # Analyze sentence
    document = analyze_hindi_sentence(hindi_sentence)
    df_tokens, root_word = extract_token_info(document)

    # Collapsible section for text analysis and dependency parse
    with st.expander("📊 Show Text Content & Dependency Parse Tree"):
        st.subheader("Token Information")
        st.dataframe(df_tokens)
        st.markdown(f"**Root (highest ranked) word:** {root_word}")

        font_path = os.path.join("utility", "NotoSansDevanagari.ttf")
        fig = plot_dependency_tree(df_tokens, font_path)
        st.pyplot(fig)

    # Process for ISL translation
    cleaned_word_tags = get_cleaned_word_tags(document)
    
    word_sign_order = cleaned_word_tags.copy()
    word_sign_order = reorder_word_signs(cleaned_word_tags, word_sign_order)
    word_sign_order = reorder_neg_wq_words(word_sign_order)
    
    stopword_removed_list = remove_stopwords(word_sign_order)
    sign_words_list = extract_sign_words(stopword_removed_list)

    

    # Dictionary and translation
    cleaned_dict = load_isl_dictionary()
    lemmatizer, iwn = get_lemmatizer_and_iwn()
    special_videos = get_special_video_dict()
    translator = init_google_translator()

    synonym_substituted_list = get_synonym_substituted_list(
        sign_words_list, cleaned_dict, translator, lemmatizer, iwn, special_videos
    )

    df_synonyms = pd.DataFrame(synonym_substituted_list, columns=["Hindi Word", "POS Tag", "ISL Dictionary Tag"])
    # synonym_substituted_list.columns = ["Hindi Word", "POS Tag", "ISL Dictionary Tag"]
    
    with st.expander("✅ Show Final ISL Order After Synonym Substitution and Stop Word Removal"):
        st.dataframe(df_synonyms)
    
    isl_hindi_english_dict = get_isl_hindi_english_dict(cleaned_dict)
    video_paths = search_videos(synonym_substituted_list)

    # Merge and play video
    merge_videos_moviepy(video_paths, "merged_isl.mp4")

    with open("merged_isl.mp4", "rb") as video_file:
        video_bytes = video_file.read()
        st.markdown("### 🎬 ISL Video:")
        st.video(video_bytes)

else:
    st.info("👋 Please enter a Hindi sentence to get the ISL translation.")
