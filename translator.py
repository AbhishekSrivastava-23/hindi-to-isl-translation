import stanza
import nlp
import os
import subprocess
import time
import cv2
import pandas as pd
import ast
import re
import pyiwn
import nltk
import functools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from nltk.corpus import wordnet as wn
from networkx.drawing.nx_pydot import graphviz_layout
from IPython.display import Video, display
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path

inflections = ast.literal_eval(Path("./utility/output.txt").read_text(encoding="utf-8"))
# build Hindi_form → base_key map
hindi_to_base = {h: base for base, forms in inflections.items() for h in forms}
# helper to normalize any token
def normalize_token(tok):
    return hindi_to_base.get(tok, tok)


nlp_pipeline = stanza.Pipeline('hi', processors='tokenize,lemma,pos,depparse')


def analyze_hindi_sentence(sentence):
    """
    Runs Stanza NLP pipeline on the input Hindi sentence.
    Returns the processed document.
    """
    return nlp_pipeline(sentence)


def extract_token_info(document):
    """
    Takes a stanza Document object,
    returns a pandas DataFrame with token info
    and the root word.
    """
    if document is None:
        return pd.DataFrame(), None
    
    token_data = [
        (token['text'], token['upos'], token['xpos'], token['head'], token['deprel'])
        for token in document.to_dict()[0]
    ]

    df_tokens = pd.DataFrame(token_data, columns=[
        'Text word', 'Universal POS tag', 'Treebank-specific POS (XPOS) tag', 'Parent index', 'Dependency relation'
    ])
    df_tokens.index += 1

    # Extract root word(s) where parent index == 0
    root_word = df_tokens.loc[df_tokens['Parent index'] == 0]['Text word'].iloc[0]

    return df_tokens, root_word


def plot_dependency_tree(df_tokens, hindi_font_path):
    """
    Plot the dependency tree given token DataFrame and path to Hindi font.
    Returns the matplotlib figure.
    """
    hindi_font = fm.FontProperties(fname=hindi_font_path)
    dependency_graph = nx.DiGraph()
    node_labels = {}

    for idx, row in df_tokens.iterrows():
        node_labels[idx] = row["Text word"]
        if row["Parent index"] != 0:  # Skip root node
            dependency_graph.add_edge(row["Parent index"], idx)

    # graph_layout = nx.spring_layout(dependency_graph, seed=42)
    graph_layout = graphviz_layout(dependency_graph, prog="dot")

    plt.figure(figsize=(8, 5))
    nx.draw(dependency_graph, pos=graph_layout, with_labels=True, labels=node_labels,
            node_color="pink", edge_color="black", node_size=2500, font_size=12, arrows=True)

    # Apply Devanagari font to all labels
    for text in plt.gca().texts:
        text.set_fontproperties(hindi_font)

    plt.title("Dependency Parse Tree", fontsize=14)

    # Instead of plt.show(), return the current figure to be rendered in Streamlit
    fig = plt.gcf()
    plt.close(fig)  # Close to avoid double display elsewhere
    return fig


def get_cleaned_word_tags(document):
    # Extract word tags
    word_tags = []
    for sentence in document.sentences:
        for word in sentence.words:
            word_info = {
                "text": word.text,
                "xpos": word.xpos,
                "id": word.id,
                "lemma": word.lemma,
                "deprel": word.deprel,
                "head": word.head
            }
            word_tags.append(word_info)

    # Filter unwanted tags
    unwanted_tags = ['VAUX', 'CC', 'SYM', 'PSP']
    cleaned_word_tags = {}
    for word_info in word_tags:
        if word_info['xpos'] not in unwanted_tags:
            cleaned_word_tags[word_info['id']] = word_info

    return cleaned_word_tags


def change_dict_order(word_dict, row1, row2):
    if row1 == row2:
        return word_dict
    items = list(word_dict.items())
    index1 = None
    index2 = None
    for i, (key, _) in enumerate(items):
        if key == row1:
            index1 = i
        elif key == row2:
            index2 = i
    entry = items.pop(index1)
    items.insert(index2, entry)
    return {k: v for k, v in items}


def reorder_word_signs(cleaned_word_tags, word_sign_order):
    for key, value in cleaned_word_tags.items():
        if value['xpos'] == 'JJ':  # Adjective
            if value['head'] != 0 and value['head'] in cleaned_word_tags:
                if cleaned_word_tags[value['head']]['xpos'] == 'NN':
                    word_sign_order = change_dict_order(word_sign_order, key, value['head'])

        elif value['xpos'] == 'RB':  # Adverb
            if value['head'] != 0 and value['head'] in cleaned_word_tags:
                if cleaned_word_tags[value['head']]['xpos'] in ['VM', 'VAUX']:
                    word_sign_order = change_dict_order(word_sign_order, key, value['head'])

    return word_sign_order


def reorder_neg_wq_words(word_sign_order):
    for key, value in word_sign_order.items():
        if value['xpos'] == 'NEG':
            last_key = list(word_sign_order.keys())[-1]
            word_sign_order = change_dict_order(word_sign_order, key, last_key)

    for key, value in word_sign_order.items():
        if value['xpos'] == 'WQ':
            last_key = list(word_sign_order.keys())[-1]
            word_sign_order = change_dict_order(word_sign_order, key, last_key)

    return word_sign_order

def remove_stopwords(word_sign_order, stopwords_path='./utility/final_stopwords.txt'):
    with open(stopwords_path, 'r', encoding='utf8') as file:
        stopword_list = [word.strip() for word in file.readlines()]

    stopword_removed_list = {
        key: value for key, value in word_sign_order.items()
        if value['text'] not in stopword_list
    }

    return stopword_removed_list

def get_xpos_to_pos_mapping():
    return {
        'NNP': 'pnoun',
        'VM': 'verb',
        'VAUX': 'verb',
        'JJ': 'adjective',
        'RB': 'adverb',
        'PRP': 'pronoun',
        'NEG': 'negative',
        'NN': 'noun',
        'RDP': 'adverb',
        'QF': 'adjective',
        'WQ': 'wh_adverb',
        'NST': 'noun_locative',
        'DEM': 'noun_refer_specific',
        'INTF': 'intensifier',
        # Add more mappings as needed
    }

def extract_sign_words(stopword_removed_list):
    xpos_to_pos = get_xpos_to_pos_mapping()
    sign_words_list = []

    for key, value in stopword_removed_list.items():
        tag = xpos_to_pos.get(value['xpos'], 'extra')
        sign_words_list.append((value['text'], tag))

    return sign_words_list




def load_isl_dictionary(file_path='./utility/isl_dict.txt'):
    with open(file_path, 'r', encoding='utf8') as file:
        raw_dict = ast.literal_eval(file.read())
    
    # Clean keys (remove content in parentheses and lowercase)
    cleaned_dict = {
        re.sub(r'_\(.*\)', '', key).strip().lower(): value
        for key, value in raw_dict.items()
    }
    
    # Lowercase all keys (if any missed)
    cleaned_dict = {key.lower(): value for key, value in cleaned_dict.items()}

    # Manually add missing or required entries
    cleaned_dict['school'] = 'स्कूल'

    return cleaned_dict

def get_isl_hindi_english_dict(cleaned_dict):
    """
    Returns a dictionary mapping Hindi words to English words
    from the provided ISL dictionary (English->Hindi).
    """
    return {hindi_word: english_word for english_word, hindi_word in cleaned_dict.items()}



def init_google_translator():
    return GoogleTranslator(source='auto', target='en')



def get_lemmatizer_and_iwn():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    iwn = pyiwn.IndoWordNet()
    return lemmatizer, iwn

def get_special_video_dict():
    return {
        'i': 'I',
        'who': '/content/drive/MyDrive/Major Project - Final Year/ISL Dictionary/W/Who_Whom.mp4',
        'whom': '/content/drive/MyDrive/Major Project - Final Year/ISL Dictionary/W/Who_Whom.mp4'
    }



def get_synonym_substituted_list(sign_words_list, isl_dict, translator, lemmatizer, iwn, special_videos):
    
    synonym_substituted_list = []

    for word, pos_tag in sign_words_list:

        # ─── normalize any inflected form to its English base key ───
        word = normalize_token(word)
        # ─────────────────────────────────────────────────────────────
        # Step 1: Translate Hindi to English
        english_word = translator.translate(word, src='hi', dest='en').lower()
        english_word_lemmatized = lemmatizer.lemmatize(english_word)

        # Special video dictionary match
        if english_word_lemmatized in special_videos:
            synonym_substituted_list.append((word, pos_tag, special_videos[english_word_lemmatized]))
            continue

        # Proper noun: keep English word
        if pos_tag == 'pnoun':
            synonym_substituted_list.append((word, pos_tag, english_word))
            continue

        # Case 1: Direct match in ISL values
        if word in isl_dict.values():
            synonym_substituted_list.append((word, pos_tag, english_word))
            continue

        # Case 2: Match Hindi synonyms
        try:
            all_hindi_synsets = iwn.synsets(word)
        except Exception:
            all_hindi_synsets = []

        for synset in all_hindi_synsets:
            if synset._head_word in isl_dict.values():
                corresponding_keys = [key for key, value in isl_dict.items() if value == synset._head_word]
                synonym_substituted_list.append((word, pos_tag, corresponding_keys[0]))
                break
        else:
            # Case 3 & 4: English and lemmatized word in ISL
            if english_word in isl_dict:
                synonym_substituted_list.append((word, pos_tag, english_word))
            elif english_word_lemmatized in isl_dict:
                synonym_substituted_list.append((word, pos_tag, english_word_lemmatized))
            else:
                # Case 5: English synonym in ISL
                all_english_synsets = wn.synonyms(english_word)
                found = False
                for syn_list in all_english_synsets:
                    for syn in syn_list:
                        if syn.lower() in isl_dict:
                            synonym_substituted_list.append((word, pos_tag, syn.lower()))
                            found = True
                            break
                    if found:
                        break
                else:
                    # Case 6: Fallback to finger spelling
                    synonym_substituted_list.append((word, pos_tag, '#'))

    return synonym_substituted_list

# Reversed dictionary mapping Hindi words to English words

#print(isl_hindi_english_dict)
sign_mapping_vowels = {
    'ा': 'आ',  # Aa
    'ि': 'इ',  # I
    'ी': 'ई',  # II
    'ु': 'उ',  # U
    'ू': 'ऊ',  # UU
    'ृ': 'ऋ',  # R
    'े': 'ए',  # E
    'ै': 'ऐ',  # AI
    'ो': 'ओ',  # O
    'ौ': 'औ',  # AU
    'ं': 'अं', # Anusvara
    'ः': 'अः'  # Visarga
}


# Example sign mapping dictionary
sign_mapping_vowels = {
    'a': 'A', 'e': 'E', 'i': 'I', 'o': 'O', 'u': 'U',
    # Add more if needed
}

# Set your actual path here
folder_path = r"C:\Users\user\Documents\My Major Project\ISL Videos"


def search_videos(final_isl_list):
    found_videos = []
    for or_word, pos_tag, isl_word in final_isl_list:
        if pos_tag == 'pnoun':  # Alphabets and FingerSpell
            for letter in isl_word:
                video_name = f"{letter.capitalize()}.mp4"
                for root, dirs, files in os.walk(folder_path):
                    full_path = os.path.join(root, video_name)
                    if os.path.isfile(full_path):
                        found_videos.append(full_path)
        elif isl_word == '#':
            for char in or_word:
                video_name = f"{sign_mapping_vowels[char]}.mp4" if char in sign_mapping_vowels else f"{char}.mp4"
                for root, dirs, files in os.walk(folder_path):
                    full_path = os.path.join(root, video_name)
                    if os.path.isfile(full_path):
                        found_videos.append(full_path)
        elif isl_word.startswith('@'):  # Special Words
            found_videos.append(isl_word[1:])
        else:
            video_name = f"{isl_word.capitalize()}.mp4"
            for root, dirs, files in os.walk(folder_path):
                full_path = os.path.join(root, video_name)
                if os.path.isfile(full_path):
                    found_videos.append(full_path)

    return found_videos


def play_videos(video_paths):
    vlc_path = r'C:\Program Files\VideoLAN\VLC\vlc.exe'  # VLC path
    for video_path in video_paths:
        subprocess.Popen([vlc_path, '--fullscreen', video_path])
        time.sleep(5)  # Delay before next video


def merge_videos_opencv(video_paths, output_path):
    # Open the first video to get properties like width, height, and FPS
    if os.path.exists(output_path):
        os.remove(output_path)
        
    cap = cv2.VideoCapture(video_paths[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter to save the merged video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)  # Write frame to output video

        cap.release()

    out.release()  # Finalize the output video



def merge_videos_moviepy(video_paths, output_path):
    # Remove existing merged video if present
    if os.path.exists(output_path):
        os.remove(output_path)

    # Load all clips
    clips = [VideoFileClip(path) for path in video_paths]

    # Concatenate all clips into one
    final_clip = concatenate_videoclips(clips)

    # Write the result to output_path with a standard codec
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')




