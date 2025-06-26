
# Rule-Based Hindi to Indian Sign Language (ISL) Translator

A natural language processing (NLP)-based system that translates **Hindi text or speech** into grammatically accurate **Indian Sign Language** using a **rule-based linguistic transformation approach**. This system empowers inclusive communication for the **deaf and hard-of-hearing community** in India.

---

## Project Workflow
<img src="https://github.com/user-attachments/assets/e1a170ff-cc0a-43b6-a18b-88aefce67d0e" alt="Project Flowchart" width="500"/>

1. **Input**: Hindi text or speech  
2. **Processing**: POS tagging, Dependency Parsing, Grammar Transformation  
3. **Mapping**: Hindi → English → ISL Signs  
4. **Fallback**: Synonyms (WordNet) → Finger-spelling  
5. **Output**: Merged ISL video representing the input sentence

---

## Features

- Hindi **text** to ISL video translation  
- Hindi **speech** to ISL translation (via Google Speech Recognition API)  
- Real-time NLP-based sentence analysis  
- POS tagging & dependency parsing (via **Stanza NLP**)  
- Grammar transfer rules (Hindi → ISL SOV structure)  
- Synonym substitution (via **WordNet**, **PyIWN**)  
- Finger-spelling fallback for out-of-vocabulary words  
- Modular, scalable, and easily extendable codebase  
- Pre-recorded ISL videos stitched using **MoviePy**

---

## Dataset Used

- **Indian Sign Language Videos** (Download all from ISLRTC):
  [Google Drive - ISL Dictionary Videos](https://drive.google.com/drive/folders/1U-Pr4r1-cupgNOOq9NH_uTsQnPSVEKco)

---

## Key Files

| File/Folder               | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `app.py`                 | Streamlit app for UI                                                    |
| `translator.py`          | Core logic for grammar transformation & video stitching                 |
| `isl_dict.txt`           | Mapping from English/Hindi words to ISL video filenames                 |
| `final_stopwords.txt`    | Stopwords removed for ISL grammar alignment                             |
| `known_pos_tags.txt`     | POS tag references from Stanza pipeline                                 |
| `testing_results.pdf`    | Accuracy results for different sentence structures                      |
| `README.md`              | This file                                                               |

---

## Installation

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Download NLP models:

```bash
python -c "import stanza; stanza.download('hi')"
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Required Directory Structure

```
Hindi_To_ISL_App/
│
├── app.py
├── translator.py
├── requirements.txt
├── README.md
│
└── utility/
    ├── isl_dict.txt
    ├── final_stopwords.txt
    ├── known_pos_tags.txt
    └── output.txt
```

---

## Notes Before Running

- Ensure **ISL videos** are downloaded and placed in the correct folder structure.
- Update the video path variable:

  ```python
  folder_path = "path/to/Videos"
  ```
  
---

## Usage

### To run the Streamlit app:

```bash
streamlit run app.py
```

### Sample Screenshots of the Application
<img src="https://github.com/user-attachments/assets/c81e618f-c7e6-46a8-8034-e7f8f058c2ee" alt="User Input Interface" width="500"/>
<img src="https://github.com/user-attachments/assets/c9c1377a-1105-4c52-931c-f4043f94ba29" alt="POS Tagging + Dependency Info" width="500"/>
<img src="https://github.com/user-attachments/assets/75cf507a-b312-4883-b66d-8571331721c8" alt="Dependency Tree Visualization" width="500"/>
<img src="https://github.com/user-attachments/assets/9587b9d6-d218-472f-84db-fe00a9e3e6a1" alt="Final Word Reordering + Mapping" width="500"/>
<img src="https://github.com/user-attachments/assets/f165fd76-61bd-4f8c-a1cc-2cbe1fb20cb6" alt='ISL sign for "I / me"' width="332"/>
<img src="https://github.com/user-attachments/assets/db55c727-ca6e-42a0-9b6b-08d8c6db4a9a" alt='ISL sign for "market"' width="332"/>
<img src="https://github.com/user-attachments/assets/5accdb61-34de-4391-9ec5-1953bf86c16b" alt='ISL sign for "go"' width="332"/>


### Accuracy: 91.0% 

See full details in [`testing_results.pdf`](./testing_results.pdf)

---

## Contributors

- [**Abhishek Srivastava**](https://github.com/AbhishekSrivastava-23) 
- [**Govind Singh Tomar**](https://github.com/govind-2003)  
- **Prashant Kumar Singh**  
(Department of CSE, Institute of Engineering & Technology, Lucknow)
