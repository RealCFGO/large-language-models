"""
LLM Tokenization og Next-Token Next-Token Prediction

1. Tokenization (BPE vs WordPiece)
2. Next-token Prediction med sandsynlighedsfordelinger
"""

import streamlit as st
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2TokenizerFast,
    BertTokenizerFast
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict
import re

# page config
st.set_page_config(
    page_title="LLM Tokenization Visualisering",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css for styling
st.markdown("""
<style>
    .token-box {
        display: inline-block;
        padding: 5px 10px;
        margin: 2px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 14px;
    }
    .token-normal {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
    }
    .token-compound {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
    }
    .token-punctuation {
        background-color: #f3e5f5;
        border: 1px solid #ba68c8;
    }
    .token-special {
        background-color: #ffebee;
        border: 1px solid #ef5350;
    }
    .token-split {
        background-color: #e8f5e9;
        border: 1px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer(model_name: str, tokenizer_type: str):
    """
    Indlæs fortrænet model og tokenizer.
    Cachet for at undgå genindlæsning ved hver interaktion.
    """
    try:
        # always load gpt model for predictions
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        # load tokenizer based on type
        if tokenizer_type == "BPE (GPT-2)":
            tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
            visualization_tokenizer = None
        elif tokenizer_type == "WordPiece (BERT)":
            # load gpt tokenizer for model
            tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
            # load bert tokenizer for viz
            visualization_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            visualization_tokenizer = None
        
        return model, tokenizer, visualization_tokenizer
    except Exception as e:
        st.error(f"Fejl ved indlæsning af model: {e}")
        return None, None, None


@st.cache_resource
def load_comparison_tokenizers():
    """
    Indlæs både BPE og WordPiece tokenizers til sammenligning.
    Cachet for at undgå genindlæsning.
    """
    bpe = GPT2TokenizerFast.from_pretrained("gpt2")
    wordpiece = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return bpe, wordpiece


def classify_token(token: str, is_continuation: bool) -> str:
    """
    Klassificer token type til farvekodning:
    - special: Special tokens som <|endoftext|>, [CLS], osv.
    - punctuation: Tegnsætningstegn
    - compound: Ordfortsættelser (f.eks. "Ġworld" i GPT-2, "##ing" i BERT)
    - split: Del af et opdelt ord
    """
    # clean token
    clean_token = token.replace('Ġ', '').replace('##', '').strip()
    
    # special tokens
    if token.startswith('<') and token.endswith('>'):
        return 'special'
    if token in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]']:
        return 'special'
    
    # punctuation
    if re.match(r'^[^\w\s]+$', clean_token):
        return 'punctuation'
    
    # continuation tokens
    if is_continuation or token.startswith('##') or token.startswith('Ġ'):
        return 'compound'
    
    # check if subword
    if not token.startswith('Ġ') and len(token) > 0:
        return 'split'
    
    return 'normal'


def visualize_tokens(text: str, tokenizer, tokenizer_type: str) -> Tuple[str, pd.DataFrame]:
    """
    Skab visuel repræsentation af tokens med farvekodning.
    Returnerer HTML til visning og en DataFrame med token information.
    """
    # tokenize
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    
    # build viz
    html_output = "<div style='line-height: 2.5;'>"
    token_data = []
    
    for idx, token in enumerate(tokens):
        # check if continuation
        is_continuation = False
        if tokenizer_type == "BPE (GPT-2)":
            is_continuation = not token.startswith('Ġ') and idx > 0
        elif tokenizer_type == "WordPiece (BERT)":
            is_continuation = token.startswith('##')
        
        # classify token
        token_class = classify_token(token, is_continuation)
        
        # display token
        display_token = token.replace('Ġ', '▁').replace('##', '')
        
        # add to html
        html_output += f'<span class="token-box token-{token_class}" title="{token_class}">{display_token}</span>'
        
        # add to data
        token_data.append({
            'Index': idx,
            'Token': token,
            'Display': display_token,
            'Type': token_class,
            'Token ID': encoded['input_ids'][idx]
        })
    
    html_output += "</div>"
    
    # add legend
    html_output += """
    <div style='margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>
        <strong>Forklaring:</strong><br>
        <span class='token-box token-compound'>Fortsættelse</span>
        <span class='token-box token-split'>Delord</span>
        <span class='token-box token-punctuation'>Tegnsætning</span>
        <span class='token-box token-special'>Special</span>
    </div>
    """
    
    return html_output, pd.DataFrame(token_data)


def get_word_token_counts(text: str, tokenizer) -> List[Tuple[str, int]]:
    """
    Vis hvor mange tokens hvert ord er delt i.
    Nyttigt til at forstå tokenization granularitet.
    """
    words = text.split()
    word_token_counts = []
    
    for word in words:
        # tokenize word
        tokens = tokenizer.tokenize(word)
        word_token_counts.append((word, len(tokens)))
    
    return word_token_counts


def predict_next_tokens(text: str, model, tokenizer, top_n: int = 10, 
                        temperature: float = 1.0, 
                        repetition_penalty: float = 1.0) -> pd.DataFrame:
    """
    Forudsig næste token og returner top-N kandidater med sandsynligheder.
    Attention mechanisms påvirker forudsigelserne.
    """
    # encode input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    
    # get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # get logits for last token
    next_token_logits = logits[0, -1, :].clone()
    
    # apply repetition penalty
    if repetition_penalty != 1.0:
        for token_id in input_ids:
            next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
    
    # apply temperature
    next_token_logits = next_token_logits / temperature
    
    # softmax to get probs
    probs = torch.softmax(next_token_logits, dim=-1)
    
    # get top-n
    top_probs, top_indices = torch.topk(probs, top_n)
    
    # convert to tokens
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        predictions.append({
            'Token': token,
            'Probability': prob.item(),
            'Percentage': f"{prob.item() * 100:.2f}%",
            'Token ID': idx.item()
        })
    
    return pd.DataFrame(predictions)


def generate_next_words(text: str, model, tokenizer, num_words: int = 3, 
                        temperature: float = 1.0, 
                        repetition_penalty: float = 1.0) -> str:
    """
    Generer de næste N ord ved at forudsige tokens iterativt.
    """
    generated_text = text
    word_count = 0
    max_tokens = num_words * 5  # safety limit
    token_count = 0
    previous_tokens = []  # track tokens to avoid repetition
    
    while word_count < num_words and token_count < max_tokens:
        # encode current text
        inputs = tokenizer(generated_text, return_tensors="pt")
        
        # get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # get next token logits
        next_token_logits = logits[0, -1, :].clone()
        
        # apply repetition penalty
        for token_id in previous_tokens[-50:]:
            next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
        
        # apply temperature
        next_token_logits = next_token_logits / temperature
        
        # sample from top-k
        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # sample from distribution
        top_probs = top_probs / top_probs.sum()  # renormalize
        next_token_id = top_indices[torch.multinomial(top_probs, 1)].item()
        
        # decode and append
        next_token_text = tokenizer.decode([next_token_id])
        generated_text += next_token_text
        previous_tokens.append(next_token_id)
        
        # count words roughly
        if ' ' in next_token_text or next_token_text.strip():
            # check if complete word
            words_in_token = len(next_token_text.strip().split())
            if words_in_token > 0:
                word_count += words_in_token
        
        token_count += 1
    
    return generated_text


def create_probability_chart(pred_df: pd.DataFrame) -> go.Figure:
    """
    Skab et horisontalt søjlediagram til sandsynlighedsvisualisering.
    """
    # gradient colors
    colors = px.colors.sequential.Blues
    
    fig = go.Figure(go.Bar(
        x=pred_df['Probability'],
        y=pred_df['Token'],
        orientation='h',
        marker=dict(
            color=pred_df['Probability'],
            colorscale='Viridis',  # vibrant colors
            showscale=True,
            colorbar=dict(title="Sandsynlighed", thickness=15, len=0.7)
        ),
        text=pred_df['Percentage'],
        textposition='inside',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.4f}<br>Percentage: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Top Next-Token Prediction Sandsynligheder",
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis_title="Sandsynlighed",
        yaxis_title="Token",
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=100, r=50, t=60, b=50)
    )
    
    return fig


def main():
    st.title("LLM Tokenization og Next-Token Prediction Visualiseret")
    st.markdown("""
    - **Tokenization**: Hvordan tekst opdeles i tokens
    - **Next-Token Preddiction**: Hvad modellen forudsiger kommer som det næste
    - **Attention Mechanisms**: Kontroller tekstgenereringsparametrene
    """)
    
    # fixed model
    model_name = "gpt2"
    st.sidebar.info("Model: **GPT-2 (small)**")
    
    # tokenizer type
    tokenizer_type = st.sidebar.selectbox(
        "Tokenizer Type",
        options=["BPE (GPT-2)", "WordPiece (BERT)"],
        index=0,
        help="Sammenlign forskellige tokenization strategier"
    )
    
    # top-n predictions
    top_n = st.sidebar.slider(
        "Top-N Forudsigelser",
        min_value=5,
        max_value=20,
        value=10,
        help="Antal af top forudsigelser at vise"
    )
    
    # attention mechanism controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Attention Mechanisms")
    st.sidebar.markdown("Kontroller hvordan modellen genererer tekst")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Lavere værdi = mere fokuseret, højere værdi = mere spredt distribution"
    )
    
    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.5,
        value=1.0,
        step=0.1,
        help="Straf gentagelse af tokens fra input teksten"
    )
    
    # load model and tokenizer
    with st.spinner(f"Indlæser GPT-2 (small)..."):
        model, tokenizer, visualization_tokenizer = load_model_and_tokenizer(model_name, tokenizer_type)
    
    if model is None or tokenizer is None:
        st.error("Kunne ikke indlæse model. Prøv venligst igen.")
        return
    
    st.sidebar.success("Model indlæst!")
    
    # main input
    st.header("Indtast Tekst")
    
    user_text = st.text_area(
        "Skriv eller indsæt tekst her:",
        value="The quick brown fox jumps over",
        height=150
    )
    
    if not user_text.strip():
        st.warning("Indtast venligst noget tekst at analysere.")
        return
    
    # create two columns
    col1, col2 = st.columns([1, 1])
    
    # column 1: tokenization
    with col1:
        st.header("Tokenization")
        
        # use viz tokenizer if available
        display_tokenizer = visualization_tokenizer if visualization_tokenizer is not None else tokenizer
        
        # visualize tokens
        html_tokens, token_df = visualize_tokens(user_text, display_tokenizer, tokenizer_type)
        st.markdown(html_tokens, unsafe_allow_html=True)
        
        # token stats
        st.subheader("Token Statistik")
        st.metric("Totale Tokens", len(token_df))
        
        # show token breakdown
        with st.expander("Se Token Detaljer"):
            st.dataframe(token_df, use_container_width=True)
        
        # word to token mapping
        st.subheader("Ord til Tokens")
        word_token_counts = get_word_token_counts(user_text, display_tokenizer)
        
        word_df = pd.DataFrame(word_token_counts, columns=['Ord', 'Token Antal'])
        st.dataframe(word_df, use_container_width=True)
        
        # highlight multi-token words
        multi_token_words = word_df[word_df['Token Antal'] > 1]
        if not multi_token_words.empty:
            st.info(f"Fandt {len(multi_token_words)} ord opdelt i flere tokens")
    
    # column 2: next-token prediction
    with col2:
        st.header("Next-Token Forudsigelse")
        
        with st.spinner("Forudsiger næste tokens..."):
            pred_df = predict_next_tokens(
                user_text, 
                model, 
                tokenizer, 
                top_n,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
        
        # display chart
        fig = create_probability_chart(pred_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # display table
        st.subheader("Forudsigelse Detaljer")
        st.dataframe(pred_df, use_container_width=True)
        
        # explanation
        st.info(f"""
        **Hvordan attention mechanisms påvirker forudsigelse:**
        
        Juster parametrene i sidebaren for at se hvordan de ændrer forudsigelserne:
        - **Temperature**: Ændrer fordelingen (lavere = mere fokuseret, højere = mere spredt)
        - **Repetition Penalty**: Reducerer sandsynlighed for ord der allerede er brugt i input
        
        Forudsigelserne opdateres automatisk når du ændrer parametrene.
        """)
    
    # educational section
    st.header("Forståelse af Resultaterne")
    
    tab1, tab2, tab3 = st.tabs(["Tokenization", "Next-Token Forudsigelse", "Attention Mechanisms"])
    
    with tab1:
        st.markdown("""
        ### Tokenization Farver
        - **Orange (Fortsættelse)**: Tokens der fortsætter tidligere ones
        - **Grøn (Delord)**: Dele af opdelte ord
        - **Lilla (Tegnsætning)**: Tegnsætningstegn
        - **Rød (Special)**: Special model tokens
        """)
    
    with tab2:
        st.markdown("""
        ### Next-Token Prediction Forklaret
        
        **Hvordan virker det?**
        Modellen behandler alle input tokens og forudsiger sandsynligheder for næste token i sekvensen.
        
        **Nøglepunkter:**
        - Modellen forudsiger ÉN token ad gangen (ikke et helt ord)
        - Sandsynligheder summerer til 100% på tværs af alle mulige tokens
        - Højere sandsynlighed = mere sikker forudsigelse
        - Kontekst fra alle tidligere tokens påvirker forudsigelsen

        **For eksempel:**
        Hvis "running" opdeles i ["run", "##ning"], forudsiger modellen "##ning" efter "run" med en vis sandsynlighed baseret på konteksten.
        """)
    
    with tab3:
        st.markdown("""
        ### Attention Mechanism Parametre
        
        **Temperature**
        - Kontrollerer hvor fokuseret sandsynlighedsfordelingen er
        - Lavere værdier (0.1-0.5): Mere deterministisk, fokuseret på højeste sandsynlighedstokens
        - Højere værdier (1.0-2.0): Mere spredt fordeling, tillader mindre sandsynlige tokens
        - Standard: 1.0 (original model distribution)
        
        **Repetition Penalty**
        - Reducerer sandsynligheden for at gentage tokens fra input teksten
        - 1.0 = ingen straf (original forudsigelser)
        - 1.5-2.0 = moderat til stærk straf mod gentagelser
        - Højere værdier tvinger modellen til at vælge nye tokens
        
        **Hvordan de arbejder sammen:**
        - Temperature justerer distributionen før softmax
        - Repetition penalty holder forudsigelserne varierede
        - Sammen giver de kontrol over både fokus og variation
        """)


if __name__ == "__main__":
    main()
