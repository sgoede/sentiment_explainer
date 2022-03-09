import torch
from flair.data import Sentence
from flair.models import TextClassifier
from flair_model_wrapper import ModelWrapper
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from interpret_flair import interpret_sentence, visualize_attributions
import numpy as np
import streamlit as st
import pandas as pd
st.set_page_config(layout="wide")
st.title("Sentiment explained - ODSC 2022 submission Review")
st.subheader("Created by: Stephan de Goede")
@st.experimental_singleton
def get_classifier():
    # load tagger
    classifier = TextClassifier.load('sentiment')
    return classifier
# Load the pretrained Flair classifier.
model_load_state = st.text("Loading Sentiment Model...")
flair_model = get_classifier()
model_load_state.text("Loading Sentiment Model... done")
# sentence object
user_input = st.text_area("The English text you want to have the sentiment from")
# Initialize state.
if "clicked" not in st.session_state:
    st.session_state.clicked = False
    st.session_state.word_scores = None
    st.session_state.ordered_lig = None
# Define callbacks to handle button clicks.
def handle_click():
    if  st.session_state.clicked == True:
        st.session_state.word_scores = word_attributions.detach().numpy()
        st.session_state.ordered_lig = [(readable_tokens[i], word_scores[i]) for i in np.argsort(word_scores)][::-1]
def handle_second_click():
    if st.session_state.clicked == True:
        word_scores = st.session_state.word_scores
        ordered_lig = st.session_state.ordered_lig

if len(user_input) >0 :
    #prediction to select the target label:
    tokenized_user_input = Sentence(user_input)
    predicted_sentiment = flair_model.predict(tokenized_user_input)
    target = tokenized_user_input.get_label_names()[0]
    st.write("the model has predicted the sentiment as:",target)
    st.write("Let's have a look at the raw model output:")
    st.write(tokenized_user_input)
    if st.button("Click here to visually inspect the outcome of the model"):
    # In order to make use of Captum's LayerIntegratedGradients method we had to rework Flair's forward function.
    # This is handled by the wrapper. The wrapper inherits functions of the Flair text-classifier object
     # and allows us to calculate attributions with respect to a target class.
        st.session_state.submitted = True
        @st.experimental_singleton
        def rework_flair_model():
            flair_model_wrapper = ModelWrapper(flair_model)
            return flair_model_wrapper
        rework_load_state = st.text("Reworking Flair's forward function....")
        flair_rework = rework_flair_model()
        rework_load_state.text("Succesfully reworked Flair's forward function.")
        # As described in the source code of documentation of Captum:
        # "Layer Integrated Gradients is a variant of Integrated Gradients that assigns an importance score to
        # layer inputs or outputs, depending on whether we attribute to the former or to the latter one."
        # In this case, we are interested how the input embeddings of the model contribute to the output.
        @st.experimental_singleton
        def LayerIntegratedGradients_Flair():
            lig = LayerIntegratedGradients(flair_rework, flair_rework.model.embeddings)
            return lig
        LayerIntegratedGradients_load_state = st.text("Calculating LayerIntegratedGradients....")
        lig = LayerIntegratedGradients_Flair()
        LayerIntegratedGradients_load_state.text("Succesfully calculated LayerIntegratedGradients")
        # create an empty list to store our attribitions results in order to visualize them using Captum.
        visualization_list = []
        #Let's run the Layer Integrated Gradient method on the two paragraphs, and determine what
        # drives the prediction. As an additional note, the number of steps & the estimation method can have an
        # impact on the attribution.
        readable_tokens, word_attributions, delta = interpret_sentence(flair_rework,
                                                                lig,
                                                                user_input,
                                                                target,
                                                                visualization_list,
                                                                n_steps=15,
                                                                estimation_method="gausslegendre",
                                                                internal_batch_size=3)
        # Let's visualize the score attribution of the model
        st.write(viz.visualize_text(visualization_list))
        st.write(" ")
        st.session_state.clicked = True
        word_scores = word_attributions.detach().numpy()
        ordered_lig = [(readable_tokens[i], word_scores[i]) for i in np.argsort(word_scores)][::-1]
        st.session_state.word_scores = word_scores
        st.session_state.ordered_lig = ordered_lig
        st.legacy_caching.clear_cache()
    if st.session_state.ordered_lig is not None:
       if st.button("Click here to see the absolute values of the words"):
            st.dataframe(pd.DataFrame(st.session_state.ordered_lig,columns=['readable tokens',' word scores']))
            st.legacy_caching.clear_cache()
st.write('''Resources used: FLAIR Framework: @inproceedings{akbik2019flair,
  title={FLAIR: An easy-to-use framework for state-of-the-art NLP},
  author={Akbik, Alan and Bergmann, Tanja and Blythe, Duncan and Rasul, Kashif and Schweter, Stefan and Vollgraf, Roland},
  booktitle={{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
  pages={54--59},
  year={2019}
}

Interpret-FLAIR: https://github.com/robinvanschaik/interpret-flair#authors''')



