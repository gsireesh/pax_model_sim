import pickle
import random

import joblib 
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")

# RANDOM_SEED = 9837452
# random.seed(RANDOM_SEED)

def init_state():
    st.session_state["examples"] = []
    st.session_state["random_labels"] = []
    st.session_state["model_labels"] = []
    st.session_state["expected_labels"] = []
    st.session_state["intialized"] = True


classes = dict(enumerate([
    "Society & Culture", 
    "Science & Mathematics", 
    "Health", 
    "Education & Reference", 
    "Computers & Internet", 
    "Sports", 
    "Business & Finance", 
    "Entertainment & Music", 
    "Family & Relationships", 
    "Politics & Government", 
]))


def get_prediction(correct_class, classes, p_correct):
    r = random.random()
    print(r)
    if r <= p_correct:
        return correct_class
    else:
        return classes[random.choice([clazz for clazz in classes if clazz != correct_class])]


def format_example(example):
    example_title = example["question_title"].iloc[0]
    example_body = example["question_content"].iloc[0]
    example_answer = example["best_answer"].iloc[0].replace('\\n', '\n')
    return f"### {example_title}\n\n{example_body}\n\n**Answer:** {example_answer}"


with open("accuracies.pkl", "rb") as f:
    accuracies = pickle.load(f)
    
n_data = [100, 500, 2500, 10000, 50000]
model_filenames = [f"model_{n}.pkl" for n in n_data]

test_df = pd.read_json("test_df.json")


### STREAMLIT SECTION

if not st.session_state.get("intialized", False):
    init_state()

st.sidebar.write("Simulated Accuracy:")
sidebar_form = st.sidebar.form("sidebar_form")
selected_idx = sidebar_form.select_slider(
    "Model accuracy to simulate",
    options=[i for i in range(1,6)],
    value=1
) - 1 
model_accuracy = accuracies[selected_idx]

sidebar_form.write(f"Accuracy: {model_accuracy:.2f}")
sidebar_form.form_submit_button("Reload app", on_click=init_state)

selected_model = model_filenames[selected_idx]
with open(selected_model, "rb") as f:
    model = joblib.load(selected_model)


example = test_df.sample()
example_topic = classes[int(example["topic"].iloc[0])]
random_label = get_prediction(example_topic, classes, model_accuracy)
model_label = classes[model.predict(example["all_text"])[0]]
print(model_label)

st.session_state["examples"].append(format_example(example))
st.session_state["expected_labels"].append(example_topic)
st.session_state["random_labels"].append(random_label)
st.session_state["model_labels"].append(model_label)


st.markdown("# Model \"Early Test\" Prototype")
st.markdown("""Imagine you are a serious™️ white collar type, and you've enlisted the 
help of your AI team to build you a text classifier. You and your team are busy, so you don't have a ton of time
to annotate data for this project. In order to iron out what the user experience of the text classifier should 
look like, you're presented with the interface below. Spend some time working with this text classifier. How 
would you react if it were deployed in your workflow today?""")

st.markdown("## Model Section")

col1, col2 = st.columns(2)
col1.button("Randomize")
use_model = col2.checkbox("Display Model Predictions Box?")
st.markdown(format_example(example))

col1, col2 = st.columns(2)
col1.markdown(f"**Expected Answer:** {example_topic}")
col2.markdown(f"**Predicted Answer:** {random_label}")



if use_model:
    with st.expander("Compare with real model"):
        display_text = [example.replace("\n", " ").replace("#", '') for example in st.session_state["examples"]]
        display_df = pd.DataFrame(
            {
                "Text": display_text,
                "Expected Label": st.session_state["expected_labels"],
                "Random Label": st.session_state["random_labels"],
                "Model Label": st.session_state["model_labels"]
            })
        st.markdown(display_df.to_markdown(index=False))
