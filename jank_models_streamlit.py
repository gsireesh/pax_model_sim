import pickle
import random

import joblib 
import pandas as pd
import plotly.express as px
import streamlit as st

# RANDOM_SEED = 9837452
# random.seed(RANDOM_SEED)



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


with open("accuracies.pkl", "rb") as f:
    accuracies = pickle.load(f)
    
n_data = [100, 500, 2500, 10000, 50000]
model_filenames = [f"model_{n}.pkl" for n in n_data]
models = []
for filename in model_filenames:
    with open(filename, "rb") as f:
        models.append(joblib.load(filename))

test_df = pd.read_json("test_df.json")


st.sidebar.write("Simulated Accuracy:")
sidebar_form = st.sidebar.form("sidebar_form")
selected_idx = sidebar_form.select_slider(
    "Model accuracy to simulate",
    options=[i for i in range(1,6)],
    value=1
) - 1 
model_accuracy = accuracies[selected_idx]

sidebar_form.write(f"Accuracy: {model_accuracy:.2f}")
use_model = sidebar_form.checkbox("Display Model Predictions Box?")
sidebar_form.form_submit_button()

selected_model = models[selected_idx]


example = test_df.sample()
example_title = example["question_title"].iloc[0]
example_body = example["question_content"].iloc[0]
example_answer = example["best_answer"].iloc[0].replace('\\n', '\n')
example_topic = classes[int(example["topic"].iloc[0])]

# model_answer = model.predict()


st.markdown("#Title")
st.markdown("Lorem Ipsum about task")

st.button("Randomize")
st.markdown(f"**{example_title}**\n\n{example_body}\n\n")
st.markdown(f"**Answer:** {example_answer}")

col1, col2 = st.columns(2)
col1.markdown(f"**Expected Answer:** {example_topic}")
col2.markdown(f"**Predicted Answer:** {get_prediction(example_topic, classes, model_accuracy)}")

