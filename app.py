
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# OBS: downlaod the model weights from the notebook before running the app

def get_response(promptInput):
    model = AutoModelForCausalLM.from_pretrained("gptsw3model126m")
    tokenizer = AutoTokenizer.from_pretrained("gptsw3model126m")

    model.eval()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = f"""
    <|endoftext|><s>
    User:
    {promptInput}

    <s>
    Bot:
    """.strip()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6,
        top_p=1,
    )[0]

    generated_text = tokenizer.decode(generated_token_ids)

    # new code to extract text after "Bot:"
    bot_text_start = generated_text.find("Bot:")
    if bot_text_start != -1:
        bot_text = generated_text[bot_text_start + len("Bot:"):].strip()
        # stop after <s>
        bot_text = bot_text.split("<s>")[0].strip()
        return bot_text
    else:
        return "Bot response not found in the generated text."


st.title("GPT-SW3 126m params")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Är träd fina?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
