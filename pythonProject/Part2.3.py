import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F
import numpy as np

def add_suffix(prefix):
    return prefix + "  Between yes and no as to whether the preceding message is spam or not, I would choose"

# model_name = "google-t5/t5-small"
model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
#for gpt only
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("tokenizer and model loaded...")

# get dataset
df = pd.read_parquet("hf://datasets/ucirvine/sms_spam/plain_text/train-00000-of-00001.parquet")

#Turn each sms message into a full prompt using add_suffix function
df["sms"] = df["sms"].apply(add_suffix)
#separate ids from embeddings so as not to break train_test_split
id_X = tokenizer(df["sms"].tolist(), return_tensors="pt", padding=True)
df_ids = id_X["input_ids"]
df_mask = id_X["attention_mask"]
df_y = df["label"]

#split the first 500 off to be dev set, rest reserved for test
dev_size = 250
dev_ids = df_ids[0:dev_size]
test_ids = df_ids[dev_size:]
dev_mask = df_mask[0:dev_size]
test_mask = df_mask[dev_size:]
dev_y = df_y.iloc[0:dev_size]
test_y = df_y.iloc[dev_size:]
print("Dataset split...")

y_pred = []
#Generate Prompt (code from chatGPT)
for i in range(len(test_ids)):
    if i % 10 == 0: print(f"processing: {i}/{len(test_ids)}")

    single_input_id = test_ids[i].unsqueeze(0)
    single_attention_mask = test_mask[i].unsqueeze(0)
    try:
        with torch.no_grad():
            #for t5, provides a set of dummy decoder ids
            # decoder_input_ids = torch.zeros_like(single_input_id)

            outputs = model(input_ids=single_input_id, attention_mask=single_attention_mask, temperature=1.50)#, decoder_input_ids=decoder_input_ids) #
            logits = outputs.logits
    except Exception as e:
        print(f"Error processing example {i}: {e}")
        exit()
    #get logits for next word
    next_token_logits = logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)

    # Get the probability of yes and no
    word_yes = " yes"
    word_no = " no"
    word_yes_id = tokenizer.encode(word_yes, add_special_tokens=False)[0]
    word_no_id = tokenizer.encode(word_no, add_special_tokens=False)[0]
    yes_probability = np.log(probs[word_yes_id].item())
    no_probability = np.log(probs[word_no_id].item())

    # print(f"Yes = {yes_probability}, No = {no_probability}, Actual = {test_y[i]}")

    #convert preds to binary and store
    if yes_probability <= -17.5:
    # if yes_probability <= no_probability: <--- for t5
        y_pred.append(1)
    else:
        y_pred.append(0)

#get and print results
print()
print("Results:")
print(classification_report(test_y, y_pred))