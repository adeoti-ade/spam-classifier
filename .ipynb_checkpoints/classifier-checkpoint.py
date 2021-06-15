import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def predict_spam_or_ham(sms=None):
    if not isinstance(sms, (list, )):
        raise AssertionError("sms must be of type list")
    if not all(isinstance(elem, list) for elem in sms):
        raise AssertionError("items inside sms must be of type list")
    s_model, tokenizer = load_model()
    preds = []
    max_length = 8
    for item in sms:
        sms_proc = tokenizer.texts_to_sequences(item)
        sms_proc = pad_sequences(sms_proc, maxlen=max_length, padding='post')
#         sms_proc = process_sms(item)
        pred = (s_model.predict(sms_proc) > 0.5).astype("int32").item()
        if pred == 1:
            result = (" ".join(item), "spam")
        if pred == 0:
            result = (" ".join(item), "ham")
        preds.append(result)
    
    return preds

def load_model(path="spam_model"):
    """
    This method loads the model from the local storage and return it as a tokenizer
    """
    s_model = tf.keras.models.load_model("spam_model")
    with open('spam_model/tokenizer.pkl', 'rb') as _input:
        tokenizer = pickle.load(_input)
    
    return s_model, tokenizer

sms_one = ["Hi, Adeoti. Please call me"]
sms_two = ["We know someone who you know that fancies you. Call 09058097218 to find out who. POBox 6, LS15HB "]

resp = predict_spam_or_ham([sms_one, sms_two])
print(resp)