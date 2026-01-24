import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


gpu = tf.config.list_physical_devices('GPU')
if gpu:
    name = tf.config.experimental.get_device_details(gpu[0]).get("device_name","unknown")
    print("CUDA available: TRUE")
    print("GPU name: ",name)
else:
    print("CUDA available: FALSE")
    print("GPU: No GPU")


MODEL_ID = "atheeq01/movie_sentima_bert"
CACHE_DIR = "../../../assets/"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = TFAutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID, cache_dir=CACHE_DIR
)

def predict_sentiment(text, temperature=1.0):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )

    outputs = model(inputs, training=False)
    logits = outputs.logits / temperature
    probs = tf.nn.softmax(logits, axis=-1)

    label = int(tf.argmax(probs, axis=-1))
    confidence = float(tf.reduce_max(probs))

    return label, confidence

