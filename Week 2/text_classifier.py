from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

# Load a pretrained DistilBERT with a classification head on top
# num_labels=2 means binary classification (toxic / not toxic) — change to match your classes
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define the LoRA configuration:
# r=8            -> rank of the small adapter matrices (lower = fewer trainable params)
# lora_alpha=16  -> scaling factor applied to the adapter output
# target_modules -> which layers get adapters injected; for DistilBERT these are the
#                   query and value projection layers inside attention
# lora_dropout   -> dropout applied inside the adapter, for regularization
# task_type      -> tells peft this is a sequence classification task
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"], lora_dropout=0.1, task_type="SEQ_CLS")

# Wrap the base model so that only the small LoRA adapter weights are trainable —
# the original DistilBERT weights stay frozen. This makes fine-tuning much cheaper.
model = get_peft_model(base_model, lora_config)