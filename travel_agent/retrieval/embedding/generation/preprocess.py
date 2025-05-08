def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("\t", " ")
    # text = text.replace("-", "")
    # text = text.replace(";", " ")
    return text
