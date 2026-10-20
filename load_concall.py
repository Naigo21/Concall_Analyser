from pypdf import PdfReader

def load_and_clean_pdf(path):
    reader = PdfReader(path)

    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"

    clean_text = raw_text.replace("\n", " ")
    clean_text = " ".join(clean_text.split())

    return clean_text
