import os
import fitz  # PyMuPDF
from flask import Flask, render_template, request, send_file
from transformers import MarianMTModel, MarianTokenizer
import pytesseract
from PIL import Image
from fpdf import FPDF





app = Flask(__name__, template_folder=".")


# Set model name for MarianMT (you can change to your desired language)
model_name = 'Helsinki-NLP/opus-mt-en-es'  # Example: English to Spanish
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Define the PDF class to use DejaVuSans font (supports more characters)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(200, 10, 'Translated Book', ln=True, align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(4)

    def chapter_body(self, body):
        # Load DejaVuSans font
        self.add_font('DejaVuSans', '', 'fonts/DejaVuSans.ttf', uni=True)  # Add this line to load the font
        self.set_font('DejaVuSans', '', 12)  # Use DejaVuSans font for Unicode support
        self.multi_cell(0, 10, body)
        self.ln()

# Function to extract text from PDF using PyMuPDF (fitz)
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page_num in range(len(doc)):  # Iterate through all pages
        page = doc.load_page(page_num)  # Load each page
        text += page.get_text()  # Extract text from the page
    return text

# Function to extract text from images using Tesseract OCR
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Function to split text into smaller chunks (e.g., paragraphs)
def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    while len(text) > chunk_size:
        # Find the last space before the chunk limit to split properly
        split_point = text.rfind(' ', 0, chunk_size)
        chunks.append(text[:split_point])
        text = text[split_point:].strip()
    if text:
        chunks.append(text)  # Add the remaining part of the text
    return chunks

def load_translation_model(src_lang, tgt_lang):
    """Dynamically load the appropriate MarianMT model for translation."""
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def translate_text(text, src_lang, tgt_lang):
    """Translate text from src_lang to tgt_lang using MarianMT."""
    tokenizer, model = load_translation_model(src_lang, tgt_lang)
    if tokenizer is None or model is None:
        return "Translation model not available for selected language pair."

    chunks = split_text_into_chunks(text)
    translated_text = ""
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        translated_text += tokenizer.decode(translated[0], skip_special_tokens=True) + '\n'
    
    return translated_text


# Function to save translated text to PDF (with correct pagination)
def save_translated_text_to_pdf(translated_text, output_pdf_path):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title("Translated Text")
    
    # Split the translated text into pages (e.g., 300-500 words per page)
    pages = split_text_into_chunks(translated_text, chunk_size=1000)
    
    for page_content in pages:
        pdf.chapter_body(page_content)  # Add content to the page
        pdf.add_page()  # Add a new page for the next chunk of text
        
    pdf.output(output_pdf_path)

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'book_file' not in request.files:
           return "No file part", 400

        file = request.files['book_file']
        input_language = request.form['input_language']
        output_language = request.form['output_language']

        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Extract text from PDF or Image
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_image(file_path)

        # Translate the extracted text
        translated_text = translate_text(text, input_language, output_language)

        # Save translated text to a new PDF in the 'outputs' folder
        translated_pdf_path = os.path.join('outputs', f"translated_{file.filename}")
        save_translated_text_to_pdf(translated_text, translated_pdf_path)

        # Pass the filename only to the template for the download link
        return render_template('index.html', translated_pdf_url=f"outputs/{os.path.basename(translated_pdf_path)}")

    return render_template('index.html')

# Route for downloading the translated PDF
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('outputs', filename), as_attachment=True)
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway assigns a PORT dynamically
    app.run(host="0.0.0.0", port=port)




