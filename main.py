from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
import xml.etree.ElementTree as ET
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from text_blind_watermark import TextBlindWatermark

# Initialize FastAPI app
app = FastAPI(title="File Watermark API", description="API for embedding and decoding watermarks in text, XML, or JSON files.")

def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

# Download NLTK resources at startup
download_nltk_resources()

class FileProcessor:
    def __init__(self, password):
        self.password = password.encode("utf-8")
        self.twm = TextBlindWatermark(pwd=self.password)
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            print("Warning: Stopwords not available. Downloading now...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words("english"))

    def embed_watermark_important_words(self, input_text, watermark_text):
        try:
            watermark = watermark_text.encode("utf-8")
            words = word_tokenize(input_text)
            watermarked_words = []

            for word in words:
                if word.lower() not in self.stop_words and word.isalpha():
                    watermarked_words.append(self.twm.add_wm_rnd(word, watermark))
                else:
                    watermarked_words.append(word)

            return " ".join(watermarked_words)
        except Exception as e:
            raise Exception(f"Error in watermark embedding: {str(e)}")

    def decode_watermark_words(self, watermarked_text):
        try:
            words = word_tokenize(watermarked_text)
            watermarked_words = {}

            for word in words:
                wm = self.twm.extract(word)
                if wm:
                    try:
                        decoded = wm.decode("utf-8")
                        watermarked_words[word] = decoded
                    except UnicodeDecodeError:
                        continue

            if watermarked_words:
                return watermarked_words
            return "No watermark detected."
        except Exception as e:
            raise Exception(f"Error in watermark decoding: {str(e)}")

    def process_xml_or_json(self, content, watermark_text, tag):
        try:
            if content.strip().startswith("<"):
                root = ET.fromstring(content)
                for text_elem in root.findall(f".//{tag}"):
                    if text_elem.text:
                        watermarked_text = self.embed_watermark_important_words(text_elem.text, watermark_text)
                        text_elem.text = watermarked_text
                return ET.tostring(root, encoding='unicode', method='xml')

            else:
                data = json.loads(content)
                def embed_json(data, tag):
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key == tag and isinstance(value, str):
                                data[key] = self.embed_watermark_important_words(value, watermark_text)
                            else:
                                embed_json(value, tag)
                    elif isinstance(data, list):
                        for item in data:
                            embed_json(item, tag)
                embed_json(data, tag)
                return json.dumps(data, indent=4)
        except Exception as e:
            raise Exception(f"Error processing content: {str(e)}")

    def decode_xml_or_json(self, content, tag):
        try:
            results = []
            if content.strip().startswith("<"):
                root = ET.fromstring(content)
                for text_elem in root.findall(f".//{tag}"):
                    if text_elem.text:
                        watermarks = self.decode_watermark_words(text_elem.text)
                        if watermarks != "No watermark detected.":
                            results.append({
                                'text': text_elem.text[:100] + '...',
                                'watermarks': watermarks
                            })
            else:
                data = json.loads(content)
                def decode_json(data, tag):
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key == tag and isinstance(value, str):
                                watermarks = self.decode_watermark_words(value)
                                if watermarks != "No watermark detected.":
                                    results.append({
                                        'text': value[:100] + '...',
                                        'watermarks': watermarks
                                    })
                            else:
                                decode_json(value, tag)
                    elif isinstance(data, list):
                        for item in data:
                            decode_json(item, tag)
                decode_json(data, tag)
            return results
        except Exception as e:
            raise Exception(f"Error decoding content: {str(e)}")

processor = FileProcessor(password="p@ssw0rd")

@app.post("/embed_watermark")
async def embed_watermark(file: UploadFile, watermark_text: str = Form(...), tag: str = Form(...)):
    try:
        content = await file.read()
        content = content.decode("utf-8")

        if file.filename.endswith(".txt"):
            watermarked_text = processor.embed_watermark_important_words(content, watermark_text)

            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "watermarked_output.txt")

            with open(output_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(watermarked_text)

            return FileResponse(output_path, filename="watermarked_output.txt")
        
        else:
            watermarked_content = processor.process_xml_or_json(content, watermark_text, tag)

            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"watermarked_output.{file.filename.split('.')[-1]}")

            with open(output_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(watermarked_content)

            return FileResponse(output_path, filename=f"watermarked_output.{file.filename.split('.')[-1]}")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/decode_watermark")
async def decode_watermark(file: UploadFile, tag: str = Form(...)):
    try:
        content = await file.read()
        content = content.decode("utf-8")

        if file.filename.endswith(".txt"):
            watermarks = processor.decode_watermark_words(content)
            if watermarks == "No watermark detected.":
                return {"message": "No watermarks detected."}
            return {"decoded_watermarks": watermarks}
        
        else:
            decoded_results = processor.decode_xml_or_json(content, tag)

            if not decoded_results:
                return {"message": "No watermarks detected in the specified tag."}

            return {"decoded_watermarks": decoded_results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "Welcome to the File Watermark API! Use /embed_watermark or /decode_watermark endpoints."}
