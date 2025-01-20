from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import xml.etree.ElementTree as ET
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from text_blind_watermark import TextBlindWatermark

# Initialize FastAPI app
app = FastAPI(
    title="File Watermark API",
    description="API for embedding and decoding watermarks in text, XML, or JSON files.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    required_packages = ['punkt', 'stopwords']
    nltk_data_dir = os.path.join(tempfile.gettempdir(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)

# Download NLTK resources at startup
ensure_nltk_data()

class FileProcessor:
    def __init__(self, password: str):
        self.password = password.encode("utf-8")
        self.twm = TextBlindWatermark(pwd=self.password)
        self._ensure_stopwords()
        self.watermarked_positions = []  # Track which words were watermarked

    def _ensure_stopwords(self):
        """Ensure stopwords are available."""
        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            # Fallback to basic stopwords if NLTK fails
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our'])

    def _tokenize_text(self, text: str) -> list:
        """Tokenize text with fallback method if NLTK fails."""
        try:
            return word_tokenize(text)
        except:
            return text.split()

    def embed_watermark_important_words(self, input_text: str, watermark_text: str) -> str:
        """Embed watermark in important words of the text."""
        try:
            watermark = watermark_text.encode("utf-8")
            words = self._tokenize_text(input_text)
            watermarked_words = []
            self.watermarked_positions = []  # Reset positions

            for i, word in enumerate(words):
                if word.lower() not in self.stop_words and word.isalpha() and len(word) > 3:
                    watermarked_word = self.twm.add_wm_rnd(word, watermark)
                    watermarked_words.append(watermarked_word)
                    self.watermarked_positions.append(i)  # Track position
                else:
                    watermarked_words.append(word)

            return " ".join(watermarked_words)
        except Exception as e:
            raise Exception(f"Error in watermark embedding: {str(e)}")

    def decode_watermark_words(self, watermarked_text: str) -> dict:
        """Decode watermarks from the text."""
        try:
            words = self._tokenize_text(watermarked_text)
            watermarked_words = {}
            
            for i, word in enumerate(words):
                if len(word) > 3:  # Only try to decode words longer than 3 characters
                    try:
                        wm = self.twm.extract(word)
                        if wm:
                            try:
                                decoded = wm.decode("utf-8")
                                if decoded:  # Only add if successfully decoded
                                    watermarked_words[word] = {
                                        'position': i,
                                        'watermark': decoded
                                    }
                            except UnicodeDecodeError:
                                continue
                    except Exception:
                        continue

            if watermarked_words:
                return {
                    'total_words': len(words),
                    'watermarked_words': watermarked_words,
                    'watermark_count': len(watermarked_words)
                }
            return {
                'total_words': len(words),
                'watermarked_words': {},
                'watermark_count': 0,
                'message': 'No watermarks detected'
            }
        except Exception as e:
            raise Exception(f"Error in watermark decoding: {str(e)}")

    def process_xml_or_json(self, content: str, watermark_text: str, tag: str) -> str:
        """Process XML or JSON content for watermark embedding."""
        try:
            if content.strip().startswith("<"):  # XML processing
                root = ET.fromstring(content)
                for text_elem in root.findall(f".//{tag}"):
                    if text_elem.text:
                        watermarked_text = self.embed_watermark_important_words(
                            text_elem.text, watermark_text
                        )
                        text_elem.text = watermarked_text
                return ET.tostring(root, encoding='unicode', method='xml')
            else:  # JSON processing
                data = json.loads(content)
                def embed_json(data, tag):
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key == tag and isinstance(value, str):
                                data[key] = self.embed_watermark_important_words(
                                    value, watermark_text
                                )
                            elif isinstance(value, (dict, list)):
                                embed_json(value, tag)
                    elif isinstance(data, list):
                        for item in data:
                            embed_json(item, tag)
                embed_json(data, tag)
                return json.dumps(data, indent=4)
        except Exception as e:
            raise Exception(f"Error processing content: {str(e)}")

    def decode_xml_or_json(self, content: str, tag: str) -> list:
        """Decode watermarks from XML or JSON content."""
        try:
            results = []
            if content.strip().startswith("<"):  # XML processing
                root = ET.fromstring(content)
                for text_elem in root.findall(f".//{tag}"):
                    if text_elem.text:
                        watermarks = self.decode_watermark_words(text_elem.text)
                        results.append({
                            'text': text_elem.text,
                            'analysis': watermarks
                        })
            else:  # JSON processing
                data = json.loads(content)
                def decode_json(data, tag):
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key == tag and isinstance(value, str):
                                watermarks = self.decode_watermark_words(value)
                                results.append({
                                    'text': value,
                                    'analysis': watermarks
                                })
                            elif isinstance(value, (dict, list)):
                                decode_json(value, tag)
                    elif isinstance(data, list):
                        for item in data:
                            decode_json(item, tag)
                decode_json(data, tag)
            return results
        except Exception as e:
            raise Exception(f"Error decoding content: {str(e)}")

# Initialize the FileProcessor with a password
processor = FileProcessor(password="your_secure_password_here")

@app.post("/embed_watermark")
async def embed_watermark(
    file: UploadFile, 
    watermark_text: str = Form(...), 
    tag: str = Form(default="")
):
    """Endpoint to embed watermark in uploaded file."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        content = await file.read()
        content = content.decode("utf-8")

        if file.filename.endswith(".txt"):
            watermarked_text = processor.embed_watermark_important_words(content, watermark_text)
            
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "watermarked_output.txt")
            
            with open(output_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(watermarked_text)
                
            return FileResponse(
                output_path, 
                filename="watermarked_output.txt",
                media_type="text/plain"
            )
        
        else:  # XML or JSON
            if not tag:
                raise HTTPException(
                    status_code=400, 
                    detail="Tag is required for XML/JSON files"
                )
                
            watermarked_content = processor.process_xml_or_json(content, watermark_text, tag)
            
            temp_dir = tempfile.gettempdir()
            ext = file.filename.split('.')[-1]
            output_path = os.path.join(temp_dir, f"watermarked_output.{ext}")
            
            with open(output_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(watermarked_content)
                
            media_type = "application/xml" if ext == "xml" else "application/json"
            return FileResponse(
                output_path, 
                filename=f"watermarked_output.{ext}",
                media_type=media_type
            )

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.post("/decode_watermark")
async def decode_watermark(
    file: UploadFile, 
    tag: str = Form(default="")
):
    """Endpoint to decode watermark from uploaded file."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
            
        content = await file.read()
        content = content.decode("utf-8")

        if file.filename.endswith(".txt"):
            watermarks = processor.decode_watermark_words(content)
            return {"decoded_watermarks": watermarks}
        
        else:  # XML or JSON
            if not tag:
                raise HTTPException(
                    status_code=400, 
                    detail="Tag is required for XML/JSON files"
                )
                
            decoded_results = processor.decode_xml_or_json(content, tag)
            return {"decoded_watermarks": decoded_results}

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.get("/")
def read_root():
    """Root endpoint that provides API information."""
    return {
        "message": "Welcome to the File Watermark API!",
        "version": "1.0.0",
        "endpoints": {
            "/embed_watermark": "POST - Embed watermark in text/XML/JSON file",
            "/decode_watermark": "POST - Decode watermark from text/XML/JSON file"
        }
    }

if __name__ == "__main__":
    import uvicorn
    ensure_nltk_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)
