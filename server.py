from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
from flask_cors import CORS  # Import CORS

# Load the saved model, feature extractor, and tokenizer
encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "gpt2"

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_checkpoint, 
    decoder_checkpoint
)
model.load_state_dict(torch.load("dense-caption-generator_pro.pt", map_location=torch.device('cpu')))
model.eval()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/", methods=["GET"])
def render():
    return render_template('page.html')

@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        # Load image
        img = Image.open(request.files['image']).convert('RGB')
        print(f"Image loaded: {img.size}")

        # Preprocess image
        inputs = feature_extractor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        # Generate caption
        with torch.no_grad():
            attention_mask = torch.ones(pixel_values.shape, dtype=torch.long)

            outputs = model.generate(
                pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=80,
                num_beams=5,
                early_stopping=True
            )
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated caption: {caption}")

        # Return the generated caption as JSON
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": f"An error has occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)