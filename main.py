import os
import time
import json
import asyncio
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import onnxruntime as ort
from google import genai
from google.genai import types
from dotenv import load_dotenv
from supabase import create_client, Client

# Load Environment Variables
load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
print(f" REAL-TIME GEMINI KEY LOADED: {api_key[:10] + '...' if api_key else 'NONE'}")

# Initialize Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

app = FastAPI()

# Initialize Google Gemini Client
genai_client = genai.Client(api_key=api_key)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (style.css) from the root directory
app.mount("/static", StaticFiles(directory="."), name="static")

CLASS_NAMES = ["Coccidiosis", "Healthy", "New Castle", "Salmonella"]
CONFIDENCE_THRESHOLD = 0.80        # Minimum top class confidence (80%)
ACTIVE_CLASS_THRESHOLD = 0.001     # 0.1% probability counts as active class

session = None

# Initialize ONNX Model Session
def init_model():
    global session
    try:
        model_path = "./models/chicken_model.onnx"
        session = ort.InferenceSession(model_path)
        print("ONNX Model successfully loaded into Python!")
    except Exception as err:
        print(f"Failed to load ONNX model: {err}")

init_model()

# Preprocess uploaded image: Center-crop to square + RGB Tensor Normalization
async def preprocess_image(image_bytes: bytes):
    TARGET_SIZE = 244

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("CORRUPTED_IMAGE")

    # --- CENTER CROP TO SQUARE (Removes distractive background noise) ---
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    img_cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Resize cleanly to target model input size
    img_resized = cv2.resize(img_cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_float = img_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_normalized = (img_float - mean) / std
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32)
    return input_tensor

def softmax(logits):
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    sum_exps = np.sum(exps)
    return exps / sum_exps

# Serve Frontend Root Route directly from Python
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>KUKU-KUKU | AI Chicken Disease Classifier</title>
    <link rel="stylesheet" href="/static/style.css" />
    <style>
      .lang-tabs {
        display: flex;
        gap: 0.4rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
      }
      .lang-tab {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #fff;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: background 0.2s;
      }
      .lang-tab.active {
        background: #3b82f6;
        border-color: #3b82f6;
        font-weight: bold;
      }
      .vet-loading {
        color: #60a5fa;
        font-style: italic;
        font-size: 0.9rem;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <header>
        <h1>🐔 KUKU-AI</h1>
        <p>Instant Chicken Disease Diagnostics & Health Analysis</p>
      </header>

      <div class="stats-grid">
        <div class="stat-card">
          <h4>Total Scans</h4>
          <p id="statTotalScans">0</p>
        </div>
        <div class="stat-card">
          <h4>Last Diagnosis</h4>
          <p id="statLastDiagnosis" style="font-size: 1.2rem">None</p>
        </div>
        <div class="stat-card">
          <h4>System Status</h4>
          <p style="color: #10b981; font-size: 1.2rem">● Active</p>
        </div>
      </div>

      <div class="card">
        <div
          class="upload-area"
          id="dropZone"
          onclick="document.getElementById('fileInput').click()"
        >
          <div class="upload-icon">📸</div>
          <h3>Upload poultry waste/droppings photo here</h3>
          <p style="color: var(--text-sub); font-size: 0.85rem; margin-top: 0.4rem;">
            Supports PNG, JPG, JPEG
          </p>
          <img id="previewImg" alt="Preview" />
        </div>

        <input type="file" id="fileInput" accept="image/*" />
        <button class="btn" id="predictBtn" disabled onclick="runInference()">
          Analyze Image
        </button>

        <div class="result-box" id="resultBox" style="display: none">
          <div class="result-header">
            <h3>Diagnosis Result</h3>
            <span class="prediction-badge" id="predictionBadge">--</span>
          </div>
          <p style="margin-bottom: 1rem; color: var(--text-sub)">
            Confidence Score:
            <strong id="confidenceText" style="color: var(--text-main)">0%</strong>
          </p>

          <div id="probabilitiesWrapper">
            <h4>Class Probabilities:</h4>
            <div id="probabilitiesContainer" style="margin-top: 0.8rem"></div>
          </div>

          <div
            id="solutionContainer"
            style="
              margin-top: 1.5rem;
              padding: 1rem;
              background: rgba(255, 255, 255, 0.05);
              border-radius: 8px;
              border-left: 4px solid #3b82f6;
            "
          >
            <div
              style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                margin-bottom: 0.6rem;
              "
            >
              <h4 id="solutionTitle" style="color: #60a5fa; margin: 0">
                📋 AI Veterinary Action Plan
              </h4>
              <div class="lang-tabs" id="langTabs">
                <button class="lang-tab active" onclick="switchLanguage('en')">English</button>
                <button class="lang-tab" onclick="switchLanguage('ha')">Hausa</button>
                <button class="lang-tab" onclick="switchLanguage('yo')">Yorùbá</button>
                <button class="lang-tab" onclick="switchLanguage('ig')">Igbo</button>
              </div>
            </div>

            <div
              id="aiVetContent"
              style="
                font-size: 0.9rem;
                line-height: 1.6;
                white-space: pre-wrap;
                color: var(--text-main);
              "
            >
              <span class="vet-loading">⏳ Generating multilingual treatment guidelines with AI...</span>
            </div>
          </div>
        </div>
      </div>

      <div class="card" id="aiVetCard" style="display: none">
        <h3>💬 Ask AI Vet Follow-Up Questions</h3>
        <p style="color: var(--text-sub); font-size: 0.85rem; margin-bottom: 1rem;">
          Need specific medication dosage or isolation tips for your farm size? Ask below.
        </p>

        <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem">
          <input
            type="text"
            id="vetQueryInput"
            placeholder="e.g., How much dosage should I give to 50 3-week-old chicks?"
            style="
              flex: 1;
              padding: 0.75rem;
              border-radius: 6px;
              border: 1px solid rgba(255, 255, 255, 0.2);
              background: rgba(0, 0, 0, 0.2);
              color: white;
            "
          />
          <button
            class="btn"
            id="askVetBtn"
            onclick="askFollowUpVet()"
            style="width: auto; padding: 0 1.2rem"
          >
            Ask AI
          </button>
        </div>

        <div
          id="aiVetFollowupResponse"
          style="
            display: none;
            padding: 1rem;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.6;
            white-space: pre-wrap;
          "
        ></div>
      </div>

      <div class="card feedback-section">
        <h3>💬 Feedback & Case Data</h3>
        <p style="color: var(--text-sub); font-size: 0.85rem; margin-bottom: 0.8rem;">
          Did the model accurately identify your flock's condition? Share feedback to improve our system.
        </p>
        <textarea
          id="feedbackText"
          placeholder="Describe symptoms or verify accuracy..."
        ></textarea>
        <button
          class="btn"
          style="background: #334155"
          onclick="submitFeedback()"
        >
          Submit Feedback
        </button>
      </div>
    </div>

    <footer>
      <p class="creator-tag">
        Designed & Built by <span class="glowing-name">Francis Abba</span>
      </p>
    </footer>

    <script>
      const fileInput = document.getElementById("fileInput");
      const previewImg = document.getElementById("previewImg");
      const predictBtn = document.getElementById("predictBtn");
      const resultBox = document.getElementById("resultBox");
      const dropZone = document.getElementById("dropZone");
      const aiVetCard = document.getElementById("aiVetCard");

      let currentDiagnosis = "";
      let currentConfidence = 0;
      let multiLangSolutions = { en: "", ha: "", yo: "", ig: "" };
      let currentLang = "en";

      let totalScans = parseInt(localStorage.getItem("totalScans") || "0");
      document.getElementById("statTotalScans").innerText = totalScans;

      let savedDiagnosis = localStorage.getItem("lastDiagnosis") || "None";
      if (savedDiagnosis !== "None") {
        document.getElementById("statLastDiagnosis").innerText = savedDiagnosis;
      }

      function handleFile(file) {
        if (file && file.type.startsWith("image/")) {
          const reader = new FileReader();
          reader.onload = (event) => {
            previewImg.src = event.target.result;
            previewImg.style.display = "block";
            predictBtn.disabled = false;
          };
          reader.readAsDataURL(file);
        }
      }

      fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--primary, #3b82f6)";
      });

      dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "";
      });

      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "";
        if (e.dataTransfer.files.length > 0) {
          fileInput.files = e.dataTransfer.files;
          handleFile(e.dataTransfer.files[0]);
        }
      });

      function switchLanguage(lang) {
        currentLang = lang;
        document.querySelectorAll(".lang-tab").forEach((tab) => tab.classList.remove("active"));
        event.target.classList.add("active");

        const contentDiv = document.getElementById("aiVetContent");
        if (multiLangSolutions[lang]) {
          contentDiv.innerText = multiLangSolutions[lang];
        } else {
          contentDiv.innerHTML = `<span class="vet-loading">Translation pending...</span>`;
        }
      }

      async function runInference() {
        const file = fileInput.files[0];
        if (!file) return;

        predictBtn.innerText = "Analyzing...";
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append("file", file);

        try {
          const response = await fetch("/predict", {
            method: "POST",
            body: formData,
          });

          const data = await response.json();

          if (response.ok) {
            currentDiagnosis = data.prediction;
            currentConfidence = data.confidence;

            document.getElementById("predictionBadge").innerText = currentDiagnosis;
            document.getElementById("confidenceText").innerText = `${currentConfidence.toFixed(1)}%`;

            const probContainer = document.getElementById("probabilitiesContainer");
            const probWrapper = document.getElementById("probabilitiesWrapper");
            probContainer.innerHTML = "";

            if (!data.isRecognized) {
              probWrapper.style.display = "none";
              aiVetCard.style.display = "none";
              document.getElementById("aiVetContent").innerText = "⚠️ This image was recognized as an Unknown Image (probabilities overlap or confidence is low). Please upload a clear photo of poultry droppings.";
            } else {
              probWrapper.style.display = "block";
              aiVetCard.style.display = "block";

              for (const [className, prob] of Object.entries(data.probabilities)) {
                const percentage = typeof prob === "number" ? prob.toFixed(1) : parseFloat(prob).toFixed(1);
                probContainer.innerHTML += `
                <div class="prob-bar-container" style="margin-bottom: 0.6rem;">
                  <div class="prob-labels" style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.2rem;">
                    <span>${className}</span>
                    <span>${percentage}%</span>
                  </div>
                  <div class="prob-track" style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                    <div class="prob-fill" style="width: ${percentage}%; background: var(--primary, #3b82f6); height: 100%; border-radius: 4px;"></div>
                  </div>
                </div>
              `;
              }
              fetchMultilingualSolutions(currentDiagnosis, currentConfidence);
            }

            resultBox.style.display = "block";

            totalScans++;
            localStorage.setItem("totalScans", totalScans);
            localStorage.setItem("lastDiagnosis", currentDiagnosis);
            
            document.getElementById("statTotalScans").innerText = totalScans;
            document.getElementById("statLastDiagnosis").innerText = currentDiagnosis;

          } else {
            alert("Error: " + data.error);
          }
        } catch (err) {
          alert("Could not connect to server. Make sure main.py is running!");
        } finally {
          predictBtn.innerText = "Analyze Image";
          predictBtn.disabled = false;
        }
      }

      async function fetchMultilingualSolutions(disease, confidence) {
        const contentDiv = document.getElementById("aiVetContent");
        contentDiv.innerHTML = `<span class="vet-loading">⏳ Generating multilingual AI treatment guidelines (English, Hausa, Yoruba, Igbo)...</span>`;

        try {
          const res = await fetch("/ask-vet", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              disease: disease,
              confidence: confidence,
              userQuery: "Provide actionable treatment and biosecurity steps in English, Hausa, Yoruba, and Igbo.",
            }),
          });

          const data = await res.json();

          if (res.ok && data.solutions) {
            multiLangSolutions = data.solutions;
            contentDiv.innerText = multiLangSolutions[currentLang] || multiLangSolutions.en;
          } else {
            contentDiv.innerText = data.advice || "Advice generated.";
          }
        } catch (err) {
          contentDiv.innerText = "⚠️ Failed to fetch AI advice. Please check server connectivity.";
        }
      }

      async function askFollowUpVet() {
        const queryInput = document.getElementById("vetQueryInput");
        const askVetBtn = document.getElementById("askVetBtn");
        const responseDiv = document.getElementById("aiVetFollowupResponse");
        const userQuery = queryInput.value.trim();

        if (!currentDiagnosis) {
          alert("Please upload and analyze an image first!");
          return;
        }

        if (!userQuery) {
          alert("Please type a follow-up question!");
          return;
        }

        askVetBtn.innerText = "Thinking...";
        askVetBtn.disabled = true;

        try {
          const res = await fetch("/ask-vet", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              disease: currentDiagnosis,
              confidence: currentConfidence,
              userQuery: userQuery,
            }),
          });

          const data = await res.json();

          if (res.ok && data.solutions) {
            multiLangSolutions = data.solutions;
            responseDiv.innerText = multiLangSolutions[currentLang] || multiLangSolutions.en;
            responseDiv.style.display = "block";
          } else {
            alert("AI Error: " + (data.error || "Unknown error"));
          }
        } catch (err) {
          alert("Could not reach server route.");
        } finally {
          askVetBtn.innerText = "Ask AI";
          askVetBtn.disabled = false;
        }
      }

      function submitFeedback() {
        const feedback = document.getElementById("feedbackText").value;
        if (!feedback.trim()) {
          alert("Please enter your feedback before submitting.");
          return;
        }
        alert("Thank you, Francis Abba! Your feedback has been recorded.");
        document.getElementById("feedbackText").value = "";
      }
    </script>
  </body>
</html>
    """

# predict POST Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a valid image file.")

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size too large. Maximum limit is 10MB.")

    if not contents:
        raise HTTPException(status_code=400, detail="No image file uploaded.")

    try:
        global session
        if session is None:
            raise HTTPException(status_code=503, detail="Model engine is still initializing. Please try again in a few seconds.")

        try:
            input_tensor = await preprocess_image(contents)
        except ValueError:
            raise HTTPException(status_code=422, detail="Corrupted or unreadable image file.")

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        output_logits = outputs[0][0]
        probabilities = softmax(output_logits)

        # Get highest probability class and top probability
        max_prob_index = int(np.argmax(probabilities))
        top_probability = float(probabilities[max_prob_index])

        # Count active classes where probability exceeds ACTIVE_CLASS_THRESHOLD (0.001)
        active_classes = int(np.sum(probabilities > ACTIVE_CLASS_THRESHOLD))

        predicted_class = CLASS_NAMES[max_prob_index]

        # Unknown detection logic matching test script
        is_unknown = False
        if active_classes == 4 or top_probability < CONFIDENCE_THRESHOLD:
            predicted_class = "Unknown Image"
            is_unknown = True

        # Background save image to Supabase if configured with debugging prints
        if supabase:
            try:
                file_extension = file.filename.split(".")[-1] if file.filename else "jpg"
                unique_filename = f"production_capture_{int(time.time())}.{file_extension}"
                
                print(f"Attempting to upload {unique_filename} to Supabase bucket 'kuku'...")
                
                response = supabase.storage.from_("kuku").upload(
                    path=unique_filename,
                    file=contents,
                    file_options={"content-type": file.content_type}
                )
                print(f"Successfully uploaded image to Supabase: {response}")
            except Exception as e:
                print(f"FAILED to background-save image to Supabase: {e}")
        else:
            print("Supabase client is NOT initialized (missing URL or KEY).")

        percentages = [float(round(p * 100, 2)) for p in probabilities]
        current_total = sum(percentages)
        diff = float(round(100.00 - current_total, 2))
        percentages[max_prob_index] = float(round(percentages[max_prob_index] + diff, 2))

        prob_map = {CLASS_NAMES[i]: percentages[i] for i in range(len(CLASS_NAMES))}

        if is_unknown:
            zero_prob_map = {className: 0.0 for className in CLASS_NAMES}
            return {
                "isRecognized": False,
                "prediction": "Unknown Image",
                "confidence": float(round(top_probability * 100, 2)),
                "message": "Image confidence is too low or class probabilities overlap.",
                "probabilities": zero_prob_map
            }

        return {
            "isRecognized": True,
            "prediction": predicted_class,
            "confidence": percentages[max_prob_index],
            "probabilities": prob_map
        }

    except HTTPException as he:
        raise he
    except Exception as inference_err:
        print(f"Inference Failure: {inference_err}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction process.")


class VetRequest(BaseModel):
    disease: str
    confidence: float
    userQuery: str = None

# /ask-vet AI Assistant Endpoint
@app.post("/ask-vet")
async def ask_vet(req: VetRequest):
    if not req.disease:
        raise HTTPException(status_code=400, detail="Diagnosis context is required.")

    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            prompt = f"""You are an expert poultry veterinarian assistant for KUKU-KUKU AI.
Provide practical disease treatment steps and biosecurity guidelines specifically customized for the diagnosed condition.
Return STRICTLY as a valid JSON object with keys "en" (English), "ha" (Hausa), "yo" (Yoruba), and "ig" (Igbo). 
If this is a follow-up query, answer the query while keeping the response split or translated across those exact same 4 keys ("en", "ha", "yo", "ig").

Disease detected: '{req.disease}' with {req.confidence}% confidence. User query: "{req.userQuery or 'Provide immediate action plan and treatment steps.'}" """

            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            
            text_response = response.text.strip()

            if text_response.startswith("```json"):
                text_response = text_response.replace("```json", "", 1).replace("```", "", 1).strip()
            elif text_response.startswith("```"):
                text_response = text_response.replace("```", "", 1).replace("```", "", 1).strip()

            parsed_response = json.loads(text_response)

            if "advice" in parsed_response and "en" not in parsed_response:
                advice_text = parsed_response["advice"]
                parsed_response = {
                    "en": advice_text,
                    "ha": advice_text,
                    "yo": advice_text,
                    "ig": advice_text
                }

            return {"solutions": parsed_response}

        except Exception as ai_err:
            attempt += 1
            wait_time = 55  
            print(f"⚠️ You hit quota (Attempt {attempt}/3). wait for {wait_time}s... before sending another request even to the vet pro")

            if attempt >= max_retries:
                print(f"Free trial exhausted after retries: {ai_err}")
                return {
                    "solutions": {
                        "en": f"Treatment guidelines for {req.disease}: Isolate infected birds immediately, maintain strict biosecurity, provide clean electrolytes, and consult a local veterinarian for targeted medication.",
                        "ha": f"Matakan magani ga {req.disease}: Keɓe tsuntsu masu cuta nan take, kiyaye tsafta, ba da ruwan elektrolit, kuma a tuntubi likitan dabba.",
                        "yo": f"Awọn igbesẹ itọju fun {req.disease}: Ya awọn ẹiyẹ ti o ni arun sọtọ lẹsẹkẹsẹ, ṣọra fun mimọ, ki o si kan si dokita eranko.",
                        "ig": f"Usoro ọgwụgwọ maka {req.disease}: Kewapụ nnụnụ ndị butere ọrịa ozugbo, debe ọcha, ma kpọtụrụ dọkịta anụmanụ."
                    }
                }

            await asyncio.sleep(wait_time)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    print("✅ ONNX Model Engine Initialized Successfully")
    print(f"🚀 KUKU-KUKU Python FastAPI Server active & listening on PORT: {port}")
    print(f"🔗 Local URL: http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
