# E-Commerce Product Classification — Word2Vec + ANN

A Streamlit app that classifies e-commerce product descriptions from typed text or microphone audio.

## Pipeline
Text/audio → speech transcription (voice only) → cleaning/tokenization → Word2Vec document vector (100 dimensions) → ANN → category and confidence.

## Project files
- `app.py`: Streamlit user interface and inference.
- `train_model.py`: trains Word2Vec, LabelEncoder, and ANN, then saves artifacts.
- `requirements.txt`: Python dependencies.
- `artifacts/`: generated model files (created by training; not included in this starter package).
- `ecommerceDataset.csv`: original dataset (keep local; not included).

## 1. Create environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Add the dataset and train
Place your original `ecommerceDataset.csv` in this folder. It should have the same two-column layout used in the notebook: category, description, without a header row.

```powershell
python train_model.py --data ecommerceDataset.csv
```

This creates:
- `artifacts/ann_model.keras`
- `artifacts/word2vec.model`
- `artifacts/label_encoder.joblib`

Do not run the app until these files exist.

## 3. Run locally
```powershell
streamlit run app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501). Allow microphone access in your browser for voice input.

Voice transcription uses Google's online Speech Recognition service through the SpeechRecognition package, so voice recognition requires internet access. Text prediction works without the transcription service.

## 4. Push to GitHub
Create an empty GitHub repository, then from this folder run:
```powershell
git init
git add .
git commit -m "Build Streamlit e-commerce product classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Replace the remote URL with your own repository URL.

## 5. Deploy on Streamlit Community Cloud
1. Push the project to GitHub.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Select **Create app**.
4. Select your repository and `main` branch.
5. Set the main file path to `app.py`.
6. Select Python 3.11 in Advanced settings if prompted, then deploy.

### Model artifact hosting
The `artifacts/` directory is ignored by Git because model files can be large. For deployment, the app must still be able to access all three artifacts. If they are small enough for your repository, remove the `artifacts/` line from `.gitignore`, add the three generated files, and push them. Otherwise, store them in an appropriate model/file hosting service and modify `load_artifacts()` to download them securely at startup. Never commit credentials or private tokens.

## Notes
- The dataset and model artifacts are not included in this starter ZIP. Train them from your original dataset before deploying.
- The training script preserves the Word2Vec and ANN hyperparameters from the supplied notebook.
- The app uses whitespace tokenization after the notebook's cleaning step; for cleaned alphabetic text this gives the same word tokens as the notebook's NLTK tokenizer.
- Speech recognition can fail with noisy audio, unsupported language, or no internet connection.
