import os
import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración
API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")
MODEL = os.getenv("MODEL")

# 1. Leer y dividir el libro en fragmentos

def dividir_en_fragmentos(texto, max_palabras=200):
    palabras = texto.split()
    fragmentos = []
    for i in range(0, len(palabras), max_palabras):
        fragmento = " ".join(palabras[i:i+max_palabras])
        fragmentos.append(fragmento)
    return fragmentos

# 2. Generar embeddings para cada fragmento

def obtener_embeddings(fragmentos, modelo_emb):
    return modelo_emb.encode(fragmentos, show_progress_bar=True)

# 3. Indexar los embeddings con FAISS

def crear_indice_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 4. Buscar los fragmentos más relevantes

def buscar_fragmentos(pregunta, modelo_emb, index, fragmentos, k=5):
    emb_pregunta = modelo_emb.encode([pregunta])
    D, I = index.search(emb_pregunta, k)
    return [fragmentos[i] for i in I[0]]

# 5. Construir prompt y consultar DeepSeek

def consultar_deepseek(contexto, pregunta):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    prompt = f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Eres un asistente profesional y solo puedes responder usando el contexto proporcionado."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}"

# Ejemplo de uso con soporte para PDF o TXT
def extraer_texto_pdf(ruta_pdf):
    import pdfplumber
    texto = ""
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto += pagina.extract_text() or ""
            texto += "\n"
    return texto

if __name__ == "__main__":
    import numpy as np
    archivo = input("Nombre del archivo del libro (.txt o .pdf): ").strip()
    if archivo.lower().endswith(".pdf"):
        txt_cache = archivo[:-4] + ".txt"
    elif archivo.lower().endswith(".txt"):
        txt_cache = archivo
    else:
        print("Formato de archivo no soportado. Usa .txt o .pdf")
        exit(1)

    # Extraer texto solo si es necesario
    if archivo.lower().endswith(".pdf") and not os.path.exists(txt_cache):
        print("Extrayendo texto del PDF...")
        texto = extraer_texto_pdf(archivo)
        with open(txt_cache, "w", encoding="utf-8") as f:
            f.write(texto)
    else:
        with open(txt_cache, "r", encoding="utf-8") as f:
            texto = f.read()

    # Fragmentos
    fragmentos = dividir_en_fragmentos(texto)
    print(f"Fragmentos generados: {len(fragmentos)}")

    # Archivos de cache para embeddings e índice
    emb_cache = txt_cache + ".embeddings.npy"
    faiss_cache = txt_cache + ".faiss"

    modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")

    # Si existen los archivos de embeddings e índice, cargarlos
    if os.path.exists(emb_cache) and os.path.exists(faiss_cache):
        print("Cargando embeddings e índice FAISS desde cache...")
        embeddings = np.load(emb_cache)
        index = faiss.read_index(faiss_cache)
    else:
        print("Generando embeddings e índice FAISS...")
        embeddings = obtener_embeddings(fragmentos, modelo_emb)
        embeddings = np.array(embeddings).astype("float32")
        index = crear_indice_faiss(embeddings)
        np.save(emb_cache, embeddings)
        faiss.write_index(index, faiss_cache)

    # Preguntar
    pregunta = input("¿Sobre qué tema quieres preguntar?: ")
    fragmentos_relevantes = buscar_fragmentos(pregunta, modelo_emb, index, fragmentos, k=5)
    contexto = "\n".join(fragmentos_relevantes)
    respuesta = consultar_deepseek(contexto, pregunta)
    print("\nRespuesta de DeepSeek:\n", respuesta)
