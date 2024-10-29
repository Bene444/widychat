# Mengimpor pustaka yang diperlukan
import streamlit as st  # Untuk membuat antarmuka pengguna berbasis web
import os  # Untuk berinteraksi dengan sistem operasi
from PyPDF2 import PdfReader  # Untuk membaca file PDF
from langchain.chat_models import ChatOpenAI  # Untuk model chat dari OpenAI
from langchain.llms import OpenAI  # Untuk menggunakan model LLM dari OpenAI
from dotenv import load_dotenv  # Untuk memuat variabel lingkungan dari file .env
from langchain.text_splitter import CharacterTextSplitter  # Untuk membagi teks menjadi potongan-potongan kecil
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # Untuk embedding menggunakan Hugging Face
from langchain.vectorstores import FAISS  # Untuk menyimpan dan mencari data berbasis vektor
from langchain.chains import ConversationalRetrievalChain  # Untuk membuat alur percakapan
from langchain.memory import ConversationBufferMemory  # Untuk menyimpan riwayat percakapan
from streamlit_chat import message  # Untuk menampilkan pesan dalam antarmuka Streamlit
from langchain.callbacks import get_openai_callback  # Untuk mendapatkan callback dari OpenAI
def main():
    # Memuat variabel lingkungan dari file .env
    load_dotenv()
    
    # Mengambil API key langsung dari file .env
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:  # Memeriksa apakah API key ada
        st.error("API key OpenAI tidak ditemukan. Pastikan .env telah dikonfigurasi dengan benar.")
        return

    st.set_page_config(page_title="Chat With files")  # Mengatur judul halaman
    st.header("Selamat datang di WidyChat")  # Menampilkan header di antarmuka pengguna

    # Memastikan state sesi untuk menyimpan informasi percakapan
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # Inisialisasi percakapan
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  # Inisialisasi riwayat chat
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None  # Status proses pemrosesan

    # Mengambil file PDF dari folder yang sudah disiapkan
    folder_path = "data"  # Pastikan folder ini ada dan berisi file PDF
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    if len(pdf_files) == 0:
        st.error("Tidak ada file PDF yang ditemukan di folder 'data'.")
        return

    selected_pdf = pdf_files[0]  # Ambil file PDF pertama yang ditemukan
    st.write(f"File yang diproses: {selected_pdf}")

    # Proses file PDF
    files_text = get_files_text(os.path.join(folder_path, selected_pdf))  # Mendapatkan teks dari file PDF
    text_chunks = get_text_chunks(files_text)  # Membagi teks menjadi potongan-potongan kecil
    vetorestore = get_vectorstore(text_chunks)  # Membuat penyimpanan vektor dari potongan teks

    # Menginisialisasi percakapan dengan model OpenAI
    st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
    st.session_state.processComplete = True  # Menandai bahwa proses telah selesai

    if st.session_state.processComplete:
        user_question = st.chat_input("Kirim pesan ke WidyChat")  # Input pertanyaan dari pengguna
        if user_question:  # Jika ada pertanyaan dari pengguna
            handel_userinput(user_question)  # Memproses input pengguna


def get_files_text(pdf_file_path):
    # Membaca file PDF dari jalur yang diberikan
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:  # Iterasi untuk setiap halaman di PDF
            text += page.extract_text()  # Mengambil teks dari halaman
    return text  # Mengembalikan teks dari PDF


def get_text_chunks(text):
    # Membagi teks menjadi potongan-potongan kecil menggunakan karakter pemisah, ukuran potongan, dan tumpang tindih
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,  # Ukuran potongan
        chunk_overlap=100,  # Tumpang tindih antara potongan
        length_function=len  # Fungsi untuk menghitung panjang teks
    )
    chunks = text_splitter.split_text(text)  # Memisahkan teks
    return chunks  # Mengembalikan potongan teks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()  # Membuat embeddings menggunakan Hugging Face
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)  # Membuat penyimpanan vektor dari potongan teks
    return knowledge_base  # Mengembalikan penyimpanan vektor


def get_conversation_chain(vetorestore, openai_api_key):
    # Menginisialisasi model OpenAI dengan API key dan model yang ditentukan
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)  # Menyimpan riwayat percakapan
    # Membuat rantai percakapan dengan model dan penyimpanan vektor
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),  # Mengambil dari penyimpanan vektor
        memory=memory
    )
    return conversation_chain  # Mengembalikan rantai percakapan


def handel_userinput(user_question):
    with get_openai_callback() as cb:  # Menggunakan callback untuk OpenAI
        response = st.session_state.conversation({'question': user_question})  # Mengirim pertanyaan ke model percakapan
    st.session_state.chat_history = response['chat_history']  # Menyimpan riwayat chat ke dalam state sesi

    response_container = st.container()  # Membuat wadah untuk menampilkan respons

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):  # Iterasi untuk setiap pesan dalam riwayat
            if i % 2 == 0:  # Jika indeks genap
                message(messages.content, is_user=True, key=str(i))  # Tampilkan pesan pengguna
            else:  # Jika indeks ganjil
                message(messages.content, key=str(i))  # Tampilkan pesan dari sistem


if __name__ == '__main__':
    main()
