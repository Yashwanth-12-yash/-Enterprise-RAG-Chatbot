
import { useEffect, useRef, useState } from "react";
import { checkBackend, uploadPDF, sendQuestion } from "./api";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [documentReady, setDocumentReady] = useState(false);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const sendingRef = useRef(false);
  const uploadingRef = useRef(false);
  const messageIdRef = useRef(0);

  // Check backend connection
  useEffect(() => {
    checkBackend()
      .then(() => setBackendStatus("Connected"))
      .catch(() => setBackendStatus("Disconnected"));
  }, []);

  // Scroll to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  function addMessage(role, content) {
    setMessages((previous) => [
      ...previous,
      {
        id: ++messageIdRef.current,
        role,
        content,
      },
    ]);
  }

  // Select PDF
  function handleFileChange(event) {
    const file = event.target.files?.[0];

    setError("");
    setSuccess("");
    setSelectedFile(null);

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      event.target.value = "";
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("File size must not exceed 10 MB.");
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
  }

  // Upload PDF
  async function handleUpload() {
    if (uploadingRef.current || !selectedFile) return;

    uploadingRef.current = true;
    setUploading(true);
    setError("");
    setSuccess("");

    try {
      const response = await uploadPDF(selectedFile);

      setDocumentReady(true);
      setSuccess(
        response.message || "PDF uploaded successfully."
      );

      setMessages([]);
      setQuestion("");
      setSelectedFile(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(err.message || "PDF upload failed.");
    } finally {
      uploadingRef.current = false;
      setUploading(false);
    }
  }

  // Ask a question
  async function handleSendQuestion(event) {
    event.preventDefault();

    if (sendingRef.current) return;

    const userQuestion = question.trim();

    if (!userQuestion) return;

    if (backendStatus !== "Connected") {
      setError("Backend is not connected.");
      return;
    }

    sendingRef.current = true;
    setChatLoading(true);
    setQuestion("");
    setError("");

    addMessage("user", userQuestion);

    try {
      const response = await sendQuestion(userQuestion);

      addMessage(
        "assistant",
        response.answer || "No answer was returned."
      );
    } catch (err) {
      addMessage(
        "assistant",
        err.message || "Unable to generate an answer."
      );
    } finally {
      sendingRef.current = false;
      setChatLoading(false);
    }
  }

  return (
    <div className="app-container">

      <header className="app-header">
        <div>
          <h1>Enterprise RAG Chatbot</h1>
          <p>Ask questions about your uploaded company documents.</p>
        </div>

        <div className="backend-status">
          <span
            className={
              backendStatus === "Connected"
                ? "status-dot connected"
                : "status-dot disconnected"
            }
          />
          Backend: {backendStatus}
        </div>
      </header>

      <main className="main-content">

        <section className="upload-section">
          <h2>Upload Company Document</h2>
          <p>Select a PDF document to build your knowledge base.</p>

          <div className="upload-controls">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              disabled={uploading}
            />

            <button
              type="button"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
            >
              {uploading ? "Processing..." : "Upload PDF"}
            </button>
          </div>

          {selectedFile && (
            <p className="selected-file">
              Selected: {selectedFile.name}
            </p>
          )}

          {success && (
            <p className="success-message">{success}</p>
          )}

          {error && (
            <p className="error-message">{error}</p>
          )}

          {documentReady && (
            <p className="document-status">
              Document processed successfully. You can now ask questions.
            </p>
          )}
        </section>

        <section className="chat-section">
          <h2>Chat with Your Documents</h2>

          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h3>Welcome to Enterprise RAG Chatbot!</h3>
                <p>Upload a PDF and ask questions about its contents.</p>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.role}`}
              >
                <strong>
                  {message.role === "user" ? "You" : "AI Assistant"}
                </strong>
                <p>{message.content}</p>
              </div>
            ))}

            {chatLoading && (
              <div className="loading-message">
                AI is searching your documents and generating an answer...
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          <form
            className="chat-input-form"
            onSubmit={handleSendQuestion}
          >
            <input
              type="text"
              placeholder="Ask a question about your documents..."
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={chatLoading}
            />

            <button
              type="submit"
              disabled={chatLoading || !question.trim()}
            >
              {chatLoading ? "Thinking..." : "Send"}
            </button>
          </form>
        </section>
      </main>

      <footer className="app-footer">
        Enterprise RAG Chatbot | React + FastAPI + FAISS + BM25 + Gemini
      </footer>
    </div>
  );
}

export default App;
