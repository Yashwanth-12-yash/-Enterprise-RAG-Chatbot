
const API_URL = "http://127.0.0.1:8000";

async function handleResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail || data.message || "Backend request failed."
    );
  }

  return data;
}

export async function checkBackend() {
  const response = await fetch(`${API_URL}/health`);
  return handleResponse(response);
}

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/upload-pdf`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
}

export async function sendQuestion(question) {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  return handleResponse(response);
}
