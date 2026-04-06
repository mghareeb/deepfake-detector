const API_URL = import.meta.env.VITE_API_URL || "";

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || "Prediction failed");
  }

  return response.json();
}
