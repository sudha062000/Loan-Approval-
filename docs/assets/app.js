// frontend/assets/app.js

const form = document.getElementById("loan-form");
const statusEl = document.getElementById("status");
const resultCard = document.getElementById("result-card");
const resultText = document.getElementById("result-text");
const resultProb = document.getElementById("result-prob");

// Backend URL
const API_BASE = "http://127.0.0.1:5000";

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  statusEl.textContent = "⏳ Predicting...";
  resultCard.classList.add("hidden");

  const formData = new FormData(form);
  const payload = {};

  // Collect values
  formData.forEach((value, key) => {
    payload[key] = value;
  });

  // 🔹 Validate required fields
  for (let key in payload) {
    if (payload[key] === "" || payload[key] === null) {
      statusEl.textContent = "⚠️ Please fill all fields before predicting.";
      return;
    }
  }

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error("Request failed");
    }

    const data = await res.json();

    const approved = data.approved;
    const prob = data.probability;
    const modelUsed = data.model_used || "Unknown";

    // Main prediction text
    resultText.textContent = approved
      ? "✅ Loan is likely to be APPROVED"
      : "❌ Loan is likely to be REJECTED";

    // Display detailed probability & model name
    resultProb.innerHTML = `
      Model confidence: <strong>${(prob * 100).toFixed(2)}%</strong><br>
      Model used: <strong>${modelUsed}</strong>
    `;

    resultCard.classList.remove("hidden");
    statusEl.textContent = "Done.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "❌ Error: Could not connect to backend.";
  }
});
