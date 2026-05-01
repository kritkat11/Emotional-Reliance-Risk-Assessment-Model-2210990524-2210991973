// Character counter — updates as user types
document.addEventListener("DOMContentLoaded", () => {
    const ta      = document.getElementById("userInput");
    const counter = document.getElementById("charCount");

    if (ta && counter) {
        ta.addEventListener("input", () => {
            const len = ta.value.length;
            counter.textContent = `${len} character${len !== 1 ? "s" : ""}`;
        });
        // Ctrl + Enter to submit
        ta.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && e.ctrlKey) analyzeRisk();
        });
    }
});


async function analyzeRisk() {
    const input = document.getElementById("userInput").value.trim();
    const btn   = document.getElementById("analyzeBtn");

    // Validate input before sending to server
    if (!input) {
        showError("Please enter some text before analyzing.");
        return;
    }
    if (input.length < 5) {
        showError("Please enter a longer message for accurate analysis.");
        return;
    }

    // Reset all cards and show loading state
    hide("resultCard");
    hide("errorCard");
    hide("warningBox");
    btn.disabled = true;
    setText("btnText", "Analyzing...");
    show("spinner");

    try {
        // Send text to Flask /predict route
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: input })
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || "Something went wrong. Please try again.");
            return;
        }

        displayResult(data);

    } catch (err) {
        showError("Cannot connect to server. Make sure app.py is running.");
    } finally {
        // Always re-enable button after request finishes
        btn.disabled = false;
        setText("btnText", "Analyze Risk");
        hide("spinner");
    }
}


function displayResult(data) {
    const emoji = { Low: "🟢", Medium: "🟠", High: "🔴" };

    // Set risk badge text and colour class
    const badge = document.getElementById("riskBadge");
    badge.textContent = `${emoji[data.risk_level]}  ${data.risk_level.toUpperCase()} RISK`;
    badge.className   = "risk-badge " + data.risk_level.toLowerCase();

    // Animate confidence bar
    const fill = document.getElementById("confidenceFill");
    setTimeout(() => { fill.style.width = data.confidence + "%"; }, 50);
    setText("confidenceVal", data.confidence + "%");

    // Set description text
    setText("descriptionText", data.description);

    // Animate probability bars for Low, Medium, High
    setBar("barLow",    "valLow",    data.probabilities["Low"]);
    setBar("barMedium", "valMedium", data.probabilities["Medium"]);
    setBar("barHigh",   "valHigh",   data.probabilities["High"]);

    // Populate tips list
    const tipsList = document.getElementById("tipsList");
    tipsList.innerHTML = "";
    if (data.tips && data.tips.length) {
        data.tips.forEach(tip => {
            const li = document.createElement("li");
            li.textContent = tip;
            tipsList.appendChild(li);
        });
    }

    // Show warning if model confidence is low
    if (data.warning) {
        setText("warningText", data.warning);
        show("warningBox");
    }

    show("resultCard");
    document.getElementById("resultCard")
        .scrollIntoView({ behavior: "smooth", block: "nearest" });
}


function setBar(barId, valId, pct) {
    setTimeout(() => {
        document.getElementById(barId).style.width = pct + "%";
    }, 80);
    setText(valId, pct + "%");
}

function showError(msg) {
    setText("errorText", msg);
    show("errorCard");
}

// Helper functions
function show(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove("hidden");
}
function hide(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add("hidden");
}
function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}