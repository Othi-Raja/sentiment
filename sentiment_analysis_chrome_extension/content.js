// Create overlay for showing results
const overlay = document.createElement("div");
overlay.style.position = "fixed";
overlay.style.top = "10px";
overlay.style.right = "10px";
overlay.style.background = "white";
overlay.style.border = "1px solid #ccc";
overlay.style.padding = "10px";
overlay.style.borderRadius = "8px";
overlay.style.boxShadow = "0px 2px 8px rgba(0,0,0,0.3)";
overlay.style.zIndex = 999999;
overlay.style.display = "none";
document.body.appendChild(overlay);

// Handle drag-over (prevent default to allow drop)
document.addEventListener("dragover", (e) => {
    e.preventDefault();
});

// Handle drop
document.addEventListener("drop", async (e) => {
    e.preventDefault();

    let text = e.dataTransfer.getData("text/plain");
    if (!text) return;

    overlay.innerText = "Analyzing...";
    overlay.style.display = "block";

    try {
        let res = await fetch("http://127.0.0.1:3001/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        let data = await res.json();
        overlay.innerText = `Sentiment: ${data.sentiment} (Score: ${data.score})`;
    } catch (err) {
        overlay.innerText = "Error: Could not connect to backend";
    }
});
