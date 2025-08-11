// chrome.action.onClicked.addListener((tab) => {
//   chrome.scripting.executeScript({
//     target: { tabId: tab.id },
//     func: waitForSelection,
//   });
// });
// function waitForSelection() {
//   document.addEventListener(
//     "mouseup",
//     () => {
//       let selectedText = window.getSelection().toString().trim();
//       if (selectedText) {
//         fetch("http://127.0.0.1:3001/analyze", {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ text: selectedText }),
//         })
//           .then((res) => res.json())
//           .then((data) => {
//             alert(`Sentiment: ${data.label}\nScore: ${data.score}`);
//           })
//           .catch((err) => console.error(err));
//       }
//     },
//     { once: true }
//   ); // removes listener after first selection
// }

chrome.action.onClicked.addListener((tab) => {
    // Inject script to detect text selection
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
            document.addEventListener("mouseup", () => {
                const selectedText = window.getSelection().toString().trim();
                if (selectedText) {
                    // Send selected text to backend
                    fetch("http://127.0.0.1:3001/analyze", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ text: selectedText })
                    })
                    .then(res => res.json())
                    .then(data => {
                        alert(
                            `Label: ${data.label}\nScore: ${data.score}\nText: ${data.text}`
                        );
                    })
                    .catch(err => {
                        console.error("Error calling backend:", err);
                        alert("Could not connect to backend.");
                    });
                }
            });
            alert("Select some text to analyze sentiment.");
        }
    });
});
