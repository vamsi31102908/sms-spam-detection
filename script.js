function checkSpam() {
    const message = document.getElementById("message").value;
    const resultElement = document.getElementById("result");
    const loader = document.getElementById("loader");

    if (!message.trim()) {
        resultElement.innerText = "⚠️ Please enter a message!";
        resultElement.className = "result";
        return;
    }

    resultElement.innerText = "";
    loader.style.display = "block";

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        loader.style.display = "none";
        resultElement.innerText = "Prediction: " + data.prediction;
        resultElement.className = "result " + (data.prediction === "Spam" ? "spam" : "ham");
    })
    .catch(error => {
        loader.style.display = "none";
        resultElement.innerText = "⚠️ Error occurred!";
        resultElement.className = "result";
        console.error("Error:", error);
    });
}
