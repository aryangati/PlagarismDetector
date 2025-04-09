document.addEventListener('DOMContentLoaded', () => {
    const smsInput = document.getElementById('smsInput');
    const predictButton = document.getElementById('predictButton');
    const resultArea = document.getElementById('resultArea');
    const resultText = document.getElementById('resultText');
    const loader = document.getElementById('loader');

    const apiUrl = 'http://localhost:5001/predict';

    predictButton.addEventListener('click', handlePrediction);
    smsInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handlePrediction();
        }
    });

    async function handlePrediction() {
        const smsText = smsInput.value.trim();

        clearResult();
        loader.classList.remove('hidden');
        predictButton.disabled = true;

        if (smsText === '') {
            displayResult('Please enter an SMS message.', 'error');
            loader.classList.add('hidden');
            predictButton.disabled = false;
            return;
        }

        try {
            const requestUrl = `${apiUrl}?sms=${encodeURIComponent(smsText)}`;
            const response = await fetch(requestUrl);

            if (!response.ok) {
                let errorMsg = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorData.detail || errorMsg;
                } catch (e) { /* Ignore */}
                throw new Error(errorMsg);
            }

            const data = await response.json();

            if (data && data.predicted_label !== undefined && data.is_spam !== undefined) {
                const displayLabel = data.predicted_label; // "Spam" or "Not Spam"

                const cssClass = data.is_spam ? 'spam' : 'ham';

                displayResult(`Result: ${displayLabel === 'Spam' ? "Plagiarised" : "Not Plagiarised"}`, cssClass);
            } else {
                throw new Error('Invalid response format from API.');
            }

        } catch (error) {
            console.error('Prediction Error:', error);
            displayResult(`Error: ${error.message}`, 'error');
        } finally {
            loader.classList.add('hidden');
            predictButton.disabled = false;
        }
    }

    function displayResult(message, type) {
        resultText.textContent = message;
        resultText.classList.remove('spam', 'ham', 'error', 'visible');
        if (type) {
            resultText.classList.add(type); 
        }
        requestAnimationFrame(() => {
            resultText.classList.add('visible');
        });
    }

    function clearResult() {
        resultText.textContent = '';
        resultText.classList.remove('spam', 'ham', 'error', 'visible');
    }
});