import React, { useState } from "react";

const PredictForm = () => {
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState("English");
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    const query = new URLSearchParams({ language }).toString();
    const res = await fetch(`http://localhost:8000/predict?${query}`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResponse(data);
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Select Language:</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="English">English</option>
            <option value="Hindi">Hindi</option>
            <option value="Spanish">Spanish</option>
            {/* Add more languages as needed */}
          </select>
        </div>
        <div>
          <label>Upload Image:</label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            accept="image/*"
          />
        </div>
        <button type="submit">Submit</button>
      </form>
      {response && (
        <div>
          <h3>Prediction Result:</h3>
          <p>Class: {response.class}</p>
          <p>Confidence: {response.confidence}</p>
          <p>Summary: {response.summary}</p>
        </div>
      )}
    </div>
  );
};

export default PredictForm;
