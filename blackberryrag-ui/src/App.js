import React, { useState } from "react";

function App() {
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");
    const [sources, setSources] = useState([]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim()) {
            setResponse("Error: Please enter a query");
            return;
        }
        const payload = { query: query };
        console.log("Sending:", JSON.stringify(payload));
        try {
            const res = await fetch("http://localhost:8000/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(`Server error: ${res.status} - ${JSON.stringify(errorData)}`);
            }
            const data = await res.json();
            setResponse(data.answer || data.error || "No response content");
            setSources(data.sources || []);
        } catch (error) {
            setResponse(`Error: ${error.message}`);
            setSources([]);
        }
    };

    return (
        <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
            <h1>BlackBerry RAG Query System</h1>
            <form onSubmit={handleSubmit}>
                <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your query"
                    style={{ width: "70%", padding: "8px", marginRight: "10px" }}
                />
                <button type="submit" style={{ padding: "8px 16px" }}>Submit</button>
            </form>
            <div style={{ marginTop: "20px" }}>
                <strong>Response:</strong>
                <p>{response}</p>
                <strong>Sources:</strong>
                {sources.length > 0 ? (
                    <ul>
                        {sources.map((source, index) => (
                            <li key={index}>{source}</li>
                        ))}
                    </ul>
                ) : (
                    <p>No sources provided</p>
                )}
            </div>
        </div>
    );
}

export default App;