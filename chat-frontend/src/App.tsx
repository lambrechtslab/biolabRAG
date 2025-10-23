import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";   // 👈 import
import remarkGfm from "remark-gfm";           // optional for tables, checkboxes

import "./App.css"; // 👈 Import your CSS file

interface Message {
  sender: "user" | "bot";
  text: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedOption, setSelectedOption] = useState("localFirst");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
	      text: input,
	      option: selectedOption,
	  }),
      });

      if (!res.ok) throw new Error("Server error");
      const data = await res.json();

      const botMsg: Message = { sender: "bot", text: data.reply };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Failed to fetch reply." },
      ]);
    }
  };

  const resetChat = async () => {
    setMessages([]);
    // setInput("");

    try {
      const res = await fetch("http://localhost:8000/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        throw new Error("Failed to notify backend");
      }
    } catch (err) {
      // The UI is already reset even if the backend notification fails.
      console.error("Unable to notify backend about reset", err);
    }
  };
			      
  return (
    <div className="chat-container">
      <h4>Demo AI RAG system for DILA @Jieyi</h4>
      {/* Chat messages */}
      <div className="chat-messages">
        {messages.map((msg, i) => (
	<div
          key={i}
          className={`message ${msg.sender === "user" ? "user" : "bot"}`}
	  >
	  <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {msg.text}
          </ReactMarkdown>
        </div>
        ))}
        <div ref={bottomRef} />
      </div>
      {/* Input box */}
      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        placeholder="Type your message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
      <table width="100%">
	<td align="left">
	  <div>
	    <button className="reset-button" onClick={resetChat}>
              New Chat
	    </button>
	  </div>
	</td>
	<td align="right">
	  {/* Radio button group */}
	  <div className="radio-group">
	    <label>
	      Knowledge Source:
	    </label>
	    <label>
              <input
		type="radio"
		value="localFirst"
		checked={selectedOption === "localFirst"}
		onChange={(e) => setSelectedOption(e.target.value)}
              />
              Local First
            </label>
            <label>
              <input
		type="radio"
		value="localOnly"
		checked={selectedOption === "localOnly"}
		onChange={(e) => setSelectedOption(e.target.value)}
              />
              Local Only
            </label>
            <label>
              <input
		type="radio"
		value="PubMedOnly"
		checked={selectedOption === "PubMedOnly"}
		onChange={(e) => setSelectedOption(e.target.value)}
              />
              PubMed Only
            </label>
	  </div>
	</td>
      </table>
    </div>
  );
}
