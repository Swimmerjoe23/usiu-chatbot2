import React, { useState, useEffect } from 'react';
import './App.css';
import usiuLogo from './assets/usiu-logo.png';
import faqIcon from './assets/FAQs.png';
import settingsIcon from './assets/settings.png';
import resourcesIcon from './assets/resources.png';
import contactIcon from './assets/contact.png';
import homeIcon from './assets/home.png';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const chatEndRef = React.useRef(null);

  // On component mount, load saved chat history from localStorage (if any)
  useEffect(() => {
    const storedHistory = localStorage.getItem('chatHistory');
    if (storedHistory) {
      setMessages(JSON.parse(storedHistory));
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Function to send a message to the backend (or simulate a response)
  const sendMessageToBackend = async (userMessage, currentChatHistory) => {
    const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
    if (!backendUrl) {
      return "Backend URL not set. Please check your .env configuration.";
    }
    try {
      const formattedHistory = currentChatHistory.map(m => [m.sender, m.text]);
      const response = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: userMessage, chat_history: formattedHistory }),
      });
      const data = await response.json();
      const newHistory = data.chat_history;
      const lastMessage = newHistory[newHistory.length - 1];
      return lastMessage[1]; // Return latest bot reply
    } catch (error) {
      console.error("Backend call failed:", error);
      return "Sorry, something went wrong.";
    }
  };

  const handleSend = async () => {
    if (input.trim() !== '') {
      // Append the user message
      setMessages([...messages, { sender: 'user', text: input }]);
      // Get reply from backend (or simulation)
      const reply = await sendMessageToBackend(input, [...messages, { sender: 'user', text: input }]);
      setMessages(prev => [...prev, { sender: 'bot', text: reply }]);
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };

  const clearChat = () => {
    setMessages([]);
    localStorage.removeItem('chatHistory');
  };

  return (
    <div className="app-container">
      {/* Sidebar: Chat History & Quick Icons Pill */}
      <div className={`sidebar ${sidebarExpanded ? 'expanded' : ''}`}>
        <div className="nav-pill">
          <div className="quick-icons">
            <img 
              src={homeIcon} 
              alt="Toggle sidebar" 
              onClick={toggleSidebar}
              style={{ cursor: 'pointer' }}
            />
            <a 
              href="https://www.usiu.ac.ke/331/frequently-asked-questions-faqs/" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              <img 
                src={faqIcon} 
                alt="FAQs" 
                style={{ cursor: 'pointer' }}
              />
            </a>
            <a 
              href="https://www.usiu.ac.ke/resource-downloads/" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              <img 
                src={resourcesIcon} 
                alt="Resources" 
                style={{ cursor: 'pointer' }}
              />
            </a>
            <a 
              href="https://www.usiu.ac.ke/contacts-directory/" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              <img 
                src={contactIcon} 
                alt="Contact" 
                style={{ cursor: 'pointer' }}
              />
            </a>
            <img 
              src={settingsIcon} 
              alt="Settings" 
              style={{ cursor: 'pointer' }}
            />
          </div>
        </div>
        <div className="chat-history-container">
          <div className="chat-history-title">Chat History</div>
          <button 
            onClick={clearChat} 
            style={{ margin: '10px', padding: '5px 10px', cursor: 'pointer' }}
          >
            Clear Chat
          </button>
          <div className="chat-history">
            {[1, 2, 3, 4].map((item) => (
              <div 
                key={item}
                className="chat-item"
                style={{ transition: 'all 0.2s' }}
              >
                📄 Chat {item}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Container */}
      <div className="chat-container">
        <header className="chat-header">
          <img src={usiuLogo} alt="USIU Logo" className="logo" />
          <div className="divider"></div>
          <h1>USIU Chatbot</h1>
        </header>
        <div className="chat-area" onKeyPress={handleKeyPress}>
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
        <div className="input-area">
          <input 
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            aria-label="Type your message"
          />
          <button 
            onClick={handleSend}
            aria-label="Send message"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;