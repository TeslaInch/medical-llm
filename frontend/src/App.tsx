import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Activity, Plus, User, Settings, ExternalLink, X } from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.PROD ? '/predict' : 'https://teslainch-scd-medical-llm.hf.space/predict';

interface Citation {
  source: string;
  content: string;
  relevance_score: number;
}

interface Message {
  id: string;
  role: 'user' | 'ai';
  content: string;
  citations?: Citation[];
  confidence?: number;
  latencyMs?: number;
}

const SUGGESTIONS = [
  {
    title: "What is Sickle Cell Disease?",
    subtitle: "A brief overview of the condition"
  },
  {
    title: "Management of acute pain",
    subtitle: "Guidelines for pain crises"
  },
  {
    title: "Pediatric fever protocol",
    subtitle: "Steps for children presenting with fever"
  },
  {
    title: "Blood transfusion criteria",
    subtitle: "When is transfusion indicated?"
  }
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [caseContext, setCaseContext] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showBanner, setShowBanner] = useState(true);
  const [loadingStep, setLoadingStep] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const LOADING_MESSAGES = [
    "Vectorizing clinical query...",
    "Searching ChromaDB for relevant ASH guidelines...",
    "Generating response on CPU cluster... (This may take up to 5-10 minutes on the free tier)",
    "Please do not refresh the page. Processing..."
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isLoading) {
      setLoadingStep(0);
      interval = setInterval(() => {
        setLoadingStep(prev => Math.min(prev + 1, LOADING_MESSAGES.length - 1));
      }, 30000); // Cycle text every 30 seconds
    }
    return () => clearInterval(interval);
  }, [isLoading]);
  
  // Extract all citations from current messages for the right panel
  const allCitations = messages.flatMap(m => m.citations || []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  const handleClearChat = () => {
    setMessages([]);
  };

  const handleInputResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = '56px';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 150)}px`;
  };

  const sendQuestion = async (questionText: string) => {
    if (!questionText.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: questionText.trim()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const historyPayload = messages.map(m => ({ role: m.role, content: m.content }));

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage.content,
          case: caseContext || undefined,
          history: historyPayload
        })
      });

      if (!response.ok) throw new Error("API request failed");

      const data = await response.json();
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'ai',
        content: data.answer,
        citations: data.citations,
        confidence: data.confidence,
        latencyMs: data.latency_ms
      };
      
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'ai',
        content: "I'm sorry, I encountered an error connecting to the inference server. Please ensure the HuggingFace Space is running."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    sendQuestion(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw' }}>
      {/* Demo Banner */}
      {showBanner && (
        <div className="demo-banner">
          <span>
            ⚠️ <strong>Demo Environment:</strong> This application is currently running on a Free-Tier CPU to minimize cloud costs. Responses may take 3-10 minutes. For enterprise GPU deployment inquiries (2-second latency), please contact the developer via <a href="mailto:davidabuh369@gmail.com">Email</a>.
          </span>
          <button onClick={() => setShowBanner(false)}><X size={16} /></button>
        </div>
      )}

      <div className="app-container" style={{ height: showBanner ? 'calc(100vh - 40px)' : '100vh' }}>
        
        {/* 1. Left Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header" style={{ lineHeight: '1.4' }}>
          Sickle Cell Disease<br/>Clinical Assistant
        </div>
        
        <button className="new-chat-btn" onClick={handleClearChat}>
          <Plus size={18} /> New Chat
        </button>
      </aside>

      {/* 2. Main Chat Area */}
      <main className="main-content">
        
        {messages.length === 0 ? (
          <>
            <div className="watermark">
              Sickle Cell<br/>Clinical AI
            </div>
            
            <div style={{ marginTop: 'auto', zIndex: 10 }}>
              <div className="suggestions-grid">
                {SUGGESTIONS.map((s, i) => (
                  <div key={i} className="suggestion-card" onClick={() => sendQuestion(s.title)}>
                    <div className="title">{s.title}</div>
                    <div className="subtitle">{s.subtitle}</div>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="message-list">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <div className="avatar">
                  {msg.role === 'ai' ? <Activity size={20} /> : <User size={20} />}
                </div>
                <div className="message-content">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  
                  {msg.latencyMs && (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                      ⏱️ {(msg.latencyMs / 1000).toFixed(2)}s
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="message ai">
                <div className="avatar"><Activity size={20} /></div>
                <div className="message-content typing-indicator">
                  <div className="dots-wrapper">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                  <div className="loading-status-text">
                    {LOADING_MESSAGES[loadingStep]}
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}

        <div className="input-wrapper">
          <form className="chat-form" onSubmit={handleSend}>
            <textarea 
              value={input}
              onChange={handleInputResize}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              rows={1}
            />
            <button type="submit" className="send-btn" disabled={!input.trim() || isLoading}>
              <Send size={18} />
            </button>
          </form>
        </div>
      </main>

      {/* 3. Right Panel */}
      <aside className="right-panel">
        
        <div className="panel-section">
          <h3>Patient Context</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: '1.4', marginBottom: '0.25rem' }}>
            Provide patient details here. The AI will read this chart and provide highly personalized recommendations based on ASH guidelines.
          </p>
          <textarea 
            className="context-input"
            placeholder="e.g. 6yo male, HbSS, pain score 8/10 in lower back..."
            value={caseContext}
            onChange={(e) => setCaseContext(e.target.value)}
          />
        </div>

        <div className="panel-section">
          <h3>Generated Citations</h3>
          <div className="citation-list">
            {allCitations.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Links to guidelines and medical documents will appear here.
              </p>
            ) : (
              allCitations.map((cit, idx) => (
                <div key={idx} className="citation-item">
                  <div className="citation-source">
                    <ExternalLink size={14} /> {cit.source}
                  </div>
                  <div style={{ fontSize: '0.75rem' }}>
                    Relevance: {(cit.relevance_score * 100).toFixed(1)}%
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
        
      </aside>

      </div>
    </div>
  );
}

export default App;
