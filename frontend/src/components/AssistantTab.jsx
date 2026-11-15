import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, 
  User, 
  Send, 
  Sparkles, 
  Trash2, 
  FileText, 
  ChevronDown, 
  ChevronUp, 
  AlertCircle,
  Lightbulb,
  Zap
} from 'lucide-react';
import { api } from '../services/api';

export default function AssistantTab({ state, horizonDays }) {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: `Hello! I am your AI Energy Assistant grounded with domain knowledge, machine learning model benchmarks, and weather analytics for Indian states. Ask me about electricity demand patterns, cooling load surges, extreme heat sensitivity, or model accuracy metrics.`,
      sources: []
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const messagesEndRef = useRef(null);

  // Fetch dynamic suggestions when state changes
  useEffect(() => {
    let isMounted = true;
    api.getSuggestions(state)
      .then((res) => {
        if (isMounted && res?.suggestions) {
          setSuggestions(res.suggestions);
        }
      })
      .catch((err) => console.error('Failed to load prompt suggestions:', err));

    return () => {
      isMounted = false;
    };
  }, [state]);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSendMessage = async (textToSend) => {
    const query = (textToSend || inputMessage).trim();
    if (!query || loading) return;

    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      content: query
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage('');
    setLoading(true);

    // Format chat history for backend
    const historyPayload = messages.map((m) => ({
      role: m.role,
      content: m.content
    }));

    try {
      const res = await api.sendChatMessage({
        message: query,
        state,
        horizonDays,
        chatHistory: historyPayload
      });

      const assistantMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.reply || 'No response generated.',
        sources: res.sources || [],
        isOutOfScope: res.is_out_of_scope
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `⚠️ Error: ${err.message || 'Failed to reach AI assistant service. Please check your backend and connection.'}`,
          sources: []
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: 'welcome-reset',
        role: 'assistant',
        content: `Chat history cleared. How can I help you analyze electricity forecasting for ${state}?`,
        sources: []
      }
    ]);
  };

  const toggleSourceExpand = (msgId) => {
    setExpandedSources((prev) => ({
      ...prev,
      [msgId]: !prev[msgId]
    }));
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* State Context Banner */}
      <div className="flex items-center justify-between bg-white dark:bg-executive-cardDark p-4 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-none bg-executive-navy/10 dark:bg-executive-cyan/15 text-executive-navy dark:text-executive-cyan">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
              AI Energy Assistant
              <span className="text-xs font-normal text-slate-500">| Grounded RAG Agent</span>
            </h2>
            <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
              Active Context: <strong className="text-executive-navy dark:text-executive-cyan">{state}</strong> ({horizonDays}d forecast window)
            </p>
          </div>
        </div>

        <button
          onClick={handleClearChat}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-rose-600 dark:text-slate-400 dark:hover:text-rose-400 px-3 py-1.5 rounded-none hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          title="Clear Conversation"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Clear Chat</span>
        </button>
      </div>

      {/* Suggested Prompt Chips */}
      {suggestions.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-executive-mutedLight dark:text-executive-mutedDark">
            <Lightbulb className="w-3.5 h-3.5 text-executive-orange dark:text-executive-gold" />
            <span>Suggested prompts for {state}:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestions.map((sug, idx) => (
              <button
                key={idx}
                onClick={() => handleSendMessage(sug)}
                disabled={loading}
                className="text-xs text-left px-3 py-1.5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark text-executive-textLight dark:text-executive-textDark hover:border-executive-navy dark:hover:border-executive-cyan hover:bg-slate-50 dark:hover:bg-slate-900 transition-all shadow-sm"
              >
                {sug}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chat Messages Container */}
      <div className="bg-white dark:bg-executive-cardDark rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex flex-col h-[520px]">
        
        {/* Messages Scroll Area */}
        <div className="flex-1 p-4 sm:p-6 overflow-y-auto space-y-4">
          {messages.map((msg) => {
            const isUser = msg.role === 'user';
            const isSourceOpen = expandedSources[msg.id];

            return (
              <div
                key={msg.id}
                className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
              >
                {!isUser && (
                  <div className="w-8 h-8 rounded-none bg-gradient-to-tr from-executive-navy to-blue-600 dark:from-executive-cyan dark:to-blue-400 text-white dark:text-slate-950 flex items-center justify-center flex-shrink-0 mt-0.5 shadow-sm">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div className={`max-w-[85%] sm:max-w-[75%] space-y-2`}>
                  <div
                    className={`p-4 rounded-none text-xs sm:text-sm leading-relaxed ${
                      isUser
                        ? 'bg-executive-navy dark:bg-executive-cyan text-white dark:text-slate-950'
                        : 'bg-slate-100 dark:bg-slate-900/90 border border-slate-200/80 dark:border-slate-800 text-executive-textLight dark:text-executive-textDark'
                    }`}
                  >
                    {msg.isOutOfScope && (
                      <div className="mb-2 p-2 rounded-none bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-200 text-xs flex items-center gap-1.5 font-medium">
                        <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                        Query out of domain scope
                      </div>
                    )}
                    <div className="whitespace-pre-wrap">{renderFormattedMessage(msg.content)}</div>
                  </div>

                  {/* Grounded Sources Accordion */}
                  {!isUser && msg.sources && msg.sources.length > 0 && (
                    <div className="bg-slate-50 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-800/80 rounded-none p-2.5 text-xs">
                      <button
                        onClick={() => toggleSourceExpand(msg.id)}
                        className="w-full flex items-center justify-between text-slate-500 dark:text-slate-400 font-semibold hover:text-executive-navy dark:hover:text-executive-cyan"
                      >
                        <span className="flex items-center gap-1.5">
                          <FileText className="w-3.5 h-3.5 text-blue-500" />
                          Grounded Sources ({msg.sources.length} citations)
                        </span>
                        {isSourceOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                      </button>

                      {isSourceOpen && (
                        <div className="mt-2 space-y-2 pt-2 border-t border-slate-200/60 dark:border-slate-800/60">
                          {msg.sources.map((src, sIdx) => (
                            <div key={sIdx} className="p-2 rounded-none bg-white dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800">
                              <div className="flex items-center justify-between text-[11px] font-bold text-executive-navy dark:text-executive-cyan mb-1">
                                <span>Citation #{sIdx + 1}</span>
                                <span className="text-[10px] px-1.5 py-0.5 rounded-none bg-slate-100 dark:bg-slate-800 text-slate-500">
                                  {src.type || 'Knowledge Document'}
                                </span>
                              </div>
                              <p className="text-[11px] text-slate-600 dark:text-slate-400 italic">
                                "{src.snippet}"
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {isUser && (
                  <div className="w-8 h-8 rounded-none bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })}

          {loading && (
            <div className="flex gap-3 items-center">
              <div className="w-8 h-8 rounded-none bg-gradient-to-tr from-executive-navy to-blue-600 dark:from-executive-cyan dark:to-blue-400 text-white dark:text-slate-950 flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 animate-spin" />
              </div>
              <div className="p-3.5 rounded-none bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-none bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="inline-block w-2 h-2 rounded-none bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="inline-block w-2 h-2 rounded-none bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                <span>Querying RAG knowledge base & ML models...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-executive-borderLight dark:border-executive-borderDark bg-slate-50/50 dark:bg-slate-900/30">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder={`Ask a question about ${state}'s demand forecast or cooling load...`}
              disabled={loading}
              className="flex-1 px-4 py-3 bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-none text-xs sm:text-sm text-executive-textLight dark:text-executive-textDark placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-executive-navy dark:focus:ring-executive-cyan"
            />
            <button
              type="submit"
              disabled={loading || !inputMessage.trim()}
              className="px-5 py-3 rounded-none bg-executive-navy dark:bg-executive-cyan text-white dark:text-slate-950 font-semibold text-xs sm:text-sm flex items-center gap-2 disabled:opacity-50 hover:opacity-90 transition-opacity"
            >
              <Send className="w-4 h-4" />
              <span className="hidden sm:inline">Send</span>
            </button>
          </form>
        </div>

      </div>

    </div>
  );
}

function renderFormattedMessage(text) {
  if (!text) return null;
  const tokens = [];
  const regex = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      tokens.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      tokens.push(
        <strong key={match.index} className="font-bold text-executive-navy dark:text-executive-cyan">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('*') && token.endsWith('*')) {
      tokens.push(
        <em key={match.index} className="italic text-slate-500 dark:text-slate-400">
          {token.slice(1, -1)}
        </em>
      );
    } else if (token.startsWith('`') && token.endsWith('`')) {
      tokens.push(
        <code key={match.index} className="px-1 py-0.5 bg-slate-200 dark:bg-slate-800 text-xs font-mono">
          {token.slice(1, -1)}
        </code>
      );
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    tokens.push(text.slice(lastIndex));
  }

  return tokens;
}

