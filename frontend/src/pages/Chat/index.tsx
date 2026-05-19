import { SendOutlined, PlusOutlined, DeleteOutlined, LikeOutlined, DislikeOutlined, ReloadOutlined, PushpinOutlined, RobotOutlined, UserOutlined } from '@ant-design/icons';
import { Button, Input, List, Space, Tag, Tooltip, message, Spin, Avatar, Empty, Popconfirm, Select } from 'antd';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { api, getToken } from '../../services/http';

interface ChatSession {
  id: string;
  name: string;
  spec?: {
    agent_id?: string;
    message_count?: number;
    last_message_at?: string;
    pinned?: boolean;
  };
  updated_at?: string;
}

interface ChatMessage {
  id: string;
  spec?: {
    role?: string;
    content?: string;
    feedback?: { rating?: string } | null;
    token_usage?: { total_tokens?: number };
    latency_ms?: number;
    rag_sources?: Array<{ title?: string; content?: string }>;
    tool_calls?: Array<{ name?: string; result?: string }>;
  };
  created_at?: string;
}

interface StreamEvent {
  type: string;
  message_id?: string;
  delta?: string;
  token_usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
  latency_ms?: number;
  sources?: Array<{ title?: string; content?: string }>;
}

export function ChatPage(): JSX.Element {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamContent, setStreamContent] = useState('');
  const [agents, setAgents] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedAgent, setSelectedAgent] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, streamContent, scrollToBottom]);

  const loadSessions = async (): Promise<void> => {
    try {
      const res = await api.post<{ items: ChatSession[] }>('/chat/sessions?page=1&page_size=50', {});
      setSessions((res as any).items || []);
    } catch {
      const res = await fetch('/api/v1/chat/sessions?page=1&page_size=50', {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      if (res.ok) {
        const data = await res.json();
        setSessions(data.items || []);
      }
    }
  };

  const loadAgents = async (): Promise<void> => {
    try {
      const res = await api.list('/agents', 1, 100);
      setAgents((res.items || []).map((a) => ({ id: a.id, name: a.name })));
    } catch { /* ignore */ }
  };

  useEffect(() => { void loadSessions(); void loadAgents(); }, []);

  const loadMessages = async (sessionId: string): Promise<void> => {
    try {
      const res = await fetch(`/api/v1/chat/sessions/${sessionId}/messages?page=1&page_size=100`, {
        headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' },
      });
      if (res.ok) {
        const data = await res.json();
        setMessages(data.items || []);
      }
    } catch { /* ignore */ }
  };

  const selectSession = (session: ChatSession): void => {
    setActiveSession(session);
    void loadMessages(session.id);
    setStreamContent('');
  };

  const createSession = async (): Promise<void> => {
    try {
      const res = await fetch('/api/v1/chat/sessions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: selectedAgent, title: '新对话' }),
      });
      if (res.ok) {
        const session = await res.json();
        await loadSessions();
        selectSession(session);
      }
    } catch { message.error('创建会话失败'); }
  };

  const deleteSession = async (sessionId: string): Promise<void> => {
    try {
      await fetch(`/api/v1/chat/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      if (activeSession?.id === sessionId) {
        setActiveSession(null);
        setMessages([]);
      }
      await loadSessions();
    } catch { message.error('删除失败'); }
  };

  const sendMessage = async (): Promise<void> => {
    if (!inputValue.trim() || !activeSession || streaming) return;
    const content = inputValue.trim();
    setInputValue('');
    setStreaming(true);
    setStreamContent('');

    const userMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      spec: { role: 'user', content },
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    try {
      abortRef.current = new AbortController();
      const res = await fetch(`/api/v1/chat/sessions/${activeSession.id}/stream`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) throw new Error('Stream failed');
      const reader = res.body?.getReader();
      if (!reader) throw new Error('No reader');

      const decoder = new TextDecoder();
      let fullContent = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event: StreamEvent = JSON.parse(line.slice(6));
              if (event.type === 'content_delta' && event.delta) {
                fullContent += event.delta;
                setStreamContent(fullContent);
              } else if (event.type === 'message_end') {
                const assistantMsg: ChatMessage = {
                  id: event.message_id || `msg-${Date.now()}`,
                  spec: {
                    role: 'assistant',
                    content: fullContent,
                    token_usage: event.token_usage,
                    latency_ms: event.latency_ms,
                  },
                  created_at: new Date().toISOString(),
                };
                setMessages((prev) => [...prev, assistantMsg]);
                setStreamContent('');
              }
            } catch { /* skip malformed */ }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        message.error('发送失败');
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const stopStreaming = (): void => {
    abortRef.current?.abort();
    setStreaming(false);
  };

  const feedbackMessage = async (messageId: string, rating: string): Promise<void> => {
    try {
      await fetch(`/api/v1/chat/messages/${messageId}/feedback`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating }),
      });
      message.success('反馈已提交');
      if (activeSession) await loadMessages(activeSession.id);
    } catch { message.error('反馈失败'); }
  };

  const regenerateMessage = async (messageId: string): Promise<void> => {
    if (!activeSession || streaming) return;
    setStreaming(true);
    setStreamContent('');
    try {
      abortRef.current = new AbortController();
      const res = await fetch(`/api/v1/chat/sessions/${activeSession.id}/messages/${messageId}/regenerate`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${getToken()}`, 'Content-Type': 'application/json' },
        signal: abortRef.current.signal,
      });
      if (!res.ok) throw new Error('Regenerate failed');
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let fullContent = '';
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event: StreamEvent = JSON.parse(line.slice(6));
              if (event.type === 'content_delta' && event.delta) {
                fullContent += event.delta;
                setStreamContent(fullContent);
              } else if (event.type === 'message_end') {
                setStreamContent('');
                if (activeSession) await loadMessages(activeSession.id);
              }
            } catch { /* skip */ }
          }
        }
      }
    } catch { message.error('重新生成失败'); }
    finally { setStreaming(false); abortRef.current = null; }
  };

  const renderMessage = (msg: ChatMessage): JSX.Element => {
    const role = msg.spec?.role || 'user';
    const content = msg.spec?.content || '';
    const isUser = role === 'user';
    const feedback = msg.spec?.feedback;
    const ragSources = msg.spec?.rag_sources || [];

    return (
      <div key={msg.id} className={`flex mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {!isUser && <Avatar icon={<RobotOutlined />} className="mr-2 bg-blue-500" />}
        <div className={`max-w-[70%] ${isUser ? 'bg-blue-50' : 'bg-gray-50'} rounded-lg p-3`}>
          <div className="whitespace-pre-wrap text-sm">{content}</div>
          {ragSources.length > 0 && (
            <div className="mt-2 border-t pt-2">
              <div className="text-xs text-gray-400 mb-1">参考来源:</div>
              {ragSources.map((s, i) => (
                <Tag key={i} className="text-xs mb-1">{s.title || `来源${i + 1}`}</Tag>
              ))}
            </div>
          )}
          {!isUser && (
            <div className="mt-2 flex items-center gap-2">
              {msg.spec?.token_usage?.total_tokens && (
                <span className="text-xs text-gray-400">{msg.spec.token_usage.total_tokens} tokens</span>
              )}
              {msg.spec?.latency_ms && (
                <span className="text-xs text-gray-400">{msg.spec.latency_ms}ms</span>
              )}
              <Tooltip title="有帮助">
                <Button
                  size="small" type="text" icon={<LikeOutlined />}
                  className={feedback?.rating === 'good' ? 'text-green-500' : ''}
                  onClick={() => void feedbackMessage(msg.id, 'good')}
                />
              </Tooltip>
              <Tooltip title="没帮助">
                <Button
                  size="small" type="text" icon={<DislikeOutlined />}
                  className={feedback?.rating === 'bad' ? 'text-red-500' : ''}
                  onClick={() => void feedbackMessage(msg.id, 'bad')}
                />
              </Tooltip>
              <Tooltip title="重新生成">
                <Button size="small" type="text" icon={<ReloadOutlined />} onClick={() => void regenerateMessage(msg.id)} />
              </Tooltip>
            </div>
          )}
        </div>
        {isUser && <Avatar icon={<UserOutlined />} className="ml-2 bg-green-500" />}
      </div>
    );
  };

  return (
    <div className="flex h-[calc(100vh-64px)]">
      {/* Session List */}
      <div className="w-72 border-r flex flex-col bg-white">
        <div className="p-3 border-b">
          <Space className="w-full" direction="vertical" size="small">
            <Select
              className="w-full" placeholder="选择 Agent" allowClear
              value={selectedAgent || undefined}
              onChange={(v) => setSelectedAgent(v || '')}
              options={agents.map((a) => ({ value: a.id, label: a.name }))}
            />
            <Button type="primary" icon={<PlusOutlined />} block onClick={() => void createSession()}>
              新建对话
            </Button>
          </Space>
        </div>
        <div className="flex-1 overflow-y-auto">
          <List
            dataSource={sessions}
            renderItem={(session) => (
              <List.Item
                className={`cursor-pointer px-3 hover:bg-gray-50 ${activeSession?.id === session.id ? 'bg-blue-50' : ''}`}
                onClick={() => selectSession(session)}
                actions={[
                  <Popconfirm key="del" title="确认删除?" onConfirm={() => void deleteSession(session.id)}>
                    <Button size="small" type="text" danger icon={<DeleteOutlined />} />
                  </Popconfirm>,
                ]}
              >
                <List.Item.Meta
                  title={
                    <span className="text-sm">
                      {session.spec?.pinned && <PushpinOutlined className="mr-1 text-orange-400" />}
                      {session.name}
                    </span>
                  }
                  description={<span className="text-xs text-gray-400">{session.spec?.message_count || 0} 条消息</span>}
                />
              </List.Item>
            )}
          />
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {activeSession ? (
          <>
            <div className="p-3 border-b bg-white flex items-center justify-between">
              <span className="font-medium">{activeSession.name}</span>
              <Space>
                {activeSession.spec?.agent_id && <Tag color="blue">Agent: {activeSession.spec.agent_id.slice(0, 8)}</Tag>}
              </Space>
            </div>
            <div className="flex-1 overflow-y-auto p-4 bg-gray-100">
              {messages.map(renderMessage)}
              {streaming && streamContent && (
                <div className="flex mb-4 justify-start">
                  <Avatar icon={<RobotOutlined />} className="mr-2 bg-blue-500" />
                  <div className="max-w-[70%] bg-gray-50 rounded-lg p-3">
                    <div className="whitespace-pre-wrap text-sm">{streamContent}</div>
                    <Spin size="small" className="mt-1" />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <div className="p-3 border-t bg-white">
              <Space.Compact className="w-full">
                <Input.TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); void sendMessage(); } }}
                  placeholder="输入消息，Shift+Enter 换行..."
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  disabled={streaming}
                  className="flex-1"
                />
                {streaming ? (
                  <Button danger onClick={stopStreaming}>停止</Button>
                ) : (
                  <Button type="primary" icon={<SendOutlined />} onClick={() => void sendMessage()} disabled={!inputValue.trim()}>
                    发送
                  </Button>
                )}
              </Space.Compact>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <Empty description="选择或创建一个对话开始聊天" />
          </div>
        )}
      </div>
    </div>
  );
}
