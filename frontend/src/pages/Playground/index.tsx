import {
  CopyOutlined,
  DeleteOutlined,
  DislikeOutlined,
  LikeOutlined,
  MenuOutlined,
  PlusOutlined,
  ReloadOutlined,
  RobotOutlined,
  SendOutlined,
  UserOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Avatar,
  Button,
  Collapse,
  Drawer,
  Empty,
  Grid,
  Input,
  List,
  message,
  Popconfirm,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
} from "antd";
import React, { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useAccess } from "../../access/context";
import { MarkdownContent } from "../../components/MarkdownContent";
import { StatusTag } from "../../components/PageHeader";
import type { ResourceRecord } from "../../models/types";
import { api, streamFetch } from "../../services/http";

interface ChatSession {
  id: string;
  name: string;
  spec?: { agent_id?: string; message_count?: number };
}
interface ChatMessage {
  id: string;
  status?: string;
  error_message?: string;
  spec?: {
    role?: string;
    content?: string;
    feedback?: { rating?: string };
    token_usage?: { total_tokens?: number };
    token_usage_source?: string;
    latency_ms?: number;
    rag_sources?: Array<{ title?: string; content?: string }>;
    tool_calls?: Array<{ name?: string; result?: unknown }>;
  };
}
export function ChatPage(): JSX.Element {
  const { user, can } = useAccess();
  const writable = can("chat:write");
  const screens = Grid.useBreakpoint();
  const mobile = screens.md === false;
  const [params] = useSearchParams();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionPage, setSessionPage] = useState(1);
  const [sessionTotal, setSessionTotal] = useState(0);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);
  const activeRef = useRef<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [messagePage, setMessagePage] = useState(1);
  const [messageTotal, setMessageTotal] = useState(0);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const streamRef = useRef(false);
  const [streamContent, setStreamContent] = useState("");
  const [agents, setAgents] = useState<ResourceRecord[]>([]);
  const [selectedAgent, setSelectedAgent] = useState("");
  const [error, setError] = useState("");
  const [sessionError, setSessionError] = useState("");
  const [agentError, setAgentError] = useState("");
  const [sessionDrawer, setSessionDrawer] = useState(false);
  const messageArea = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const messageRequest = useRef(0);
  const cacheKey = `aipulse:last-session:${user?.id}`;
  const loadMessages = async (sessionId: string, number = 1) => {
    const request = ++messageRequest.current;
    setLoading(true);
    try {
      const result = await api.list<ChatMessage>(
        `/chat/sessions/${sessionId}/messages`,
        number,
        100,
      );
      if (
        alive.current &&
        activeRef.current === sessionId &&
        request === messageRequest.current
      ) {
        setMessages((previous) =>
          number === 1 ? result.items : [...result.items, ...previous],
        );
        setMessageTotal(result.total);
        setMessagePage(number);
      }
    } catch (reason) {
      if (activeRef.current === sessionId) setError((reason as Error).message);
    } finally {
      if (alive.current && request === messageRequest.current)
        setLoading(false);
    }
  };
  const selectSession = (session: ChatSession) => {
    if (streamRef.current) return;
    setActiveSession(session);
    if (session.spec?.agent_id) setSelectedAgent(session.spec.agent_id);
    activeRef.current = session.id;
    localStorage.setItem(cacheKey, session.id);
    setMessages([]);
    setError("");
    setStreamContent("");
    setSessionDrawer(false);
    void loadMessages(session.id);
  };
  const loadSessions = async (number = 1, restore = false) => {
    try {
      const result = await api.list<ChatSession>("/chat/sessions", number, 50);
      if (!alive.current) return;
      setSessionError("");
      setSessions((previous) =>
        number === 1 ? result.items : [...previous, ...result.items],
      );
      setSessionTotal(result.total);
      setSessionPage(number);
      if (restore) {
        const saved = result.items.find(
          (session) => session.id === localStorage.getItem(cacheKey),
        );
        if (saved) selectSession(saved);
      }
    } catch (reason) {
      if (alive.current) setSessionError((reason as Error).message);
    }
  };
  const loadAgents = async () => {
    if (!can("agents:read")) return;
    try {
      const result = await api.list("/agents", 1, 200);
      if (alive.current) {
        setAgents(result.items.filter((agent) => agent.enabled));
        setAgentError("");
        if (params.get("agent")) setSelectedAgent(params.get("agent")!);
      }
    } catch (reason) {
      if (alive.current) setAgentError((reason as Error).message);
    }
  };
  useEffect(() => {
    alive.current = true;
    void loadSessions(1, true);
    void loadAgents();
    return () => {
      alive.current = false;
      abortRef.current?.abort();
    };
  }, []);
  useEffect(() => {
    const area = messageArea.current;
    if (area && messagePage === 1)
      area.scrollTo({
        top: area.scrollHeight,
        behavior: streaming ? "auto" : "smooth",
      });
  }, [messages, streamContent, streaming]);
  const createSession = async () => {
    if (!selectedAgent || creating || streamRef.current) return;
    setCreating(true);
    try {
      const session = await api.post<ChatSession>("/chat/sessions", {
        agent_id: selectedAgent,
        title: "新对话",
      });
      selectSession(session);
      await loadSessions();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setCreating(false);
    }
  };
  const deleteSession = async (session: ChatSession) => {
    try {
      await api.remove("/chat/sessions", session.id);
      if (activeRef.current === session.id) {
        activeRef.current = "";
        setActiveSession(null);
        setMessages([]);
        localStorage.removeItem(cacheKey);
      }
      await loadSessions();
    } catch (reason) {
      message.error((reason as Error).message);
    }
  };
  const executeStream = async (messageId?: string) => {
    if (
      !activeSession ||
      streamRef.current ||
      (!messageId && !inputValue.trim())
    )
      return;
    const sessionId = activeSession.id;
    const content = inputValue.trim();
    streamRef.current = true;
    setStreaming(true);
    setError("");
    setStreamContent("");
    if (!messageId) {
      setInputValue("");
      setMessages((previous) => [
        ...previous,
        { id: `temporary-${Date.now()}`, spec: { role: "user", content } },
      ]);
    }
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await streamFetch(
        `/chat/sessions/${sessionId}/${messageId ? `messages/${messageId}/regenerate` : "stream"}`,
        messageId ? {} : { content },
        controller.signal,
      );
      if (!response.ok) {
        let detail = "";
        try {
          const data = await response.json();
          detail = data.message || data.detail;
        } catch {
          /* response has no JSON body */
        }
        throw new Error(detail || `请求失败（${response.status}）`);
      }
      if (!response.body) throw new Error("服务未返回消息内容");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullContent = "";
      const processLine = (line: string) => {
        if (!line.startsWith("data:")) return;
        const raw = line.slice(5).trim();
        if (!raw || raw === "[DONE]") return;
        let event: Record<string, any>;
        try {
          event = JSON.parse(raw);
        } catch {
          return;
        }
        if (!alive.current) return;
        if (event.type === "content_delta") {
          fullContent += event.delta || "";
          setStreamContent(fullContent);
        }
        if (event.type === "error")
          setError(event.error || "回复失败，请刷新查看记录");
        if (event.type === "approval_required")
          setError("工具调用等待审批。当前对话暂不支持审批后的自动续跑。");
      };
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";
        lines.forEach(processLine);
        if (done) {
          processLine(buffer);
          break;
        }
      }
    } catch (reason) {
      if ((reason as Error).name !== "AbortError" && alive.current) {
        setError((reason as Error).message);
        if (!messageId) setInputValue(content);
      }
    } finally {
      streamRef.current = false;
      abortRef.current = null;
      if (alive.current) {
        setStreaming(false);
        setStreamContent("");
        await Promise.all([loadMessages(sessionId), loadSessions()]);
      }
    }
  };
  const feedback = async (id: string, rating: string) => {
    try {
      await api.post(`/chat/messages/${id}/feedback`, { rating });
      if (activeSession) await loadMessages(activeSession.id);
    } catch (reason) {
      message.error((reason as Error).message);
    }
  };
  const copy = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      message.success("已复制");
    } catch {
      message.error("复制失败，请手动选择文本");
    }
  };
  const sidebar = (
    <div className="chat-session-sidebar">
      <div className="chat-sidebar-heading">
        <strong>我的对话</strong>
        <Button
          type="text"
          aria-label="刷新对话列表"
          icon={<ReloadOutlined />}
          disabled={streaming}
          onClick={() => void loadSessions()}
        />
      </div>
      <div className="chat-session-controls">
        <Select
          aria-label="选择 Agent"
          placeholder="选择智能体"
          showSearch
          optionFilterProp="label"
          value={selectedAgent || undefined}
          onChange={setSelectedAgent}
          options={agents.map((agent) => ({
            value: agent.id,
            label: agent.name,
          }))}
          disabled={streaming}
        />
        <Button
          type="primary"
          icon={<PlusOutlined />}
          block
          onClick={() => void createSession()}
          disabled={!selectedAgent || !writable || streaming}
          loading={creating}
        >
          新建对话
        </Button>
        {agentError && (
          <Alert
            type="error"
            message={agentError}
            action={
              <Button size="small" onClick={() => void loadAgents()}>
                重试
              </Button>
            }
          />
        )}
      </div>
      <div className="chat-session-list">
        {sessionError && (
          <Alert
            type="error"
            message={sessionError}
            action={
              <Button size="small" onClick={() => void loadSessions()}>
                重试
              </Button>
            }
          />
        )}
        <List
          dataSource={sessions}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="暂无对话记录"
              />
            ),
          }}
          renderItem={(session) => (
            <List.Item
              className={
                activeSession?.id === session.id ? "session-active" : ""
              }
              actions={
                writable
                  ? [
                      <Popconfirm
                        key="delete"
                        title="删除此对话及消息？"
                        onConfirm={() => deleteSession(session)}
                      >
                        <Button
                          type="text"
                          size="small"
                          danger
                          aria-label={`删除对话 ${session.name}`}
                          disabled={streaming}
                          icon={<DeleteOutlined />}
                        />
                      </Popconfirm>,
                    ]
                  : []
              }
            >
              <button
                className="session-select"
                disabled={streaming}
                onClick={() => selectSession(session)}
              >
                <strong>{session.name}</strong>
                <small>{session.spec?.message_count || 0} 条消息</small>
              </button>
            </List.Item>
          )}
        />
        {sessions.length < sessionTotal && (
          <Button
            block
            type="text"
            onClick={() => void loadSessions(sessionPage + 1)}
          >
            加载更多对话
          </Button>
        )}
      </div>
    </div>
  );
  const activeAgent = agents.find(
    (agent) => agent.id === activeSession?.spec?.agent_id,
  );
  return (
    <div className="chat-workspace">
      {!mobile && sidebar}
      <Drawer
        placement="left"
        width={300}
        title="对话记录"
        open={mobile && sessionDrawer}
        onClose={() => setSessionDrawer(false)}
        styles={{ body: { padding: 0 } }}
      >
        {sidebar}
      </Drawer>
      <section className="chat-main">
        <header className="chat-heading">
          <Space>
            {mobile && (
              <Button
                type="text"
                aria-label="打开对话记录"
                icon={<MenuOutlined />}
                onClick={() => setSessionDrawer(true)}
              />
            )}
            <strong>{activeSession?.name || "对话"}</strong>
            {activeAgent && <Tag>{activeAgent.name}</Tag>}
          </Space>
          {activeSession && (
            <Button
              type="text"
              aria-label="刷新消息"
              icon={<ReloadOutlined />}
              disabled={streaming}
              onClick={() => void loadMessages(activeSession.id)}
            />
          )}
        </header>
        {activeSession ? (
          <>
            {error && (
              <Alert
                type="warning"
                showIcon
                message={error}
                closable
                onClose={() => setError("")}
              />
            )}
            <div className="chat-messages" ref={messageArea}>
              {messages.length < messageTotal && (
                <Button
                  className="older-messages"
                  loading={loading}
                  onClick={() =>
                    void loadMessages(activeSession.id, messagePage + 1)
                  }
                >
                  加载更早消息
                </Button>
              )}
              {loading && !messages.length && (
                <div className="chat-loading">
                  <Spin />
                </div>
              )}
              {!loading && !messages.length && (
                <div className="conversation-start">
                  <RobotOutlined />
                  <h3>可以开始了</h3>
                  <p>
                    描述你的任务，{activeAgent?.name || "智能体"}
                    将根据配置回答。
                  </p>
                </div>
              )}
              {messages.map((msg) => {
                const isUser = msg.spec?.role === "user";
                return (
                  <article
                    key={msg.id}
                    className={`chat-message ${isUser ? "message-user" : ""}`}
                  >
                    <Avatar
                      icon={isUser ? <UserOutlined /> : <RobotOutlined />}
                    />
                    <div className="message-body">
                      <div className="message-author">
                        {isUser ? "你" : activeAgent?.name || "智能体"}
                      </div>
                      {msg.status &&
                        [
                          "failed",
                          "waiting_approval",
                          "cancelled",
                          "pending",
                        ].includes(msg.status) && (
                          <StatusTag status={msg.status} />
                        )}
                      {msg.error_message && (
                        <p className="message-error">{msg.error_message}</p>
                      )}
                      <div className="message-content">
                        {isUser ? (
                          msg.spec?.content
                        ) : (
                          <MarkdownContent>
                            {msg.spec?.content || ""}
                          </MarkdownContent>
                        )}
                      </div>
                      {!!msg.spec?.rag_sources?.length && (
                        <Collapse
                          size="small"
                          className="mt-3"
                          items={[
                            {
                              key: "sources",
                              label: `参考来源 · ${msg.spec.rag_sources.length}`,
                              children: msg.spec.rag_sources.map(
                                (source, index) => (
                                  <div key={index}>
                                    <strong>
                                      {source.title || `来源 ${index + 1}`}
                                    </strong>
                                    <p>{source.content}</p>
                                  </div>
                                ),
                              ),
                            },
                          ]}
                        />
                      )}
                      {!isUser && (
                        <div className="message-actions">
                          <Tooltip title="复制回答">
                            <Button
                              size="small"
                              type="text"
                              aria-label="复制回答"
                              icon={<CopyOutlined />}
                              onClick={() => void copy(msg.spec?.content || "")}
                            />
                          </Tooltip>
                          {writable && (
                            <>
                              <Tooltip title="有帮助">
                                <Button
                                  aria-label="有帮助"
                                  size="small"
                                  type="text"
                                  icon={<LikeOutlined />}
                                  className={
                                    msg.spec?.feedback?.rating === "good"
                                      ? "feedback-active"
                                      : ""
                                  }
                                  onClick={() => void feedback(msg.id, "good")}
                                />
                              </Tooltip>
                              <Tooltip title="没帮助">
                                <Button
                                  aria-label="没帮助"
                                  size="small"
                                  type="text"
                                  icon={<DislikeOutlined />}
                                  className={
                                    msg.spec?.feedback?.rating === "bad"
                                      ? "feedback-active"
                                      : ""
                                  }
                                  onClick={() => void feedback(msg.id, "bad")}
                                />
                              </Tooltip>
                              <Tooltip title="重新生成">
                                <Button
                                  aria-label="重新生成"
                                  size="small"
                                  type="text"
                                  disabled={streaming}
                                  icon={<ReloadOutlined />}
                                  onClick={() => void executeStream(msg.id)}
                                />
                              </Tooltip>
                            </>
                          )}
                          {!!msg.spec?.token_usage?.total_tokens && (
                            <span>
                              {msg.spec.token_usage_source === "estimated"
                                ? "约 "
                                : ""}
                              {msg.spec.token_usage.total_tokens} tokens
                            </span>
                          )}
                          {!!msg.spec?.latency_ms && (
                            <span>
                              {(msg.spec.latency_ms / 1000).toFixed(1)} s
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </article>
                );
              })}
              {streaming && (
                <article className="chat-message">
                  <Avatar icon={<RobotOutlined />} />
                  <div className="message-body">
                    <div className="message-author">
                      {activeAgent?.name || "智能体"}
                    </div>
                    {streamContent ? (
                      <div className="message-content">
                        <MarkdownContent>{streamContent}</MarkdownContent>
                      </div>
                    ) : (
                      <Space>
                        <Spin size="small" />
                        <span className="secondary-text">正在生成回答…</span>
                      </Space>
                    )}
                  </div>
                </article>
              )}
            </div>
            <div className="chat-composer">
              <div className="composer-input">
                <Input.TextArea
                  aria-label="输入消息"
                  placeholder="输入消息..."
                  value={inputValue}
                  onChange={(event) => setInputValue(event.target.value)}
                  onPressEnter={(event) => {
                    if (
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing &&
                      event.keyCode !== 229
                    ) {
                      event.preventDefault();
                      void executeStream();
                    }
                  }}
                  autoSize={{ minRows: 2, maxRows: 6 }}
                  disabled={streaming || !writable}
                />
                {streaming ? (
                  <Button danger onClick={() => abortRef.current?.abort()}>
                    停止接收
                  </Button>
                ) : (
                  <Button
                    type="primary"
                    icon={<SendOutlined />}
                    disabled={!inputValue.trim() || !writable}
                    onClick={() => void executeStream()}
                  >
                    发送
                  </Button>
                )}
              </div>
              <div className="composer-hint">
                {streaming
                  ? "停止接收后，已发出的外部调用可能继续。"
                  : writable
                    ? "Enter 发送 · Shift + Enter 换行"
                    : "当前账号仅可查看对话记录"}
              </div>
            </div>
          </>
        ) : (
          <div className="chat-welcome">
            <span className="welcome-icon">
              <RobotOutlined />
            </span>
            <h2>用 AIPulse 开始工作</h2>
            <p>
              {agents.length
                ? "选择一个智能体，新建对话并描述你的任务。"
                : "还没有可用的智能体，请先完成配置。"}
            </p>
            {agents.length ? (
              <Button
                type="primary"
                disabled={!selectedAgent || !writable}
                onClick={() => void createSession()}
              >
                开始新对话
              </Button>
            ) : can("agents:write") ? (
              <Link to="/build/agents/create">
                <Button type="primary">创建智能体</Button>
              </Link>
            ) : (
              <p>请联系管理员分配智能体及访问权限。</p>
            )}
            {mobile && (
              <Button className="mt-3" onClick={() => setSessionDrawer(true)}>
                选择智能体与历史对话
              </Button>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
