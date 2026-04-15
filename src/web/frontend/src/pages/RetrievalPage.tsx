import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';

import {
  createQAConversation,
  getQAConversation,
  listKGCandidates,
  listQAConversations,
  sendQAConversationMessage,
} from '../api';
import type { KGCandidate, QAConversationSummary, QAMessage } from '../types';

function resolveDefaultGraphPath(items: KGCandidate[]): string {
  if (!items.length) return '';
  const recentTask = items.find((item) => item.source === 'task');
  return recentTask?.path ?? items[0].path;
}

export function RetrievalPage() {
  const [kgCandidates, setKgCandidates] = useState<KGCandidate[]>([]);
  const [selectedKgPath, setSelectedKgPath] = useState('');
  const [question, setQuestion] = useState('');
  const [maxHop, setMaxHop] = useState(2);
  const [seedTopK, setSeedTopK] = useState(5);
  const [maxContextItems, setMaxContextItems] = useState(30);

  const [conversations, setConversations] = useState<QAConversationSummary[]>([]);
  const [activeConversation, setActiveConversation] = useState<QAConversationSummary | null>(null);
  const [messages, setMessages] = useState<QAMessage[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(() => (typeof window === 'undefined' ? true : window.innerWidth >= 1280));

  const [initializing, setInitializing] = useState(true);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const messageContainerRef = useRef<HTMLDivElement | null>(null);

  const activeConversationId = activeConversation?.id ?? '';
  const activeConversationTitle = activeConversation?.title?.trim() || '新对话';
  const activeConversationPath = activeConversation?.kg_path ?? '';
  const activeConversationLastUpdated = activeConversation?.updated_at ?? '';

  const loadConversationDetail = async (conversationId: string) => {
    setLoadingConversation(true);
    setError('');
    try {
      const detail = await getQAConversation(conversationId);
      setActiveConversation(detail.conversation);
      setMessages(detail.messages);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingConversation(false);
    }
  };

  useEffect(() => {
    const bootstrap = async () => {
      setInitializing(true);
      setError('');
      try {
        const [kgResponse, conversationResponse] = await Promise.all([listKGCandidates(), listQAConversations()]);
        setKgCandidates(kgResponse.items);
        setSelectedKgPath((current) => {
          if (current && kgResponse.items.some((item) => item.path === current)) return current;
          return resolveDefaultGraphPath(kgResponse.items);
        });
        setConversations(conversationResponse.items);

        if (conversationResponse.items.length > 0) {
          await loadConversationDetail(conversationResponse.items[0].id);
        } else {
          setActiveConversation(null);
          setMessages([]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setInitializing(false);
      }
    };

    void bootstrap();
  }, []);

  useEffect(() => {
    if (!messageContainerRef.current) return;
    messageContainerRef.current.scrollTop = messageContainerRef.current.scrollHeight;
  }, [messages, activeConversationId]);

  const onRefreshSideData = async () => {
    setError('');
    try {
      const [kgResponse, conversationResponse] = await Promise.all([listKGCandidates(), listQAConversations()]);
      setKgCandidates(kgResponse.items);
      setSelectedKgPath((current) => {
        if (current && kgResponse.items.some((item) => item.path === current)) return current;
        return resolveDefaultGraphPath(kgResponse.items);
      });
      setConversations(conversationResponse.items);
      if (activeConversationId && !conversationResponse.items.some((item) => item.id === activeConversationId)) {
        if (conversationResponse.items[0]) {
          await loadConversationDetail(conversationResponse.items[0].id);
        } else {
          setActiveConversation(null);
          setMessages([]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onCreateConversation = async () => {
    if (!selectedKgPath) {
      setError('请先选择一个可用图谱。');
      return;
    }
    setCreatingConversation(true);
    setError('');
    try {
      const created = await createQAConversation({
        kg_path: selectedKgPath,
      });
      setConversations((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      await loadConversationDetail(created.id);
      setQuestion('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreatingConversation(false);
    }
  };

  const onSwitchConversation = async (conversationId: string) => {
    if (!conversationId || conversationId === activeConversationId) return;
    await loadConversationDetail(conversationId);
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!activeConversationId) {
      setError('请先在右侧新建或选择一个对话。');
      return;
    }
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;

    setError('');
    setSending(true);

    try {
      const data = await sendQAConversationMessage(activeConversationId, {
        question: trimmedQuestion,
        max_hop: maxHop,
        seed_top_k: seedTopK,
        max_context_items: maxContextItems,
      });
      setMessages((current) => [...current, data.user_message, data.assistant_message]);
      setActiveConversation(data.conversation);
      setConversations((current) => [data.conversation, ...current.filter((item) => item.id !== data.conversation.id)]);
      setQuestion('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSending(false);
    }
  };

  const sortedConversations = useMemo(
    () =>
      [...conversations].sort(
        (a, b) => new Date(b.updated_at || b.created_at).getTime() - new Date(a.updated_at || a.created_at).getTime()
      ),
    [conversations]
  );

  return (
    <div className="space-y-3">
      <section className="panel p-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="m-0 font-display text-lg font-semibold">检索</h2>
            <p className="m-0 mt-1 text-sm text-app-muted">以会话方式持续问答检索，支持历史记录与参数调整</p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" className="soft-button" onClick={() => void onRefreshSideData()}>
              刷新数据
            </button>
            <button type="button" className="soft-button" onClick={() => setSidebarOpen((current) => !current)}>
              {sidebarOpen ? '收起侧栏' : '展开侧栏'}
            </button>
          </div>
        </div>
      </section>

      <div className={sidebarOpen ? 'grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_360px]' : 'grid grid-cols-1 gap-3'}>
        <section className="panel flex min-h-[620px] flex-col overflow-hidden">
          <header className="border-b border-app-border px-4 py-3">
            <p className="m-0 text-xs text-app-muted">当前对话</p>
            <h3 className="m-0 mt-1 text-base font-semibold">{activeConversationTitle}</h3>
            <p className="m-0 mt-1 truncate text-xs text-app-muted">
              {activeConversationPath ? `图谱：${activeConversationPath}` : '尚未选择会话'}
            </p>
            {activeConversationLastUpdated ? (
              <p className="m-0 mt-1 text-xs text-app-muted">{`更新时间：${formatDateTime(activeConversationLastUpdated)}`}</p>
            ) : null}
          </header>

          <div ref={messageContainerRef} className="custom-scrollbar flex-1 space-y-3 overflow-auto bg-slate-50/40 px-4 py-4">
            {initializing || loadingConversation ? (
              <p className="m-0 text-sm text-app-muted">加载中...</p>
            ) : null}
            {!initializing && !loadingConversation && !activeConversation ? (
              <div className="rounded-xl border border-dashed border-app-border bg-white p-4 text-sm text-app-muted">
                请在右侧新建一个对话，或选择已有对话继续问答。
              </div>
            ) : null}
            {!initializing && !loadingConversation && activeConversation && messages.length === 0 ? (
              <div className="rounded-xl border border-dashed border-app-border bg-white p-4 text-sm text-app-muted">
                当前对话还没有消息，输入问题开始检索。
              </div>
            ) : null}
            {!initializing && !loadingConversation
              ? messages.map((message) => {
                  const isUser = message.role === 'user';
                  const rawContext = message.qa_response_snapshot.retrieved_context;
                  const rawIntermediate = message.qa_response_snapshot.intermediate;
                  return (
                    <article key={message.id} className={isUser ? 'ml-auto max-w-[82%]' : 'mr-auto max-w-[88%]'}>
                      <div
                        className={
                          isUser
                            ? 'rounded-2xl rounded-tr-md bg-emerald-500 px-4 py-3 text-sm leading-6 text-white'
                            : 'rounded-2xl rounded-tl-md border border-app-border bg-white px-4 py-3 text-sm leading-6 text-slate-800'
                        }
                      >
                        {message.content}
                      </div>
                      <p className={isUser ? 'm-0 mt-1 text-right text-xs text-app-muted' : 'm-0 mt-1 text-xs text-app-muted'}>
                        {formatDateTime(message.created_at)}
                      </p>
                      {!isUser ? (
                        <details className="mt-1">
                          <summary className="cursor-pointer text-xs font-semibold text-app-muted">检索详情</summary>
                          <div className="mt-2 space-y-2">
                            <details>
                              <summary className="cursor-pointer text-xs text-app-muted">检索上下文</summary>
                              <pre className="json-panel mt-1 max-h-72">
                                {JSON.stringify(rawContext ?? {}, null, 2)}
                              </pre>
                            </details>
                            <details>
                              <summary className="cursor-pointer text-xs text-app-muted">中间信息</summary>
                              <pre className="json-panel mt-1 max-h-72">
                                {JSON.stringify(rawIntermediate ?? {}, null, 2)}
                              </pre>
                            </details>
                          </div>
                        </details>
                      ) : null}
                    </article>
                  );
                })
              : null}
          </div>

          <form onSubmit={onSubmit} className="border-t border-app-border bg-white p-4">
            <label className="block text-xs text-app-muted">
              问题
              <textarea
                className="field mt-1 min-h-24"
                placeholder={activeConversationId ? '在当前对话中继续提问' : '请先新建或选择一个对话'}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                disabled={!activeConversationId || sending}
              />
            </label>
            <div className="mt-3 flex items-center justify-end">
              <button type="submit" className="brand-button" disabled={!activeConversationId || sending || !question.trim()}>
                {sending ? '检索中...' : '发送问题'}
              </button>
            </div>
          </form>
        </section>

        {sidebarOpen ? (
          <aside className="panel custom-scrollbar max-h-[80vh] space-y-4 overflow-auto p-4">
            <section className="space-y-2">
              <h3 className="m-0 text-sm font-semibold">新建对话</h3>
              <label className="block text-xs text-app-muted">
                会话图谱（固定）
                <select className="field mt-1" value={selectedKgPath} onChange={(event) => setSelectedKgPath(event.target.value)}>
                  <option value="">请选择图谱</option>
                  {kgCandidates.map((candidate) => (
                    <option key={`${candidate.source}:${candidate.path}`} value={candidate.path}>
                      {candidate.display_name}
                    </option>
                  ))}
                </select>
              </label>
              <button type="button" className="brand-button w-full justify-center" onClick={() => void onCreateConversation()} disabled={creatingConversation || !selectedKgPath}>
                {creatingConversation ? '创建中...' : '新建对话'}
              </button>
            </section>

            <section className="space-y-2">
              <h3 className="m-0 text-sm font-semibold">对话列表</h3>
              {sortedConversations.length === 0 ? (
                <div className="rounded-xl border border-dashed border-app-border bg-slate-50 px-3 py-4 text-sm text-app-muted">暂无对话</div>
              ) : (
                <div className="space-y-2">
                  {sortedConversations.map((conversation) => {
                    const active = conversation.id === activeConversationId;
                    return (
                      <button
                        key={conversation.id}
                        type="button"
                        className={`w-full rounded-xl border px-3 py-2 text-left transition ${
                          active ? 'border-emerald-300 bg-emerald-50' : 'border-app-border bg-white hover:bg-slate-50'
                        }`}
                        onClick={() => void onSwitchConversation(conversation.id)}
                      >
                        <p className="m-0 truncate text-sm font-semibold text-app-text">{conversation.title || '新对话'}</p>
                        <p className="m-0 mt-1 truncate text-xs text-app-muted">{conversation.last_message_preview || '暂无消息'}</p>
                        <p className="m-0 mt-1 text-xs text-app-muted">
                          {`${conversation.message_count} 条消息 · ${formatDateTime(conversation.updated_at || conversation.created_at)}`}
                        </p>
                      </button>
                    );
                  })}
                </div>
              )}
            </section>

            <section className="space-y-2">
              <h3 className="m-0 text-sm font-semibold">检索参数</h3>
              <label className="block text-xs text-app-muted">
                当前会话图谱（只读）
                <input className="field mt-1" value={activeConversationPath} disabled />
              </label>
              <div className="grid grid-cols-1 gap-2">
                <label className="block text-xs text-app-muted">
                  max_hop
                  <input
                    className="field mt-1"
                    type="number"
                    min={1}
                    max={3}
                    value={maxHop}
                    onChange={(event) => setMaxHop(Number(event.target.value) || 2)}
                  />
                </label>
                <label className="block text-xs text-app-muted">
                  seed_top_k
                  <input
                    className="field mt-1"
                    type="number"
                    min={1}
                    max={20}
                    value={seedTopK}
                    onChange={(event) => setSeedTopK(Number(event.target.value) || 5)}
                  />
                </label>
                <label className="block text-xs text-app-muted">
                  max_context_items
                  <input
                    className="field mt-1"
                    type="number"
                    min={1}
                    max={100}
                    value={maxContextItems}
                    onChange={(event) => setMaxContextItems(Number(event.target.value) || 30)}
                  />
                </label>
              </div>
            </section>
          </aside>
        ) : null}
      </div>

      {error ? (
        <section className="panel p-3">
          <p className="m-0 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">{error}</p>
        </section>
      ) : null}
    </div>
  );
}

function formatDateTime(value: string): string {
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return value;
  return new Date(timestamp).toLocaleString('zh-CN', { hour12: false });
}
