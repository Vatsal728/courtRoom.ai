import {
  AlertCircle,
  Bell,
  CheckCircle2,
  ChevronDown,
  ClipboardList,
  Copy,
  Download,
  FileText,
  Folder,
  Loader,
  LogOut,
  MessageSquare,
  Mic,
  Moon,
  MoreHorizontal,
  Paperclip,
  Pencil,
  Plus,
  Scale,
  Send,
  Settings,
  ShieldCheck,
  Square,
  Star,
  Sun,
  Trash2,
  X,
  Zap
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { LanguageDropdown } from './components/LanguageDropdown';
import './index.css';
import {
  downloadPDF,
  generatePDFNotice,
  generateRTIApplication,
  getEvidenceChecklist,
  getStrategy,
  getUserDocuments,
  getUserPDFs,
  auditDocument,
  healthCheck,
  streamQuery,
  uploadDocument
} from './lib/api';
import { AuthContext, AuthProvider, useAuth } from './lib/auth';
import { getLanguageInfo, localizeDomain, uiLabel } from './lib/language';
import type { CaseStrategyResult, DocumentAuditResult, Language, ToolCard } from './lib/types';

const shortTitle = (t: string) => (t.length > 42 ? `${t.slice(0, 39)}…` : t);

// ============ DRAFT PREVIEW HELPERS ============
// The model streams raw JSON (`format: json`). Until the final formatted answer
// swaps in, render only the completed string fields as a readable live draft.

const JSON_DRAFT_RE = /"([A-Za-z_]+)"\s*:\s*"((?:[^"\\]|\\.)*)"/g;

function jsonDraftToText(raw: string, lang: Language = 'en'): string {
  const text = raw.trim();
  if (!text) return '';
  const lines: string[] = [];
  const shown = new Set<string>();
  let m: RegExpExecArray | null;
  while ((m = JSON_DRAFT_RE.exec(text)) !== null) {
    const key = m[1];
    if (key !== 'short_answer' && key !== 'is_this_illegal') continue;
    let val = m[2]
      .replace(/\\n/g, '\n')
      .replace(/\\"/g, '"')
      .replace(/\\u([0-9a-fA-F]{4})/g, (_, u) => String.fromCharCode(parseInt(u, 16)));
    if (!val.trim() || shown.has(val)) continue;
    shown.add(val);
    lines.push(key === 'is_this_illegal' ? `${uiLabel(lang, 'is_this_illegal')}\n${val}` : val);
  }
  return lines.join('\n\n');
}

// Strip heavy fields (full statute text) before persisting sessions, so
// localStorage never balloons toward the ~5MB quota.
function stripContent(msgs: Message[]): Message[] {
  return msgs.map((m) => ({
    ...m,
    results: Array.isArray(m.results)
      ? m.results.map((r) => (r && typeof r === 'object' ? { ...r, content: undefined } : r))
      : m.results,
  }));
}

// ============ LOGIN PAGE ============

function LoginPage() {
  const { login, register } = useAuth();
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isRegister) {
        await register(email, password, name);
      } else {
        await login(email, password);
      }
      window.location.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-gradient-to-br from-bg via-card to-bg flex items-center justify-center p-4">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl"></div>
      </div>

      <div className="w-full max-w-sm relative z-10">
        {/* Logo */}
        <div className={`text-center ${isRegister ? 'mb-3' : 'mb-6'}`}>
          <div className="flex items-center justify-center gap-3 mb-1">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center shadow-lg flex-shrink-0">
              <Scale className="w-5.5 h-5.5 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-ink leading-none">
              court<span className="text-accent">Room</span>.ai
            </h1>
          </div>
          <p className="text-[10px] text-ink-3 font-semibold uppercase tracking-wider">AI Legal Assistant for India</p>
        </div>

        {/* Form Card */}
        <div className={`bg-card/90 backdrop-blur-xl border border-line rounded-2xl shadow-pop transition-all duration-300 ${isRegister ? 'p-5 space-y-3.5' : 'p-7 space-y-5'
          }`}>
          <div>
            <h2 className="text-xl font-bold text-ink mb-1">
              {isRegister ? 'Create Account' : 'Welcome Back'}
            </h2>
            <p className="text-xs text-ink-2">
              {isRegister ? 'Join thousands using AI legal guidance' : 'Sign in to your account'}
            </p>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-err/10 border border-err/30 flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-err flex-shrink-0 mt-0.5" />
              <p className="text-err text-xs">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className={isRegister ? 'space-y-3' : 'space-y-4'}>
            {isRegister && (
              <div>
                <label className="block text-xs font-semibold text-ink-2 mb-1.5">Full Name</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  className="ui-input text-sm"
                  required
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-semibold text-ink-2 mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="ui-input text-sm"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-ink-2 mb-1.5">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="ui-input text-sm"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="ui-btn-primary"
            >
              {loading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Loading...
                </>
              ) : (
                <>{isRegister ? 'Create Account' : 'Sign In'}</>
              )}
            </button>

            <button
              type="button"
              onClick={() => setIsRegister(!isRegister)}
              className="w-full text-xs text-ink-3 hover:text-ink transition font-medium cursor-pointer"
            >
              {isRegister ? 'Already have an account? Sign in' : "Don't have an account? Register"}
            </button>
          </form>
        </div>

        <p className={`text-center text-[10px] text-ink-3 ${isRegister ? 'mt-2' : 'mt-5'}`}>
          AI-powered legal guidance for Indian citizens
        </p>
      </div>
    </div>
  );
}

// ============ CHAT MESSAGE ============

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  results?: any[];
  domain?: string;
  confidence?: number;
  lang?: Language;
  tools?: ToolCard[];
  timestamp: Date;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  timestamp: Date;
}

// ============ TOOL ARTIFACT CARDS (Phase 4) ============

const inr = (n: number) => `Rs ${Math.round(n).toLocaleString('en-IN')}`;

function ToolArtifactCard({
  tool,
  onOpenStrategy,
  onOpenAudit,
  onContinue,
}: {
  tool: ToolCard;
  onOpenStrategy: (s: CaseStrategyResult) => void;
  onOpenAudit: (a: DocumentAuditResult) => void;
  onContinue: () => void;
}) {
  const isReady = tool.status === 'ready';
  const isNeedsInput = tool.status === 'needs_input';
  const iconColor = isReady ? 'text-ok' : isNeedsInput ? 'text-warn' : 'text-err';

  return (
    <div className="mt-4 rounded-2xl bg-bg border border-line p-4">
      <div className="flex items-start gap-3">
        <div className={`w-8 h-8 rounded-lg bg-card border border-line flex items-center justify-center flex-shrink-0 ${iconColor}`}>
          {tool.type === 'case_strategy' ? <ClipboardList className="w-4 h-4" /> : tool.type === 'document_audit' ? <ShieldCheck className="w-4 h-4" /> : <FileText className="w-4 h-4" />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="text-sm font-bold text-ink">{tool.title}</p>
            <span className={`text-[9px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full border ${
              isReady
                ? 'bg-ok/10 text-ok border-ok/25'
                : isNeedsInput
                  ? 'bg-warn/10 text-warn border-warn/25'
                  : 'bg-err/10 text-err border-err/25'
            }`}>
              {tool.status === 'needs_input' ? 'needs details' : tool.status}
            </span>
          </div>
          {tool.message && <p className="text-xs text-ink-2 mt-1 leading-relaxed">{tool.message}</p>}

          <div className="flex flex-wrap items-center gap-2 mt-3">
            {isReady && tool.type === 'legal_notice' && tool.pdf_id && (
              <button
                onClick={() => downloadPDF(tool.pdf_id!, tool.filename || 'legal_notice.pdf').catch(() => { })}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-btn hover:bg-btn-hover text-white text-xs font-semibold transition cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" /> Download PDF
              </button>
            )}
            {isReady && tool.type === 'rti_application' && tool.application && (
              <>
                <button
                  onClick={() => navigator.clipboard?.writeText(tool.application || '').then(() => onContinue()).catch(() => { })}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-btn hover:bg-btn-hover text-white text-xs font-semibold transition cursor-pointer"
                >
                  <Copy className="w-3.5 h-3.5" /> Copy draft
                </button>
                <details className="flex-1 min-w-0">
                  <summary className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-card border border-line text-ink-2 hover:text-ink text-xs font-semibold transition cursor-pointer select-none">
                    View draft
                  </summary>
                  <pre className="mt-2 p-3 rounded-xl bg-card border border-line text-ink text-[11px] font-mono leading-relaxed whitespace-pre-wrap max-h-60 overflow-y-auto">
                    {tool.application}
                  </pre>
                </details>
              </>
            )}
            {isReady && tool.type === 'case_strategy' && tool.strategy && (
              <button
                onClick={() => onOpenStrategy(tool.strategy!)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-btn hover:bg-btn-hover text-white text-xs font-semibold transition cursor-pointer"
              >
                <ClipboardList className="w-3.5 h-3.5" /> Open Case Strategy
              </button>
            )}
            {isReady && tool.type === 'document_audit' && tool.audit && (
              <button
                onClick={() => onOpenAudit(tool.audit!)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-btn hover:bg-btn-hover text-white text-xs font-semibold transition cursor-pointer"
              >
                <ShieldCheck className="w-3.5 h-3.5" /> Open Document Audit
              </button>
            )}
            {isReady && tool.type === 'case_strategy' && tool.strategy?.compensation_estimate && (
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-card border border-line text-xs font-bold text-accent">
                {inr(tool.strategy.compensation_estimate.min_amount)} – {inr(tool.strategy.compensation_estimate.max_amount)}
              </span>
            )}
            {isReady && tool.type === 'document_audit' && tool.audit && (
              <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold border ${
                tool.audit.risk === 'LOW' ? 'bg-ok/10 text-ok border-ok/25' : tool.audit.risk === 'HIGH' ? 'bg-err/10 text-err border-err/25' : 'bg-warn/10 text-warn border-warn/25'
              }`}>
                {tool.audit.score}% · {tool.audit.risk} risk
              </span>
            )}
            {isNeedsInput && (
              <button
                onClick={onContinue}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-card border border-line text-ink-2 hover:text-ink text-xs font-semibold transition cursor-pointer"
              >
                Continue in chat
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ MAIN APP CONTENT ============

function AppContent() {
  const auth = React.useContext(AuthContext);
  if (!auth) return null;

  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('theme');
    const initial = saved === 'dark' || (saved !== 'light' && (window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false));
    document.documentElement.classList.toggle('dark', initial);
    return initial;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const toggleTheme = () => setDarkMode((d) => !d);

  const getGreeting = () => {
    const hr = new Date().getHours();
    const name = auth.userName ? auth.userName.split(' ')[0] : 'User';
    if (hr < 12) return `Morning, ${name}`;
    if (hr < 17) return `Afternoon, ${name}`;
    return `Evening, ${name}`;
  };

  const [activeView, setActiveView] = useState<'chat' | 'notice' | 'evidence' | 'rti' | 'strategy' | 'audit' | 'settings'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState<Language>('en');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [artifactsOpen, setArtifactsOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const [expandedSource, setExpandedSource] = useState<string | null>(null);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>(() => {
    const storageKey = `chat_sessions_v2_${auth.userId || 'anonymous'}`;
    try {
      const scoped = JSON.parse(localStorage.getItem(storageKey) || '[]');
      if (Array.isArray(scoped) && scoped.length > 0) return scoped;
      const v1 = JSON.parse(localStorage.getItem(`chat_sessions_${auth.userId || 'anonymous'}`) || '[]');
      if (Array.isArray(v1) && v1.length > 0) {
        localStorage.setItem(storageKey, JSON.stringify(v1));
        return v1;
      }
      return [];
    } catch {
      return [];
    } finally {
      try {
        const keys = Object.keys(localStorage);
        keys.forEach((key) => {
          if (key.startsWith('chat_sessions_') && !key.startsWith('chat_sessions_v2_')) {
            localStorage.removeItem(key);
          }
        });
      } catch {
        // ignore cleanup errors
      }
    }
  });
  const [showPDFModal, setShowPDFModal] = useState(false);
  const [showDocsModal, setShowDocsModal] = useState(false);
  const [userDocuments, setUserDocuments] = useState<any[]>([]);
  const [userPDFs, setUserPDFs] = useState<any[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('cached_user_pdfs') || '[]');
    } catch {
      return [];
    }
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const bottomInputRef = useRef<HTMLTextAreaElement>(null);
  const generationIdRef = useRef(0);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [chatMenuId, setChatMenuId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  // Phase B UI state
  const [workspaceMenuOpen, setWorkspaceMenuOpen] = useState(false);
  const [notifMenuOpen, setNotifMenuOpen] = useState(false);
  const [attachment, setAttachment] = useState<{ fileId: string; filename: string; size: number } | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState<Record<string, boolean>>({});

  const showToast = useCallback((text: string) => {
    setToast(text);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3000);
  }, []);

  // Bookmarks (working minimal): star a message -> listed in sidebar
  const [bookmarks, setBookmarks] = useState<{
    id: string;
    messageId: string;
    text: string;
    chatTitle: string;
    chatId: string | null;
    timestamp: number;
  }[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(`bookmarks_v1_${auth.userId || 'anonymous'}`) || '[]');
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(`bookmarks_v1_${auth.userId || 'anonymous'}`, JSON.stringify(bookmarks));
    } catch {
      // ignore quota errors
    }
  }, [bookmarks, auth.userId]);

  const toggleBookmark = useCallback((msg: Message, chatTitle: string) => {
    setBookmarks((prev) => {
      const exists = prev.find((b) => b.messageId === msg.id);
      if (exists) return prev.filter((b) => b.messageId !== msg.id);
      return [
        {
          id: `bm-${Date.now()}`,
          messageId: msg.id,
          text: msg.content.slice(0, 120),
          chatTitle,
          chatId: activeChatId,
          timestamp: Date.now(),
        },
        ...prev,
      ].slice(0, 100);
    });
  }, [activeChatId]);

  const groupSessions = useCallback(() => {
    const now = new Date();
    const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const startOfYesterday = startOfDay - 86400000;
    const startOfWeek = startOfDay - 6 * 86400000;
    const groups: { label: string; sessions: ChatSession[] }[] = [
      { label: 'Today', sessions: [] },
      { label: 'Yesterday', sessions: [] },
      { label: 'Previous 7 Days', sessions: [] },
      { label: 'Older', sessions: [] },
    ];
    for (const s of chatSessions) {
      const t = new Date(s.timestamp).getTime();
      if (t >= startOfDay) groups[0].sessions.push(s);
      else if (t >= startOfYesterday) groups[1].sessions.push(s);
      else if (t >= startOfWeek) groups[2].sessions.push(s);
      else groups[3].sessions.push(s);
    }
    return groups.filter((g) => g.sessions.length > 0);
  }, [chatSessions]);

  const handleUpload = useCallback(async (file: File) => {
    try {
      showToast(`Uploading ${file.name}...`);
      const res = await uploadDocument(file, auth.userId || 'anonymous');
      setAttachment({ fileId: res.file_id, filename: res.filename, size: res.size });
      showToast('Document attached');
    } catch (err) {
      showToast(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }, [auth.userId, showToast]);

  const handleVoiceClick = useCallback(() => {
    showToast('🎤 Voice input is coming soon');
  }, [showToast]);

  // Streaming / generation control
  const [draft, setDraft] = useState<{ id: string; text: string; status: string | null; step?: string } | null>(null);
  const liveLangRef = useRef<Language>('en');
  const [stopController, setStopController] = useState<AbortController | null>(null);
  const [editingUserMessageId, setEditingUserMessageId] = useState<string | null>(null);

  // Freemium Gate
  const [freeQueriesCount, setFreeQueriesCount] = useState<number>(() => {
    return Number(localStorage.getItem('free_queries_count') || '0');
  });

  // Notice Form states
  const [noticeData, setNoticeData] = useState({
    sender_name: '',
    sender_address: '',
    sender_email: '',
    recipient_name: '',
    recipient_address: '',
    issue_type: 'Legal Dispute',
    issue_description: '',
    applicable_section: '',
    demand_amount: ''
  });
  const [noticeLoading, setNoticeLoading] = useState(false);
  const [noticeResult, setNoticeResult] = useState<any>(null);
  const [noticeError, setNoticeError] = useState('');

  // Pre-fill the notice sender from the logged-in profile
  useEffect(() => {
    if (auth.userName || auth.email) {
      setNoticeData((prev) => ({
        ...prev,
        sender_name: prev.sender_name || auth.userName || '',
        sender_email: prev.sender_email || auth.email || '',
      }));
    }
  }, [auth.userName, auth.email]);

  // Evidence Checklist states
  const [evidenceDomain, setEvidenceDomain] = useState<string>('consumer');
  const [evidenceChecklist, setEvidenceChecklist] = useState<any>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState('');

  // RTI states
  const [rtiData, setRtiData] = useState({
    applicant_name: '',
    applicant_email: '',
    applicant_phone: '',
    applicant_address: '',
    pio_office: '',
    pio_address: '',
    information_sought: ''
  });
  const [rtiLoading, setRtiLoading] = useState(false);
  const [rtiResult, setRtiResult] = useState<any>(null);
  const [rtiError, setRtiError] = useState('');

  // Case Strategy states
  const [strategyDescription, setStrategyDescription] = useState('');
  const [strategyDomain, setStrategyDomain] = useState('auto');
  const [strategyViewData, setStrategyViewData] = useState<CaseStrategyResult | null>(null);
  const [strategyLoading, setStrategyLoading] = useState(false);
  const [strategyError, setStrategyError] = useState('');

  // Document Audit states
  const [auditText, setAuditText] = useState('');
  const [auditDomain, setAuditDomain] = useState('rent');
  const [auditViewData, setAuditViewData] = useState<DocumentAuditResult | null>(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState('');

  // Fetch backend health and user PDFs
  useEffect(() => {
    healthCheck().catch(() => { /* backend offline; handled on query send */ });

    if (auth.userId) {
      getUserPDFs(auth.userId)
        .then((pdfs) => {
          setUserPDFs(pdfs);
          localStorage.setItem('cached_user_pdfs', JSON.stringify(pdfs));
        })
        .catch(console.error);

      getUserDocuments(auth.userId)
        .then(setUserDocuments)
        .catch(console.error);
    }
  }, [auth.userId]);

  // Auto scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, draft]);

  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim()) return;

    // Freemium restriction check
    if (!auth.isLoggedIn && freeQueriesCount >= 1) {
      alert("🔒 Free query limit reached. Please register or sign in to proceed.");
      return;
    }

    const userMessageId = Date.now().toString();
    const queryText = inputValue;

    setMessages((prev) => {
      if (!editingUserMessageId) {
        return [...prev, { id: userMessageId, type: 'user', content: queryText, timestamp: new Date() }];
      }
      const withoutEdited = [...prev];
      const idx = withoutEdited.findIndex((m) => m.id === editingUserMessageId);
      if (idx !== -1) {
        withoutEdited.splice(idx, 1);
        if (idx < withoutEdited.length && withoutEdited[idx].type === 'assistant') {
          withoutEdited.splice(idx, 1);
        }
      }
      return [...withoutEdited, { id: userMessageId, type: 'user', content: queryText, timestamp: new Date() }];
    });
    setEditingUserMessageId(null);
    setInputValue('');
    setLoading(true);
    liveLangRef.current = targetLanguage;

    const draftId = `draft-${Date.now()}`;
    const draftText = { current: '' };
    setDraft({ id: draftId, text: '', status: 'Starting...', step: 'starting' });

    const controller = new AbortController();
    setStopController(controller);
    const genId = ++generationIdRef.current;

    const resetLoading = () => {
      setLoading(false);
      setStopController((cur) => (cur === controller ? null : cur));
    };

    try {
      await streamQuery(queryText, targetLanguage, {
        onStatus: (_step, message, data) => {
          if (generationIdRef.current !== genId) return;
          if (data?.step === 'language' && typeof data?.language === 'string') {
            liveLangRef.current = data.language as Language;
          }
          setDraft((d) => (d ? { ...d, status: message || 'Working on it...', step: _step } : d));
        },
        onToken: (text) => {
          if (generationIdRef.current !== genId) return;
          draftText.current += text;
          setDraft((d) => (d ? { ...d, text: jsonDraftToText(draftText.current, liveLangRef.current) } : d));
        },
        onFinal: (data) => {
          if (generationIdRef.current !== genId) return;
          setDraft(null);
          const answerLang: Language = data.query_language || targetLanguage;
          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            type: 'assistant',
            content: data.response || data.full_response || jsonDraftToText(draftText.current, answerLang) || 'No output could be generated.',
            results: data.sources ?? [],
            domain: data.domain,
            confidence: data.confidence,
            lang: answerLang,
            tools: Array.isArray(data.tools) ? data.tools : undefined,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, assistantMessage]);

          if (!auth.isLoggedIn) {
            const nextCount = freeQueriesCount + 1;
            setFreeQueriesCount(nextCount);
            localStorage.setItem('free_queries_count', String(nextCount));
          }
          if (data.query_id) {
            localStorage.setItem('lastQueryId', data.query_id);
          }
          resetLoading();
        },
        onError: (message) => {
          if (generationIdRef.current !== genId) return;
          setDraft(null);
          const errMessage: Message = {
            id: (Date.now() + 1).toString(),
            type: 'assistant',
            content: draftText.current
              ? `${jsonDraftToText(draftText.current, liveLangRef.current)}\n\n❌ ${message}`
              : `❌ Error: ${message}`,
            lang: liveLangRef.current,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errMessage]);
          resetLoading();
        },
      }, controller.signal, auth.userId || undefined);
    } catch (error) {
      if (generationIdRef.current !== genId) return;
      const aborted = error instanceof DOMException && error.name === 'AbortError';
      setDraft(null);
      if (draftText.current) {
        const partial: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: jsonDraftToText(draftText.current, liveLangRef.current) + (aborted ? '\n\n⏹ Generation stopped.' : ''),
          lang: liveLangRef.current,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, partial]);
      } else if (!aborted) {
        const errMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: `❌ Error: ${error instanceof Error ? error.message : 'Failed to process query'}`,
          lang: liveLangRef.current,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errMessage]);
      }
      resetLoading();
    }
  }, [inputValue, auth.userId, freeQueriesCount, auth.isLoggedIn, targetLanguage, editingUserMessageId]);

  const cancelInFlight = useCallback(() => {
    stopController?.abort();
    generationIdRef.current += 1;
    setLoading(false);
    setDraft(null);
    setStopController(null);
  }, [stopController]);

  const handleStop = useCallback(() => {
    stopController?.abort();
  }, [stopController]);

  const handleEditMessage = useCallback((msgId: string, text: string) => {
    setEditingUserMessageId(msgId);
    setInputValue(text);
    setActiveView('chat');
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => bottomInputRef.current?.focus(), 50);
  }, []);

  const handleCopyMessage = useCallback((text: string) => {
    navigator.clipboard?.writeText(text).catch(() => { });
  }, []);

  // Notice Form Submit
  const handleNoticeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setNoticeLoading(true);
    setNoticeError('');
    setNoticeResult(null);
    try {
      const data = await generatePDFNotice(noticeData, auth.userId || 'anonymous');
      setNoticeResult(data);
      // Reload PDFs list
      if (auth.userId) {
        getUserPDFs(auth.userId)
          .then((pdfs) => {
            setUserPDFs(pdfs);
            localStorage.setItem('cached_user_pdfs', JSON.stringify(pdfs));
          })
          .catch(console.error);
      }
    } catch (err: any) {
      setNoticeError(err.message || 'Failed to generate notice');
    } finally {
      setNoticeLoading(false);
    }
  };

  // Evidence Checklist Submit
  const handleFetchChecklist = async (e: React.FormEvent) => {
    e.preventDefault();
    setEvidenceLoading(true);
    setEvidenceError('');
    setEvidenceChecklist(null);
    try {
      const data = await getEvidenceChecklist(evidenceDomain);
      setEvidenceChecklist(data);
    } catch (err: any) {
      setEvidenceError(err.message || 'Failed to fetch evidence');
    } finally {
      setEvidenceLoading(false);
    }
  };

  // RTI Form Submit
  const handleRtiSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setRtiLoading(true);
    setRtiError('');
    setRtiResult(null);
    try {
      const data = await generateRTIApplication(rtiData);
      setRtiResult(data);
    } catch (err: any) {
      setRtiError(err.message || 'Failed to generate RTI draft');
    } finally {
      setRtiLoading(false);
    }
  };

  // Case Strategy Submit
  const handleStrategySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStrategyLoading(true);
    setStrategyError('');
    setStrategyViewData(null);
    try {
      const data = await getStrategy({
        case_description: strategyDescription,
        ...(strategyDomain !== 'auto' ? { domain: strategyDomain } : {}),
      });
      setStrategyViewData(data);
    } catch (err: any) {
      setStrategyError(err.message || 'Failed to build strategy');
    } finally {
      setStrategyLoading(false);
    }
  };

  // Document Audit Submit
  const handleAuditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuditLoading(true);
    setAuditError('');
    setAuditViewData(null);
    try {
      const data = await auditDocument(auditText, auditDomain);
      setAuditViewData(data);
    } catch (err: any) {
      setAuditError(err.message || 'Failed to audit document');
    } finally {
      setAuditLoading(false);
    }
  };

  const openStrategyFromCard = useCallback((s: CaseStrategyResult) => {
    setStrategyViewData(s);
    setActiveView('strategy');
  }, []);

  const openAuditFromCard = useCallback((a: DocumentAuditResult) => {
    setAuditViewData(a);
    setActiveView('audit');
  }, []);

  // Chat sessions: sync active session with messages + persist to localStorage
  useEffect(() => {
    if (activeChatId && messages.length > 0) {
      setChatSessions((prev) => prev.map((s) => (s.id === activeChatId ? { ...s, messages: stripContent(messages), timestamp: new Date() } : s)));
    }
  }, [messages, activeChatId]);

  useEffect(() => {
    const storageKey = `chat_sessions_v2_${auth.userId || 'anonymous'}`;
    try {
      localStorage.setItem(storageKey, JSON.stringify(chatSessions));
    } catch {
      // ignore quota / serialization errors
    }
  }, [chatSessions, auth.userId]);

  // Persist the active chat when the user leaves the site / hides the tab,
  // so work is never lost even without clicking "New Chat".
  const messagesRef = useRef(messages);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  const chatSessionsRef = useRef(chatSessions);
  useEffect(() => { chatSessionsRef.current = chatSessions; }, [chatSessions]);
  const activeChatIdRef = useRef(activeChatId);
  useEffect(() => { activeChatIdRef.current = activeChatId; }, [activeChatId]);
  const userIdRef = useRef(auth.userId || 'anonymous');
  useEffect(() => { userIdRef.current = auth.userId || 'anonymous'; }, [auth.userId]);

  useEffect(() => {
    const saveActiveChat = () => {
      const msgs = messagesRef.current;
      const key = `chat_sessions_v2_${userIdRef.current}`;
      if (msgs.length === 0) return;
      let next = chatSessionsRef.current;
      const activeId = activeChatIdRef.current;
      if (activeId) {
        next = next.map((s) => (s.id === activeId ? { ...s, messages: stripContent(msgs), timestamp: new Date() } : s));
      } else {
        const firstUser = msgs.find((m) => m.type === 'user');
        const raw = firstUser?.content || 'New chat';
        const title = raw.split(/\s+/).slice(0, 5).join(' ') || 'New chat';
        next = [{ id: `saved-${Date.now()}`, title, messages: stripContent(msgs), timestamp: new Date() }, ...next].slice(0, 50);
      }
      try {
        localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // ignore quota / serialization errors
      }
    };
    const onHide = () => {
      if (document.visibilityState === 'hidden') saveActiveChat();
    };
    window.addEventListener('beforeunload', saveActiveChat);
    document.addEventListener('visibilitychange', onHide);
    return () => {
      window.removeEventListener('beforeunload', saveActiveChat);
      document.removeEventListener('visibilitychange', onHide);
    };
  }, []);

  const startNewChat = () => {
    cancelInFlight();
    if (messages.length > 0) {
      const firstUser = messages.find((m) => m.type === 'user');
      const raw = firstUser?.content || 'New chat';
      const title = raw.split(/\s+/).slice(0, 5).join(' ') || 'New chat';
      const session: ChatSession = { id: Date.now().toString(), title, messages: stripContent(messages), timestamp: new Date() };
      setChatSessions((prev) => [session, ...prev].slice(0, 20));
    }
    setMessages([]);
    setActiveChatId(null);
    setExpandedSource(null);
    setInputValue('');
    setEditingUserMessageId(null);
    setActiveView('chat');
  };

  const openChat = (session: ChatSession) => {
    cancelInFlight();
    setMessages([...session.messages]);
    setActiveChatId(session.id);
    setExpandedSource(null);
    setActiveView('chat');
  };

  const closeChatMenu = () => {
    setChatMenuId(null);
    setConfirmDeleteId(null);
  };

  const deleteChat = (sessionId: string) => {
    cancelInFlight();
    setChatSessions((prev) => prev.filter((s) => s.id !== sessionId));
    closeChatMenu();
    if (activeChatId === sessionId) {
      setActiveChatId(null);
      setMessages([]);
      setExpandedSource(null);
    }
  };

  const openBookmark = useCallback((bm: { chatId: string | null; messageId: string }) => {
    if (bm.chatId) {
      const session = chatSessions.find((s) => s.id === bm.chatId);
      if (session) {
        openChat(session);
        setTimeout(() => {
          document.getElementById(`msg-${bm.messageId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 120);
        return;
      }
    }
    showToast('Bookmark message not found in an open chat');
  }, [chatSessions, openChat, showToast]);

  if (!auth.isLoggedIn) {
    return <LoginPage />;
  }

  const sessionTitle = activeChatId
    ? chatSessions.find((s) => s.id === activeChatId)?.title || 'Current chat'
    : 'Current chat';

  return (
    <div className="h-screen bg-bg flex overflow-hidden text-ink font-sans">
      {/* ── Sidebar ─────────────────────────────────────────────── */}
      <aside
        className={`${sidebarOpen ? 'w-64' : 'w-0'
          } bg-surface border-r border-line flex flex-col transition-[width] duration-300 ease-[cubic-bezier(.22,1,.36,1)] overflow-hidden z-20 flex-shrink-0`}
      >
        {/* Brand row */}
        <div className="px-4 pt-4 pb-3 flex items-center justify-between">
          <div className="flex items-center gap-2 select-none">
            <div className="w-8 h-8 rounded-lg bg-btn text-white flex items-center justify-center shadow-sm">
              <Scale className="w-4.5 h-4.5" />
            </div>
            <span className="text-[15px] font-bold tracking-tight">court<span className="text-accent">Room</span>.ai</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1.5 rounded-lg border border-line bg-card text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
            title="Close sidebar"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>
        </div>

        {/* New Chat */}
        <div className="px-3 pb-2">
          <button
            onClick={startNewChat}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-btn hover:bg-btn-hover text-white text-sm font-semibold transition-colors cursor-pointer shadow-sm"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>

        {/* AI Legal Advisor */}
        <div className="px-3 pb-1">
          <button
            onClick={() => { setActiveView('chat'); }}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${activeView === 'chat' ? 'bg-active text-ink font-semibold' : 'text-ink-2 hover:bg-hover hover:text-ink'
              }`}
          >
            <MessageSquare className={`w-4 h-4 ${activeView === 'chat' ? 'text-accent' : 'text-ink-3'}`} />
            AI Legal Advisor
          </button>
        </div>

        {/* Artifacts accordion */}
        <div className="px-3 pb-1">
          <button
            onClick={() => setArtifactsOpen(!artifactsOpen)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold text-ink-2 hover:bg-hover hover:text-ink transition-colors cursor-pointer"
          >
            <span className="flex items-center gap-3">
              <ClipboardList className="w-4 h-4 text-ink-3" />
              Artifacts
            </span>
            <ChevronDown className={`w-3.5 h-3.5 text-ink-3 transition-transform duration-200 ${artifactsOpen ? 'rotate-180' : ''}`} />
          </button>
          {artifactsOpen && (
            <div className="mt-0.5 space-y-0.5">
              {[
                { label: 'Legal Notice', view: 'notice', icon: FileText, action: () => setActiveView('notice') },
                { label: 'Case Strategy', view: 'strategy', icon: ClipboardList, action: () => setActiveView('strategy') },
                { label: 'Document Audit', view: 'audit', icon: ShieldCheck, action: () => setActiveView('audit') },
                { label: 'Evidence Checklist', view: 'evidence', icon: CheckCircle2, action: () => setActiveView('evidence') },
                { label: 'RTI Application', view: 'rti', icon: Zap, action: () => setActiveView('rti') },
              ].map((item) => (
                <button
                  key={item.label}
                  onClick={() => { item.action(); setArtifactsOpen(false); }}
                  className={`w-full flex items-center gap-3 pl-[34px] pr-3 py-2 rounded-lg text-sm transition-colors cursor-pointer ${activeView === item.view ? 'bg-active text-ink font-semibold' : 'text-ink-2 hover:bg-hover hover:text-ink'
                    }`}
                >
                  <item.icon className={`w-4 h-4 ${activeView === item.view ? 'text-accent' : 'text-ink-3'}`} />
                  {item.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Documents accordion */}
        <div className="px-3 pb-1">
          <button
            onClick={() => setDocsOpen(!docsOpen)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold text-ink-2 hover:bg-hover hover:text-ink transition-colors cursor-pointer"
          >
            <span className="flex items-center gap-3">
              <Folder className="w-4 h-4 text-ink-3" />
              Documents
            </span>
            <ChevronDown className={`w-3.5 h-3.5 text-ink-3 transition-transform duration-200 ${docsOpen ? 'rotate-180' : ''}`} />
          </button>
          {docsOpen && (
            <div className="mt-0.5 space-y-0.5">
              {[
                { label: 'My Documents', icon: FileText, action: () => setShowDocsModal(true) },
              ].map((item) => (
                <button
                  key={item.label}
                  onClick={() => { item.action(); setDocsOpen(false); }}
                  className="w-full flex items-center gap-3 pl-[34px] pr-3 py-2 rounded-lg text-sm transition-colors cursor-pointer text-ink-2 hover:bg-hover hover:text-ink"
                >
                  <item.icon className="w-4 h-4 text-ink-3" />
                  {item.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-3 py-2 border-t border-line-2 mt-2 space-y-4">
          {/* Bookmarks */}
          <div>
            <div className="flex items-center gap-2 px-3 py-1.5 text-[11px] font-bold text-ink-3 uppercase tracking-wider">
              <Star className="w-3.5 h-3.5 text-warn" />
              Bookmarks
            </div>
            {bookmarks.length === 0 ? (
              <p className="text-xs text-ink-3 px-3 py-1.5">Star a reply to save it here</p>
            ) : (
              <div className="space-y-0.5">
                {bookmarks.slice(0, 10).map((bm) => (
                  <button
                    key={bm.id}
                    onClick={() => openBookmark(bm)}
                    className="w-full flex items-start gap-2 px-3 py-2 rounded-lg text-left transition-colors cursor-pointer hover:bg-hover"
                    title={bm.text}
                  >
                    <Star className="w-3 h-3 text-warn flex-shrink-0 mt-0.5" />
                    <span className="min-w-0 flex-1">
                      <span className="block text-xs text-ink-2 line-clamp-2 leading-snug">{bm.text}</span>
                      <span className="block text-[9px] text-ink-3 mt-0.5">{bm.chatTitle}</span>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* History (grouped) */}
          <div>
            <div className="flex items-center gap-2 px-3 py-1.5 text-[11px] font-bold text-ink-3 uppercase tracking-wider">
              <MessageSquare className="w-3.5 h-3.5" />
              History
            </div>
            {groupSessions().length === 0 ? (
              <p className="text-xs text-ink-3 px-3 py-1.5">No chats yet — ask a legal question</p>
            ) : (
              groupSessions().map((group) => (
                <div key={group.label} className="mb-1.5">
                  <p className="px-3 py-1 text-[10px] font-semibold text-ink-3">{group.label}</p>
                  <div className="space-y-0.5">
                    {group.sessions.slice(0, 12).map((session) => (
                      <div key={session.id} className="relative group/item">
                        <div className={`w-full flex items-center gap-1 pl-3 pr-2 py-2 rounded-lg transition-colors text-xs cursor-pointer ${activeChatId === session.id ? 'bg-active' : 'hover:bg-hover'
                          }`}>
                          <button
                            onClick={() => { openChat(session); closeChatMenu(); }}
                            className="flex items-center gap-2 min-w-0 flex-1 text-left"
                            title={session.title}
                          >
                            <MessageSquare className={`w-3.5 h-3.5 flex-shrink-0 ${activeChatId === session.id ? 'text-accent' : 'text-ink-3'}`} />
                            <span className="min-w-0 flex-1">
                              <span className="block text-ink-2 line-clamp-1 leading-snug">{shortTitle(session.title)}</span>
                            </span>
                          </button>
                          <button
                            onClick={() => {
                              if (chatMenuId === session.id) closeChatMenu();
                              else { setChatMenuId(session.id); setConfirmDeleteId(null); }
                            }}
                            className="p-1 rounded text-ink-3 opacity-0 group-hover/item:opacity-100 hover:text-ink hover:bg-hover transition cursor-pointer shrink-0"
                            title="Options"
                          >
                            <MoreHorizontal className="w-3.5 h-3.5" />
                          </button>
                        </div>

                        {chatMenuId === session.id && (
                          <div className="absolute right-2 top-8 z-30 w-40 bg-card border border-line rounded-xl shadow-pop p-1.5">
                            <button
                              onClick={() => setConfirmDeleteId(session.id)}
                              className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-err hover:bg-err/5 transition text-xs font-semibold cursor-pointer"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                              Delete
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Sidebar footer: Settings + Profile */}
        <div className="border-t border-line px-3 py-2.5 space-y-1 relative">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => { setActiveView('settings'); setProfileMenuOpen(false); }}
              className={`flex-1 flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer ${activeView === 'settings' ? 'bg-active text-ink font-semibold' : 'text-ink-2 hover:bg-hover hover:text-ink'
                }`}
            >
              <Settings className={`w-4 h-4 ${activeView === 'settings' ? 'text-accent' : 'text-ink-3'}`} />
              Settings
            </button>
            <button
              onClick={toggleTheme}
              className="p-2.5 rounded-lg border border-line bg-card text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>

          {profileMenuOpen && (
            <div className="absolute bottom-16 left-3 right-3 bg-card border border-line rounded-xl shadow-pop p-1.5 z-30 space-y-1">
              <button
                onClick={() => {
                  setShowPDFModal(true);
                  setProfileMenuOpen(false);
                }}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-ink-2 hover:bg-hover transition text-xs font-semibold cursor-pointer"
              >
                <Download className="w-4 h-4 text-accent" />
                All PDFs ({userPDFs.length})
              </button>
              <button
                onClick={auth.logout}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-err hover:bg-err/5 transition text-xs font-semibold cursor-pointer"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          )}

          <button
            onClick={() => { setProfileMenuOpen(!profileMenuOpen); }}
            className="w-full flex items-center justify-between gap-2 rounded-xl p-2 hover:bg-hover transition cursor-pointer"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-full bg-active text-ink flex items-center justify-center font-bold text-xs flex-shrink-0 select-none border border-line">
                {auth.userName?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="text-left min-w-0">
                <p className="text-xs font-semibold text-ink leading-tight truncate">{auth.userName || 'User'}</p>
                <p className="text-[10px] text-ink-3 leading-normal">Free plan</p>
              </div>
            </div>
            <ChevronDown className={`w-3.5 h-3.5 text-ink-3 transition-transform duration-200 flex-shrink-0 ${profileMenuOpen ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </aside>


      {/* Main Container */}
      <div className="flex-1 flex flex-col overflow-hidden bg-bg">
        {/* Top Header */}
        <header className="border-b border-line bg-card/85 backdrop-blur-md px-6 py-3 flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1.5 rounded-lg border border-line bg-card text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
                title="Open sidebar"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
              </button>
            )}

            {/* Workspace dropdown (placeholder) */}
            <div className="relative">
              <button
                onClick={() => { setWorkspaceMenuOpen(!workspaceMenuOpen); setNotifMenuOpen(false); }}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-semibold text-ink hover:bg-hover transition cursor-pointer"
              >
                My Workspace
                <ChevronDown className={`w-3.5 h-3.5 text-ink-3 transition-transform duration-200 ${workspaceMenuOpen ? 'rotate-180' : ''}`} />
              </button>
              {workspaceMenuOpen && (
                <div className="absolute left-0 top-11 w-56 bg-card border border-line rounded-xl shadow-pop p-1.5 z-30 space-y-0.5">
                  {['My Workspace', 'Personal', 'Team Workspace'].map((ws, i) => (
                    <button
                      key={ws}
                      onClick={() => { setWorkspaceMenuOpen(false); showToast(i === 0 ? '' : 'Workspace switching is coming soon'); }}
                      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors cursor-pointer ${i === 0 ? 'bg-active text-ink font-semibold' : 'text-ink-2 hover:bg-hover hover:text-ink'
                        }`}
                    >
                      <CheckCircle2 className={`w-4 h-4 ${i === 0 ? 'text-ok' : 'text-ink-3'}`} />
                      {ws}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <LanguageDropdown
              value={targetLanguage}
              onChange={setTargetLanguage}
            />

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg border border-line bg-card text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            {/* Notifications bell (placeholder) */}
            <div className="relative">
              <button
                onClick={() => { setNotifMenuOpen(!notifMenuOpen); setWorkspaceMenuOpen(false); }}
                className="relative p-2 rounded-lg border border-line bg-card text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
                title="Notifications"
              >
                <Bell className="w-4 h-4" />
                <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-warn"></span>
              </button>
              {notifMenuOpen && (
                <div className="absolute right-0 top-11 w-60 bg-card border border-line rounded-xl shadow-pop p-1.5 z-30">
                  <p className="px-3 py-2.5 text-xs text-ink-3">No new notifications</p>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 pl-1.5">
              <div className="w-8 h-8 rounded-full bg-active text-ink flex items-center justify-center text-xs font-bold border border-line select-none">
                {auth.userName?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="hidden md:block text-right">
                <p className="text-xs font-semibold text-ink leading-tight">{auth.userName || 'User'}</p>
                {auth.email && <p className="text-[10px] text-ink-3 leading-normal">{auth.email}</p>}
              </div>
            </div>
          </div>
        </header>

        {/* View Selection Controller */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Chat Advisor View */}
          {activeView === 'chat' && (
            <div className="h-full flex flex-col overflow-hidden">
              {messages.length === 0 ? (
                /* Welcome / Search State (Zero Scroll) */
                <div className="h-full flex flex-col justify-between items-center pt-10 px-6 pb-0 bg-bg overflow-hidden select-none">
                  {/* Top Spacer */}
                  <div className="h-2"></div>

                  {/* Centered Hero */}
                  <div className="max-w-2xl w-full flex flex-col items-center flex-1 justify-center px-6">
                    <div className="flex items-center gap-3 mb-3 text-[15px] font-semibold text-ink-2">
                      <span className="w-10 h-10 rounded-xl bg-btn text-white flex items-center justify-center shadow-sm">
                        <Scale className="w-5 h-5" />
                      </span>
                      <span>{getGreeting()}</span>
                    </div>

                    <h1 className="text-3xl md:text-4xl font-bold text-ink tracking-tight text-center mb-2.5">
                      ⚖️ Legal AI Assistant
                    </h1>
                    <p className="text-sm text-ink-2 text-center mb-8 max-w-md">
                      Ask about Indian laws, get cited answers in your language.
                    </p>

                    {/* Quick Action Cards */}
                    <div className="grid grid-cols-2 gap-3 w-full mb-8">
                      {[
                        { icon: '🛒', label: 'Defective product or refund', q: 'I bought a defective product and the seller is refusing a refund. What are my rights?' },
                        { icon: '🏠', label: 'Rent or landlord dispute', q: 'My landlord is not returning my security deposit. What should I do?' },
                        { icon: '💳', label: 'Online fraud or hacking', q: 'Someone hacked my bank account and stole money. What legal action can I take?' },
                        { icon: '⚖️', label: 'Workplace or salary issue', q: 'My employer has not paid my salary for three months. What are my legal options?' },
                        { icon: '🧭', label: 'Build a case strategy', q: 'Build me a case strategy: my landlord cut the water supply and is not returning my 50000 deposit since 15-07-2024. What should I do?' },
                        { icon: '🔍', label: 'Audit my document', q: 'Please audit this rent agreement document I have' },
                      ].map((card, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            setInputValue(card.q);
                            setTimeout(() => handleSendMessage(), 60);
                          }}
                          className="flex items-center gap-3 p-4 rounded-2xl bg-card border border-line hover:border-[#cfcfcf] hover:shadow-card transition-colors text-left cursor-pointer"
                        >
                          <span className="text-xl">{card.icon}</span>
                          <span className="text-xs font-medium text-ink-2 leading-snug">{card.label}</span>
                        </button>
                      ))}
                    </div>

                    {/* Hero Input */}
                    <div className="w-full bg-card border border-line focus-within:border-[#cfcfcf] focus-within:ring-4 focus-within:ring-black/[0.03] rounded-2xl px-4 py-2.5 shadow-card relative transition-all">
                      <textarea
                        ref={bottomInputRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        placeholder="Ask about any Indian law..."
                        className="w-full bg-transparent border-0 outline-none focus:ring-0 text-ink placeholder-ink-3 text-sm resize-none min-h-[44px] leading-relaxed"
                      />

                      {attachment && (
                        <div className="flex items-center gap-2 mt-1.5 px-1">
                          <span className="inline-flex items-center gap-1.5 text-[11px] bg-active text-ink-2 border border-line rounded-lg px-2.5 py-1 max-w-[240px]">
                            <Paperclip className="w-3 h-3 text-accent flex-shrink-0" />
                            <span className="truncate">{attachment.filename}</span>
                            <button onClick={() => setAttachment(null)} className="text-ink-3 hover:text-ink cursor-pointer ml-1">
                              <X className="w-3 h-3" />
                            </button>
                          </span>
                        </div>
                      )}

                      <div className="flex items-center justify-between pt-2">
                        <div className="flex items-center gap-1">
                          {/* Upload (real) */}
                          <label
                            className="p-2 rounded-lg text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
                            title="Attach a document"
                          >
                            <Paperclip className="w-4.5 h-4.5" />
                            <input
                              type="file"
                              className="hidden"
                              onChange={(e) => {
                                const f = e.target.files?.[0];
                                if (f) handleUpload(f);
                                e.target.value = '';
                              }}
                            />
                          </label>
                          {/* Voice (stub) */}
                          <button
                            onClick={handleVoiceClick}
                            className="p-2 rounded-lg text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
                            title="Voice input"
                          >
                            <Mic className="w-4.5 h-4.5" />
                          </button>
                        </div>

                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-ink-3 bg-bg px-2 py-0.5 rounded-full border border-line font-medium">
                            Qwen 2.5 Law
                          </span>
                          {loading ? (
                            <button
                              onClick={handleStop}
                              className="p-2 rounded-xl bg-btn hover:bg-btn-hover text-white transition flex items-center justify-center cursor-pointer"
                              title="Stop generating"
                            >
                              <Square className="w-4 h-4" />
                            </button>
                          ) : (
                            <button
                              onClick={handleSendMessage}
                              disabled={!inputValue.trim() && !attachment}
                              className="p-2 rounded-xl bg-btn hover:bg-btn-hover text-white transition disabled:opacity-40 disabled:hover:bg-btn flex items-center justify-center cursor-pointer"
                              title="Send"
                            >
                              <Send className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Results / Messages State (Scrollable) */
                <div className="flex-1 flex flex-col overflow-hidden bg-bg">
                  {/* Messages list (full-bleed scroll so the scrollbar hugs the desktop edge) */}
                  <div className="flex-1 overflow-y-auto">
                    <div className="max-w-3xl mx-auto w-full p-6 space-y-6">
                      {messages.map((msg) => (
                        <div key={msg.id} id={`msg-${msg.id}`} className={`flex ${msg.type === 'user' ? 'flex-col items-end' : 'gap-3 justify-start'}`}>
                          <div className={`flex gap-3 ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.type === 'assistant' && (
                              <div className="w-8 h-8 rounded-full bg-card border border-line flex items-center justify-center flex-shrink-0 shadow-sm">
                                <Scale className="w-4 h-4 text-accent" />
                              </div>
                            )}

                            <div
                              lang={msg.lang === 'gu' ? 'gu' : undefined}
                              className={`max-w-2xl rounded-2xl px-5 py-3.5 shadow-card ${msg.type === 'user'
                                ? 'bg-btn text-white font-medium rounded-br-md'
                                : 'bg-card border border-line text-ink'
                                }`}
                            >
                              {msg.type === 'assistant' && msg.domain && (
                                <div className="mb-3 flex items-center gap-4 text-xs font-semibold text-accent bg-active px-2.5 py-1.5 rounded-lg border border-line">
                                  <span>{localizeDomain(msg.lang || 'en', msg.domain)}</span>
                                </div>
                              )}

                              <p className="text-sm md:text-[15px] whitespace-pre-wrap leading-relaxed">{msg.content}</p>

                              {msg.type === 'assistant' && msg.tools && msg.tools.length > 0 && (
                                <div className="space-y-2">
                                  {msg.tools.map((tool, idx) => (
                                    <ToolArtifactCard
                                      key={idx}
                                      tool={tool}
                                      onOpenStrategy={openStrategyFromCard}
                                      onOpenAudit={openAuditFromCard}
                                      onContinue={() => bottomInputRef.current?.focus()}
                                    />
                                  ))}
                                </div>
                              )}

                              {msg.type === 'assistant' && msg.results && msg.results.length > 0 && (
                                <div className="mt-4 border-t border-line-2 pt-3">
                                  <button
                                    onClick={() => setSourcesOpen((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }))}
                                    className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-active hover:bg-hover transition-colors cursor-pointer"
                                  >
                                    <span className="text-[11px] font-semibold text-ink-2">
                                      {uiLabel(msg.lang || 'en', 'sources')} ({msg.results.length})
                                    </span>
                                    <span className="flex items-center gap-1.5 text-[11px] font-medium text-accent">
                                      {sourcesOpen[msg.id] ? uiLabel(msg.lang || 'en', 'hide') : uiLabel(msg.lang || 'en', 'view')}
                                      <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${sourcesOpen[msg.id] ? 'rotate-180' : ''}`} />
                                    </span>
                                  </button>

                                  {sourcesOpen[msg.id] && (
                                    <div className="mt-2 grid grid-cols-1 gap-2">
                                      {msg.results.map((result, idx) => {
                                        const cardKey = `${msg.id}-${idx}`;
                                        const isOpen = expandedSource === cardKey;
                                        return (
                                          <div key={idx} className="p-3 rounded-xl bg-card border border-line text-xs">
                                            <div className="flex items-center justify-between gap-2 mb-1.5">
                                              <div className="flex items-center gap-2 min-w-0">
                                                <span className="w-5 h-5 rounded-full bg-active text-accent text-[10px] font-bold flex items-center justify-center flex-shrink-0">{idx + 1}</span>
                                                <span className="font-semibold text-ink truncate">{result.section_title || result.section}</span>
                                              </div>
                                              <span className="text-[9px] px-2 py-0.5 rounded-full bg-bg border border-line text-ink-2 uppercase tracking-wide whitespace-nowrap flex-shrink-0">{result.source_act}</span>
                                            </div>
                                            <div className="flex items-center gap-1.5 mb-1">
                                              <span className={`text-[9px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wide whitespace-nowrap ${(result.status || 'active') === 'historical' ? 'bg-amber-50 text-amber-700 border border-amber-200' : (result.status || 'active') === 'pending' ? 'bg-blue-50 text-blue-700 border border-blue-200' : 'bg-emerald-50 text-emerald-700 border border-emerald-200'}`}>
                                                {result.status || 'active'}
                                              </span>
                                              {result.replaced_by && (
                                                <span className="text-[9px] text-ink-3 whitespace-nowrap truncate">replaced by {result.replaced_by}</span>
                                              )}
                                            </div>
                                            <div className="text-ink-3 mb-1.5">
                                              <span className="text-accent font-semibold">{result.section}</span>
                                              {result.topic && <span> · {result.topic}</span>}
                                            </div>
                                            <div className={`text-ink-3 italic text-[11px] ${isOpen ? '' : 'line-clamp-2'}`}>"{result.content_preview}"</div>
                                            {result.content && result.content.length > 300 && (
                                              <button
                                                onClick={() => setExpandedSource(isOpen ? null : cardKey)}
                                                className="mt-1.5 text-[10px] font-semibold text-accent hover:underline cursor-pointer"
                                              >
                                                {isOpen ? `${uiLabel(msg.lang || 'en', 'read_less')} ▴` : `${uiLabel(msg.lang || 'en', 'read_full')} ▾`}
                                              </button>
                                            )}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>

                            {msg.type === 'user' && (
                              <div className="w-8 h-8 rounded-full bg-active border border-line flex items-center justify-center flex-shrink-0">
                                <span className="text-accent text-sm font-bold">{auth.userName?.charAt(0).toUpperCase()}</span>
                              </div>
                            )}
                          </div>
                          <div className="flex items-center gap-1.5 mt-1.5 pl-1">
                            {msg.type === 'user' ? (
                              <>
                                <button
                                  onClick={() => handleEditMessage(msg.id, msg.content)}
                                  className="flex items-center gap-1 text-[10px] text-ink-3 hover:text-accent transition cursor-pointer"
                                  title="Edit and resend"
                                >
                                  <Pencil className="w-3 h-3" /> {uiLabel(msg.lang || 'en', 'edit')}
                                </button>
                                <button
                                  onClick={() => handleCopyMessage(msg.content)}
                                  className="flex items-center gap-1 text-[10px] text-ink-3 hover:text-accent transition cursor-pointer"
                                  title="Copy query"
                                >
                                  <Copy className="w-3 h-3" /> {uiLabel(msg.lang || 'en', 'copy')}
                                </button>
                              </>
                            ) : (
                              <>
                                <button
                                  onClick={() => toggleBookmark(msg, sessionTitle)}
                                  className={`flex items-center gap-1 text-[10px] transition cursor-pointer ${bookmarks.some((b) => b.messageId === msg.id) ? 'text-warn' : 'text-ink-3 hover:text-warn'
                                    }`}
                                  title={bookmarks.some((b) => b.messageId === msg.id) ? 'Remove bookmark' : 'Bookmark this reply'}
                                >
                                  <Star className={`w-3 h-3 ${bookmarks.some((b) => b.messageId === msg.id) ? 'fill-warn' : ''}`} />
                                  {bookmarks.some((b) => b.messageId === msg.id) ? uiLabel(msg.lang || 'en', 'saved') : uiLabel(msg.lang || 'en', 'save')}
                                </button>
                                <button
                                  onClick={() => handleCopyMessage(msg.content)}
                                  className="flex items-center gap-1 text-[10px] text-ink-3 hover:text-accent transition cursor-pointer"
                                  title="Copy reply"
                                >
                                  <Copy className="w-3 h-3" /> {uiLabel(msg.lang || 'en', 'copy')}
                                </button>
                              </>
                            )}
                          </div>
                        </div>
                      ))}

                      {draft && (
                        <div className="flex gap-3 justify-start">
                          <div className="w-8 h-8 rounded-full bg-card border border-line flex items-center justify-center flex-shrink-0">
                            <Scale className="w-4 h-4 text-accent animate-pulse" />
                          </div>
                          <div lang={liveLangRef.current === 'gu' ? 'gu' : undefined} className="max-w-2xl bg-card border border-line rounded-2xl px-5 py-3.5 shadow-card">
                            {liveLangRef.current !== 'en' && (draft.step === 'formatting' || draft.step === 'translating') ? (
                              <div className="flex items-center gap-2 text-xs text-accent">
                                <Loader className="w-4 h-4 animate-spin" />
                                {`Translating answer to ${getLanguageInfo(liveLangRef.current).native}...`}
                              </div>
                            ) : draft.text ? (
                              <p className="text-sm md:text-[15px] whitespace-pre-wrap leading-relaxed">{draft.text}</p>
                            ) : (
                              <div className="flex items-center gap-2 text-xs text-accent">
                                <Loader className="w-4 h-4 animate-spin" />
                                {draft.status ?? 'Working on it...'}
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                  </div>

                  {/* Input bar under messages */}
                  <div className="p-4 md:p-5 max-w-3xl mx-auto w-full">
                    <div className="bg-card border border-line focus-within:border-[#cfcfcf] focus-within:ring-4 focus-within:ring-black/[0.03] rounded-2xl px-4 py-2.5 shadow-card transition-all">
                      <textarea
                        ref={bottomInputRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        placeholder="Describe your legal issue... (Enter to send, Shift+Enter for new line)"
                        className="flex-1 w-full bg-transparent border-0 outline-none focus:ring-0 text-ink placeholder-ink-3 text-sm md:text-base resize-none leading-relaxed"
                        rows={1}
                      />

                      {attachment && (
                        <div className="flex items-center gap-2 mt-1.5 px-1">
                          <span className="inline-flex items-center gap-1.5 text-[11px] bg-active text-ink-2 border border-line rounded-lg px-2.5 py-1 max-w-[240px]">
                            <Paperclip className="w-3 h-3 text-accent flex-shrink-0" />
                            <span className="truncate">{attachment.filename}</span>
                            <button onClick={() => setAttachment(null)} className="text-ink-3 hover:text-ink cursor-pointer ml-1">
                              <X className="w-3 h-3" />
                            </button>
                          </span>
                        </div>
                      )}

                      <div className="flex items-center justify-between pt-2">
                        <div className="flex items-center gap-1">
                          <label
                            className="p-2 rounded-lg text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
                            title="Attach a document"
                          >
                            <Paperclip className="w-4.5 h-4.5" />
                            <input
                              type="file"
                              className="hidden"
                              onChange={(e) => {
                                const f = e.target.files?.[0];
                                if (f) handleUpload(f);
                                e.target.value = '';
                              }}
                            />
                          </label>
                          <button
                            onClick={handleVoiceClick}
                            className="p-2 rounded-lg text-ink-3 hover:text-ink hover:bg-hover transition cursor-pointer"
                            title="Voice input"
                          >
                            <Mic className="w-4.5 h-4.5" />
                          </button>
                        </div>

                        <div className="flex items-center gap-2">
                          {loading ? (
                            <button
                              onClick={handleStop}
                              className="p-2 rounded-xl bg-btn hover:bg-btn-hover text-white transition flex items-center justify-center flex-shrink-0 cursor-pointer"
                              title="Stop generating"
                            >
                              <Square className="w-4 h-4" />
                            </button>
                          ) : (
                            <button
                              onClick={handleSendMessage}
                              disabled={!inputValue.trim()}
                              className="p-2 rounded-xl bg-btn hover:bg-btn-hover text-white transition disabled:opacity-40 disabled:hover:bg-btn flex items-center justify-center flex-shrink-0 cursor-pointer"
                              title="Send"
                            >
                              <Send className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Notice Generator View */}
          {activeView === 'notice' && (
            <div className="p-6 md:p-8 max-w-3xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-ink mb-1">📄 Legal Notice</h2>
                <p className="text-ink-2 text-sm">Your name is filled from your profile. Add who it's for and what happened, then generate and download the PDF.</p>
              </div>

              <form onSubmit={handleNoticeSubmit} className="ui-panel p-6 space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="ui-label">Sender Name <span className="text-ink-3 font-normal">(from your profile)</span></label>
                    <input type="text" className="ui-input" value={noticeData.sender_name} onChange={(e) => setNoticeData({ ...noticeData, sender_name: e.target.value })} required />
                  </div>
                  <div>
                    <label className="ui-label">Recipient Name</label>
                    <input type="text" className="ui-input" placeholder="Who is the notice for?" value={noticeData.recipient_name} onChange={(e) => setNoticeData({ ...noticeData, recipient_name: e.target.value })} required />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="ui-label">Sender Email</label>
                    <input type="email" className="ui-input" value={noticeData.sender_email} onChange={(e) => setNoticeData({ ...noticeData, sender_email: e.target.value })} />
                  </div>
                  <div>
                    <label className="ui-label">Claim Amount (Rs.) <span className="text-ink-3 font-normal">(optional)</span></label>
                    <input type="text" className="ui-input" placeholder="e.g. 50000" value={noticeData.demand_amount} onChange={(e) => setNoticeData({ ...noticeData, demand_amount: e.target.value })} />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="ui-label">Sender Address <span className="text-ink-3 font-normal">(optional)</span></label>
                    <textarea rows={2} className="ui-input resize-none" value={noticeData.sender_address} onChange={(e) => setNoticeData({ ...noticeData, sender_address: e.target.value })} />
                  </div>
                  <div>
                    <label className="ui-label">Recipient Address <span className="text-ink-3 font-normal">(optional)</span></label>
                    <textarea rows={2} className="ui-input resize-none" value={noticeData.recipient_address} onChange={(e) => setNoticeData({ ...noticeData, recipient_address: e.target.value })} />
                  </div>
                </div>

                <div>
                  <label className="ui-label">What happened?</label>
                  <textarea rows={3} className="ui-input" placeholder="Describe the facts: dates, amounts, what went wrong..." value={noticeData.issue_description} onChange={(e) => setNoticeData({ ...noticeData, issue_description: e.target.value })} required />
                </div>

                <div>
                  <label className="ui-label">Applicable Law Section <span className="text-ink-3 font-normal">(optional)</span></label>
                  <input type="text" className="ui-input" placeholder="e.g. Section 12 of Consumer Protection Act" value={noticeData.applicable_section} onChange={(e) => setNoticeData({ ...noticeData, applicable_section: e.target.value })} />
                </div>

                <button type="submit" disabled={noticeLoading} className="ui-btn-primary">
                  {noticeLoading ? <Loader className="w-5 h-5 animate-spin" /> : 'Compile Legal Notice PDF'}
                </button>
              </form>

              {noticeError && <div className="p-3 bg-err/5 border border-err/20 text-err text-sm rounded-xl">{noticeError}</div>}
              {noticeResult && (
                <div className="p-4 bg-ok/5 border border-ok/20 text-ok rounded-xl flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-sm">✅ Notice Created successfully!</p>
                    <p className="text-xs text-ink-2">{noticeResult.filename}</p>
                  </div>
                  <button onClick={() => noticeResult.pdf_id && downloadPDF(noticeResult.pdf_id, noticeResult.filename)} className="px-4 py-2 bg-ok hover:opacity-90 text-white font-semibold rounded-xl flex items-center gap-2 transition text-sm cursor-pointer">
                    <Download className="w-4 h-4" /> Download PDF
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Evidence Checklist View */}
          {activeView === 'evidence' && (
            <div className="p-6 md:p-8 max-w-3xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-ink mb-1">📋 Evidence Checklist</h2>
                <p className="text-ink-2 text-sm">Generate lists of legal documents and evidence required by target tribunals.</p>
              </div>

              <form onSubmit={handleFetchChecklist} className="flex gap-4">
                <select className="flex-1 ui-input" value={evidenceDomain} onChange={(e) => setEvidenceDomain(e.target.value)}>
                  <option value="consumer">Consumer Court</option>
                  <option value="labour">Labour Tribunal</option>
                  <option value="rent">Rent Control Board</option>
                  <option value="rti">RTI Appeals Commission</option>
                  <option value="criminal">Criminal Court</option>
                  <option value="cyber">Cyber Cell</option>
                </select>
                <button type="submit" disabled={evidenceLoading} className="px-6 py-3 bg-btn hover:bg-btn-hover disabled:opacity-50 text-white font-semibold rounded-xl transition cursor-pointer">
                  {evidenceLoading ? 'Fetching...' : 'Get checklist'}
                </button>
              </form>

              {evidenceError && <div className="p-3 bg-err/5 border border-err/20 text-err text-sm rounded-xl">{evidenceError}</div>}
              {evidenceChecklist && (
                <div className="ui-panel p-6 space-y-4">
                  <div className="p-3 bg-warn/5 border border-warn/20 text-warn rounded-xl text-sm italic font-medium">
                    ⚠️ {evidenceChecklist.instruction}
                  </div>
                  <div className="space-y-3">
                    {evidenceChecklist.items.map((item: string, i: number) => (
                      <label key={i} className="flex items-center gap-3 p-3 rounded-xl bg-bg hover:bg-hover border border-line cursor-pointer text-ink-2 text-sm transition-colors">
                        <input type="checkbox" className="w-5 h-5 rounded border-line text-accent focus:ring-accent bg-card" />
                        <span>{item}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* RTI View */}
          {activeView === 'rti' && (
            <div className="p-6 md:p-8 max-w-3xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-ink mb-1">📝 RTI Application Draft</h2>
                <p className="text-ink-2 text-sm">Create and print Right to Information applications targeting PIO officers.</p>
              </div>

              <form onSubmit={handleRtiSubmit} className="ui-panel p-6 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="ui-label">Applicant Name</label>
                    <input type="text" className="ui-input" value={rtiData.applicant_name} onChange={(e) => setRtiData({ ...rtiData, applicant_name: e.target.value })} required />
                  </div>
                  <div>
                    <label className="ui-label">Email</label>
                    <input type="email" className="ui-input" value={rtiData.applicant_email} onChange={(e) => setRtiData({ ...rtiData, applicant_email: e.target.value })} required />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="ui-label">Phone</label>
                    <input type="text" className="ui-input" value={rtiData.applicant_phone} onChange={(e) => setRtiData({ ...rtiData, applicant_phone: e.target.value })} required />
                  </div>
                  <div>
                    <label className="ui-label">Address</label>
                    <textarea rows={2} className="ui-input resize-none" value={rtiData.applicant_address} onChange={(e) => setRtiData({ ...rtiData, applicant_address: e.target.value })} required />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="ui-label">PIO Office Name</label>
                    <input type="text" className="ui-input" value={rtiData.pio_office} onChange={(e) => setRtiData({ ...rtiData, pio_office: e.target.value })} required />
                  </div>
                  <div>
                    <label className="ui-label">PIO Address</label>
                    <textarea rows={2} className="ui-input resize-none" value={rtiData.pio_address} onChange={(e) => setRtiData({ ...rtiData, pio_address: e.target.value })} required />
                  </div>
                </div>

                <div>
                  <label className="ui-label">Information Sought</label>
                  <textarea rows={3} className="ui-input" placeholder="State clearly what information you seek..." value={rtiData.information_sought} onChange={(e) => setRtiData({ ...rtiData, information_sought: e.target.value })} required />
                </div>

                <button type="submit" disabled={rtiLoading} className="ui-btn-primary">
                  {rtiLoading ? 'Compiling Draft...' : 'Generate RTI draft'}
                </button>
              </form>

              {rtiError && <div className="p-3 bg-err/5 border border-err/20 text-err text-sm rounded-xl">{rtiError}</div>}
              {rtiResult && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-ink-2">RTI Draft Preview</h4>
                  <textarea readOnly value={rtiResult.application} className="w-full h-80 p-4 bg-card border border-line rounded-xl text-ink font-mono text-sm leading-relaxed outline-none shadow-card" />
                  <p className="text-xs text-ink-3">💡 Send the draft copy with a court stamp/postal order of Rs. 10 to PIO.</p>
                </div>
              )}
            </div>
          )}

          {/* Case Strategy View */}
          {activeView === 'strategy' && (
            <div className="p-6 md:p-8 max-w-3xl mx-auto w-full space-y-6 overflow-y-auto">
              <div>
                <h2 className="text-2xl font-bold text-ink mb-1">🧭 Case Strategy</h2>
                <p className="text-ink-2 text-sm">Step-by-step plan: legal route, forums, evidence, deadlines and an estimated compensation range.</p>
              </div>

              <form onSubmit={handleStrategySubmit} className="ui-panel p-6 space-y-4">
                <div>
                  <label className="ui-label">Describe your situation</label>
                  <textarea
                    rows={4}
                    className="ui-input resize-none"
                    placeholder="e.g. My landlord is not returning my Rs 50,000 deposit since I vacated on 01-08-2024..."
                    value={strategyDescription}
                    onChange={(e) => setStrategyDescription(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label className="ui-label">Legal area (optional)</label>
                  <select className="ui-input" value={strategyDomain} onChange={(e) => setStrategyDomain(e.target.value)}>
                    <option value="auto">Auto-detect</option>
                    <option value="rent">Rent / Landlord</option>
                    <option value="consumer">Consumer</option>
                    <option value="labor">Labor / Salary</option>
                    <option value="criminal">Criminal</option>
                    <option value="cyber">Cyber</option>
                    <option value="defamation">Defamation</option>
                    <option value="family">Family</option>
                    <option value="commercial">Commercial / Contract</option>
                    <option value="civil">Civil</option>
                  </select>
                </div>
                <button type="submit" disabled={strategyLoading} className="ui-btn-primary">
                  {strategyLoading ? <Loader className="w-5 h-5 animate-spin" /> : 'Build strategy'}
                </button>
              </form>

              {strategyError && <div className="p-3 bg-err/5 border border-err/20 text-err text-sm rounded-xl">{strategyError}</div>}

              {strategyViewData && (
                <div className="ui-panel p-6 space-y-5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-accent bg-active px-2.5 py-1 rounded-lg border border-line">
                      {localizeDomain('en', strategyViewData.domain)}
                    </span>
                  </div>

                  <p className="text-sm text-ink leading-relaxed">{strategyViewData.summary}</p>

                  <div className="rounded-xl bg-accent/5 border border-accent/20 p-4">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-ink-3 mb-1">Estimated compensation range</p>
                    <p className="text-xl font-bold text-accent">
                      {inr(strategyViewData.compensation_estimate.min_amount)} – {inr(strategyViewData.compensation_estimate.max_amount)}
                      <span className="text-xs font-semibold text-ink-3"> ({strategyViewData.compensation_estimate.currency})</span>
                    </p>
                    <p className="text-xs text-ink-3 mt-1">{strategyViewData.compensation_estimate.basis}</p>
                    {strategyViewData.compensation_estimate.notes?.map((n, i) => (
                      <p key={i} className="text-[11px] text-ink-2 mt-1">• {n}</p>
                    ))}
                  </div>

                  {strategyViewData.assessment && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {strategyViewData.assessment.strengths.length > 0 && (
                        <div className="rounded-xl bg-ok/5 border border-ok/20 p-3.5">
                          <p className="text-[10px] font-bold uppercase tracking-wider text-ok mb-1.5">Strengths</p>
                          <ul className="space-y-1">
                            {strategyViewData.assessment.strengths.map((s, i) => (
                              <li key={i} className="flex items-start gap-2 text-xs text-ink-2">
                                <CheckCircle2 className="w-3.5 h-3.5 text-ok flex-shrink-0 mt-0.5" /> {s}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {strategyViewData.assessment.weaknesses.length > 0 && (
                        <div className="rounded-xl bg-warn/5 border border-warn/20 p-3.5">
                          <p className="text-[10px] font-bold uppercase tracking-wider text-warn mb-1.5">Watch out</p>
                          <ul className="space-y-1">
                            {strategyViewData.assessment.weaknesses.map((w, i) => (
                              <li key={i} className="flex items-start gap-2 text-xs text-ink-2">
                                <AlertCircle className="w-3.5 h-3.5 text-warn flex-shrink-0 mt-0.5" /> {w}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-wider text-ink-3 mb-1.5">Where to go</p>
                    <ul className="space-y-1">
                      {strategyViewData.legal_route.forums.map((f, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs text-ink-2">
                          <Scale className="w-3.5 h-3.5 text-accent flex-shrink-0 mt-0.5" /> {f}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {strategyViewData.deadline && (
                    <div className="rounded-xl bg-card border border-line p-3.5">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-ink-3 mb-1">Filing deadline</p>
                      <p className="text-xs text-ink font-semibold">{strategyViewData.deadline.deadline_date}</p>
                      <p className="text-[11px] text-ink-2 mt-0.5">{strategyViewData.deadline.description}</p>
                      <p className="text-[11px] text-ink-3 mt-0.5">{strategyViewData.deadline.status}</p>
                    </div>
                  )}

                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-wider text-ink-3 mb-1.5">Action plan</p>
                    <ol className="space-y-1.5">
                      {strategyViewData.action_plan.map((a, i) => (
                        <li key={i} className="flex items-start gap-2.5 text-xs text-ink-2">
                          <span className="w-5 h-5 rounded-full bg-active text-accent text-[10px] font-bold flex items-center justify-center flex-shrink-0">{i + 1}</span>
                          {a}
                        </li>
                      ))}
                    </ol>
                  </div>

                  {strategyViewData.evidence_checklist?.length > 0 && (
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-wider text-ink-3 mb-1.5">Evidence checklist</p>
                      <div className="flex flex-wrap gap-1.5">
                        {strategyViewData.evidence_checklist.map((item, i) => (
                          <span key={i} className="text-[11px] text-ink-2 bg-active border border-line rounded-lg px-2 py-1">{item}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {strategyViewData.disclaimer && (
                    <p className="text-[10px] text-ink-3 italic border-t border-line-2 pt-3">⚠️ {strategyViewData.disclaimer}</p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Document Audit View */}
          {activeView === 'audit' && (
            <div className="p-6 md:p-8 max-w-3xl mx-auto w-full space-y-6 overflow-y-auto">
              <div>
                <h2 className="text-2xl font-bold text-ink mb-1">🔍 Document Audit</h2>
                <p className="text-ink-2 text-sm">Paste a document (rent agreement, appointment letter, contract) to check required clauses.</p>
              </div>

              <form onSubmit={handleAuditSubmit} className="ui-panel p-6 space-y-4">
                <div>
                  <label className="ui-label">Document text</label>
                  <textarea
                    rows={8}
                    className="ui-input resize-none font-mono text-xs leading-relaxed"
                    placeholder="Paste the full document text here..."
                    value={auditText}
                    onChange={(e) => setAuditText(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label className="ui-label">Document type</label>
                  <select className="ui-input" value={auditDomain} onChange={(e) => setAuditDomain(e.target.value)}>
                    <option value="rent">Rent Agreement</option>
                    <option value="labor">Appointment / Employment Letter</option>
                    <option value="consumer">Consumer / Purchase</option>
                    <option value="commercial">Contract / Commercial</option>
                    <option value="criminal">Complaint / Affidavit</option>
                    <option value="civil">Other / Civil</option>
                  </select>
                </div>
                <button type="submit" disabled={auditLoading} className="ui-btn-primary">
                  {auditLoading ? <Loader className="w-5 h-5 animate-spin" /> : 'Audit document'}
                </button>
              </form>

              {auditError && <div className="p-3 bg-err/5 border border-err/20 text-err text-sm rounded-xl">{auditError}</div>}

              {auditViewData && (
                <div className="ui-panel p-6 space-y-5">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-bold text-ink">{auditViewData.document_type}</p>
                      <p className="text-xs text-ink-3 mt-0.5">{auditViewData.present_count} of {auditViewData.total_checks} required clauses found</p>
                    </div>
                    <span className={`px-3 py-1.5 rounded-xl text-xs font-bold border ${
                      auditViewData.risk === 'LOW' ? 'bg-ok/10 text-ok border-ok/25' : auditViewData.risk === 'HIGH' ? 'bg-err/10 text-err border-err/25' : 'bg-warn/10 text-warn border-warn/25'
                    }`}>
                      {auditViewData.risk} RISK
                    </span>
                  </div>

                  <div className="h-2 rounded-full bg-active border border-line overflow-hidden">
                    <div className={`h-full rounded-full transition-all ${auditViewData.score >= 80 ? 'bg-ok' : auditViewData.score >= 50 ? 'bg-warn' : 'bg-err'}`} style={{ width: `${auditViewData.score}%` }} />
                  </div>
                  <p className="text-xs text-ink-2 -mt-2">Score: {auditViewData.score}%</p>

                  {auditViewData.audit_intro && <p className="text-xs text-ink-2">{auditViewData.audit_intro}</p>}

                  {auditViewData.issues.length > 0 && (
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-wider text-err mb-1.5">Missing clauses</p>
                      <ul className="space-y-1.5">
                        {auditViewData.issues.map((iss) => (
                          <li key={iss.id} className="flex items-start gap-2 text-xs text-ink-2 p-2.5 rounded-lg bg-err/5 border border-err/15">
                            <AlertCircle className="w-3.5 h-3.5 text-err flex-shrink-0 mt-0.5" />
                            <span className="min-w-0">
                              <span className="font-semibold text-ink">{iss.label}</span>
                              <span className="text-ink-3"> ({iss.severity})</span>
                              {iss.hint && <span className="block text-ink-3 mt-0.5">{iss.hint}</span>}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {auditViewData.present.length > 0 && (
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-wider text-ok mb-1.5">Present</p>
                      <div className="flex flex-wrap gap-1.5">
                        {auditViewData.present.map((p) => (
                          <span key={p.id} className="inline-flex items-center gap-1 text-[11px] text-ink-2 bg-ok/10 border border-ok/20 rounded-lg px-2 py-1">
                            <CheckCircle2 className="w-3 h-3 text-ok" /> {p.label}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {auditViewData.suggestions.length > 0 && (
                    <div className="rounded-xl bg-active border border-line p-3.5">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-ink-3 mb-1.5">Suggested additions</p>
                      <ul className="space-y-1">
                        {auditViewData.suggestions.map((s, i) => (
                          <li key={i} className="text-[11px] text-ink-2">• {s}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {auditViewData.disclaimer && (
                    <p className="text-[10px] text-ink-3 italic border-t border-line-2 pt-3">⚠️ {auditViewData.disclaimer}</p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Settings View (placeholder) */}
          {activeView === 'settings' && (
            <div className="p-6 md:p-8 max-w-2xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-ink mb-1">⚙️ Settings</h2>
                <p className="text-ink-2 text-sm">Account and preferences.</p>
              </div>

              <div className="ui-panel p-6 space-y-5">
                <div>
                  <h4 className="text-sm font-semibold text-ink mb-3">Profile</h4>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-full bg-active text-ink flex items-center justify-center text-lg font-bold border border-line select-none">
                      {auth.userName?.charAt(0).toUpperCase() || 'U'}
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-ink">{auth.userName || 'User'}</p>
                      {auth.email && <p className="text-xs text-ink-3">{auth.email}</p>}
                      <p className="text-[10px] text-ink-3 mt-0.5">Free plan</p>
                    </div>
                  </div>
                </div>

                <div className="border-t border-line-2 pt-5 space-y-3">
                  {[
                    { label: 'Answer language', hint: `Current: ${targetLanguage.toUpperCase()}`, onChange: () => { } },
                    { label: 'Email notifications', hint: 'Coming soon', onChange: () => { } },
                  ].map((row) => (
                    <div key={row.label} className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-ink">{row.label}</p>
                        <p className="text-xs text-ink-3">{row.hint}</p>
                      </div>
                      <span className="text-ink-3 text-xs">Coming soon</span>
                    </div>
                  ))}
                </div>

                <button onClick={auth.logout} className="ui-btn-primary bg-err hover:bg-err/90">
                  <LogOut className="w-4 h-4" /> Logout
                </button>
              </div>
            </div>
          )}

          {/* Deadline View removed */}
        </div>
      </div>

      {/* PDF Modal */}
      {showPDFModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowPDFModal(false)}>
          <div className="bg-card border border-line rounded-2xl max-w-2xl w-full max-h-[500px] overflow-y-auto p-6 shadow-pop" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-line-2">
              <h3 className="text-lg font-bold text-ink">Generated Legal Notice PDFs</h3>
              <button
                onClick={() => setShowPDFModal(false)}
                className="p-2 hover:bg-hover rounded-lg transition text-ink-3 hover:text-ink cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {userPDFs.length === 0 ? (
              <p className="text-ink-3 text-center py-6">No PDFs generated yet</p>
            ) : (
              <div className="space-y-3">
                {userPDFs.map((pdf, i) => (
                  <div key={i} className="p-4 rounded-xl bg-bg border border-line flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-ink text-sm truncate">{pdf.filename}</p>
                      <p className="text-xs text-ink-3 mt-1">🏷️ compensation claim: Rs. {pdf.demand_amount || 'N/A'}</p>
                    </div>
                    <button
                      onClick={() => downloadPDF(pdf._id, pdf.filename)}
                      className="px-3.5 py-2 rounded-lg bg-btn hover:bg-btn-hover text-white font-semibold flex items-center gap-2 transition text-sm cursor-pointer"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* My Documents Modal */}
      {showDocsModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowDocsModal(false)}>
          <div className="bg-card border border-line rounded-2xl max-w-2xl w-full max-h-[500px] overflow-y-auto p-6 shadow-pop" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-line-2">
              <h3 className="text-lg font-bold text-ink">My Documents</h3>
              <button
                onClick={() => setShowDocsModal(false)}
                className="p-2 hover:bg-hover rounded-lg transition text-ink-3 hover:text-ink cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <label className="flex items-center justify-center gap-2 w-full py-2.5 mb-4 rounded-xl bg-btn hover:bg-btn-hover text-white font-semibold text-sm cursor-pointer transition">
              <Paperclip className="w-4 h-4" />
              Upload a document
              <input
                type="file"
                accept=".pdf,.txt,.doc,.docx,.md"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    handleUpload(f).then(() => {
                      if (auth.userId) {
                        getUserDocuments(auth.userId).then(setUserDocuments).catch(console.error);
                      }
                    });
                  }
                  e.target.value = '';
                }}
              />
            </label>

            {userDocuments.length === 0 ? (
              <p className="text-ink-3 text-center py-6">No documents uploaded yet. Upload a PDF to search your own files in chat.</p>
            ) : (
              <div className="space-y-3">
                {userDocuments.map((doc, i) => (
                  <div key={i} className="p-4 rounded-xl bg-bg border border-line flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-ink text-sm truncate">{doc.filename}</p>
                      <p className="text-xs text-ink-3 mt-1">
                        {doc.chunk_count != null ? `${doc.chunk_count} chunks` : ''}
                        {doc.provider ? ` · ${doc.provider}` : ''}
                        {doc.created_at ? ` · ${new Date(doc.created_at).toLocaleDateString()}` : ''}
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        getUserDocuments(auth.userId || 'anonymous').then(setUserDocuments).catch(console.error);
                        showToast('Document list refreshed');
                      }}
                      className="px-3 py-1.5 rounded-lg bg-bg border border-line hover:bg-hover text-ink-2 text-xs font-semibold transition cursor-pointer"
                    >
                      Refresh
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Delete Chat Confirmation Modal */}
      {confirmDeleteId && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={closeChatMenu}
        >
          <div
            className="bg-card border border-line rounded-2xl max-w-md w-full p-7 text-center shadow-pop"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mx-auto w-14 h-14 rounded-full bg-err/10 flex items-center justify-center mb-4">
              <Trash2 className="w-7 h-7 text-err" />
            </div>
            <h3 className="text-xl font-bold text-ink mb-2">Delete this chat?</h3>
            <p className="text-sm text-ink-2 mb-6 leading-relaxed">
              <span className="text-ink font-semibold">
                {shortTitle(chatSessions.find((s) => s.id === confirmDeleteId)?.title || '')}
              </span>{' '}
              will be permanently deleted. This cannot be undone.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => deleteChat(confirmDeleteId)}
                className="flex-1 px-4 py-3 rounded-xl bg-err hover:bg-err/90 text-white font-semibold text-sm transition cursor-pointer"
              >
                Yes, delete
              </button>
              <button
                onClick={closeChatMenu}
                className="flex-1 px-4 py-3 rounded-xl bg-hover hover:bg-active text-ink font-semibold text-sm transition cursor-pointer"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[60]">
          <div className="bg-btn text-white text-xs font-medium px-4 py-2.5 rounded-xl shadow-pop flex items-center gap-2">
            {toast}
          </div>
        </div>
      )}
    </div>
  );
}

// ============ MAIN APP WITH PROVIDER ============

function KeyedAppContent() {
  const auth = React.useContext(AuthContext);
  return <AppContent key={auth?.userId ?? 'anon'} />;
}

export default function App() {
  return (
    <AuthProvider>
      <KeyedAppContent />
    </AuthProvider>
  );
}
