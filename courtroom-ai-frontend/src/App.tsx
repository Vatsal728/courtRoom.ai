import {
  AlertCircle,
  Check,
  ChevronDown,
  Clock,
  Download,
  FileText,
  Loader,
  LogOut,
  MessageSquare,
  Plus,
  Scale, Send,
  X,
  Zap
} from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import './index.css';
import {
  checkDeadline,
  downloadPDF,
  generatePDFNotice,
  generateRTIApplication,
  getEvidenceChecklist,
  getUserPDFs,
  healthCheck,
  submitQuery
} from './lib/api';
import { AuthContext, AuthProvider, useAuth } from './lib/auth';
import { detectLanguage } from './lib/language';

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
    <div className="h-screen w-screen overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl"></div>
      </div>

      <div className="w-full max-w-sm relative z-10">
        {/* Logo */}
        <div className={`text-center ${isRegister ? 'mb-3' : 'mb-6'}`}>
          <div className="flex items-center justify-center gap-3 mb-1">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center shadow-lg flex-shrink-0">
              <Scale className="w-5.5 h-5.5 text-slate-900" />
            </div>
            <h1 className="text-2xl font-bold text-white leading-none">
              court<span className="text-amber-400">Room</span>.ai
            </h1>
          </div>
          <p className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">AI Legal Assistant for India</p>
        </div>

        {/* Form Card */}
        <div className={`bg-slate-800/50 backdrop-blur-xl border border-slate-700 rounded-2xl shadow-2xl transition-all duration-300 ${isRegister ? 'p-5 space-y-3.5' : 'p-7 space-y-5'
          }`}>
          <div>
            <h2 className="text-xl font-bold text-white mb-1">
              {isRegister ? 'Create Account' : 'Welcome Back'}
            </h2>
            <p className="text-xs text-slate-400">
              {isRegister ? 'Join thousands using AI legal guidance' : 'Sign in to your account'}
            </p>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-300 text-xs">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className={isRegister ? 'space-y-3' : 'space-y-4'}>
            {isRegister && (
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">Full Name</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  className="w-full px-3.5 py-2.5 bg-slate-900/50 border border-slate-600 rounded-lg focus:border-amber-500 focus:bg-slate-900 outline-none text-white placeholder-slate-500 transition text-sm"
                  required
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full px-3.5 py-2.5 bg-slate-900/50 border border-slate-600 rounded-lg focus:border-amber-500 focus:bg-slate-900 outline-none text-white placeholder-slate-500 transition text-sm"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-3.5 py-2.5 bg-slate-900/50 border border-slate-600 rounded-lg focus:border-amber-500 focus:bg-slate-900 outline-none text-white placeholder-slate-500 transition text-sm"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-lg bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 disabled:opacity-50 text-slate-900 font-bold transition flex items-center justify-center gap-2 text-sm shadow-md"
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
              className="w-full text-xs text-slate-400 hover:text-white transition font-medium"
            >
              {isRegister ? 'Already have an account? Sign in' : "Don't have an account? Register"}
            </button>
          </form>
        </div>

        <p className={`text-center text-[10px] text-slate-500 ${isRegister ? 'mt-2' : 'mt-5'}`}>
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
  timestamp: Date;
}

// ============ MAIN APP CONTENT ============

function AppContent() {
  const auth = React.useContext(AuthContext);
  if (!auth) return null;

  const getGreeting = () => {
    const hr = new Date().getHours();
    const name = auth.userName ? auth.userName.split(' ')[0] : 'User';
    if (hr < 12) return `Morning, ${name}`;
    if (hr < 17) return `Afternoon, ${name}`;
    return `Evening, ${name}`;
  };

  const [activeView, setActiveView] = useState<'chat' | 'notice' | 'evidence' | 'rti' | 'deadline'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showPDFModal, setShowPDFModal] = useState(false);
  const [userPDFs, setUserPDFs] = useState<any[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('cached_user_pdfs') || '[]');
    } catch {
      return [];
    }
  });
  const [backendStatus, setBackendStatus] = useState<string>('Checking...');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);

  // Freemium Gate
  const [freeQueriesCount, setFreeQueriesCount] = useState<number>(() => {
    return Number(localStorage.getItem('free_queries_count') || '0');
  });

  // Notice Form states
  const [noticeData, setNoticeData] = useState({
    sender_name: '',
    sender_address: '',
    recipient_name: '',
    recipient_address: '',
    issue_type: 'Defective Product',
    issue_description: '',
    applicable_section: '',
    demand_amount: ''
  });
  const [noticeLoading, setNoticeLoading] = useState(false);
  const [noticeResult, setNoticeResult] = useState<any>(null);
  const [noticeError, setNoticeError] = useState('');

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

  // Deadline states
  const [deadlineData, setDeadlineData] = useState({
    case_type: 'consumer',
    incident_date: ''
  });
  const [deadlineLoading, setDeadlineLoading] = useState(false);
  const [deadlineResult, setDeadlineResult] = useState<any>(null);
  const [deadlineError, setDeadlineError] = useState('');

  // Fetch backend health and user PDFs
  useEffect(() => {
    healthCheck()
      .then((data) => {
        setBackendStatus(`🟢 Connected (Service: ${data.service})`);
      })
      .catch(() => {
        setBackendStatus('🔴 Offline (FastAPI backend disconnected)');
      });

    if (auth.userId) {
      getUserPDFs(auth.userId)
        .then((pdfs) => {
          setUserPDFs(pdfs);
          localStorage.setItem('cached_user_pdfs', JSON.stringify(pdfs));
        })
        .catch(console.error);
    }
  }, [auth.userId]);

  // Auto scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim()) return;

    // Freemium restriction check
    if (!auth.isLoggedIn && freeQueriesCount >= 1) {
      alert("🔒 Free query limit reached. Please register or sign in to proceed.");
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const detectedLang = detectLanguage(inputValue);
      const activeUserId = auth.userId || 'anonymous';
      const result = await submitQuery(inputValue, activeUserId, detectedLang);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: result.response,
        results: result.sources,
        domain: result.domain,
        confidence: result.confidence,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (!auth.isLoggedIn) {
        const nextCount = freeQueriesCount + 1;
        setFreeQueriesCount(nextCount);
        localStorage.setItem('free_queries_count', String(nextCount));
      }

      if (result.query_id) {
        localStorage.setItem('lastQueryId', result.query_id);
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `❌ Error: ${error instanceof Error ? error.message : 'Failed to process query'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  }, [inputValue, auth.userId, freeQueriesCount, auth.isLoggedIn]);

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

  // Deadline Form Submit
  const handleDeadlineSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setDeadlineLoading(true);
    setDeadlineError('');
    setDeadlineResult(null);
    try {
      const parts = deadlineData.incident_date.split('-');
      const formattedDate = `${parts[2]}-${parts[1]}-${parts[0]}`;
      const data = await checkDeadline(deadlineData.case_type, formattedDate);
      setDeadlineResult(data);
    } catch (err: any) {
      setDeadlineError(err.message || 'Error calculating deadline');
    } finally {
      setDeadlineLoading(false);
    }
  };

  if (!auth.isLoggedIn) {
    return <LoginPage />;
  }

  return (
    <div className="h-screen bg-slate-950 flex overflow-hidden text-slate-100 font-sans">
      {/* Sidebar */}
      <div
        className={`${sidebarOpen ? 'w-64' : 'w-0'
          } bg-[#0b0f19] border-r border-[#21376d]/30 flex flex-col transition-all duration-300 overflow-hidden z-20`}
      >
        <div className="p-4 border-b border-[#21376d]/20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Scale className="w-5 h-5 text-amber-500" />
            <span className="text-base font-bold text-white">court<span className="text-amber-400">Room</span>.ai</span>
          </div>

          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1.5 rounded-lg border border-slate-700 bg-slate-900/60 text-slate-400 hover:text-white hover:border-[#21376d] focus:outline-none focus:ring-2 focus:ring-[#21376d]/50 transition flex items-center justify-center cursor-pointer"
            title="Close sidebar"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>
        </div>

        {/* Sidebar Navigation */}
        <div className="p-4 space-y-1.5">
          <button
            onClick={() => { setActiveView('chat'); }}
            className={`relative w-full flex items-center gap-3 pl-5 pr-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${activeView === 'chat' ? 'bg-[#21376d]/20 text-amber-400 font-semibold shadow-sm' : 'text-slate-350 hover:bg-[#21376d]/10 hover:text-white'
              }`}
          >
            {activeView === 'chat' && (
              <div className="absolute left-0 top-2 bottom-2 w-[3px] bg-amber-500 rounded-r"></div>
            )}
            <MessageSquare className={`w-4 h-4 ${activeView === 'chat' ? 'text-amber-400' : 'text-slate-400'}`} />
            AI Legal Advisor
          </button>
          <button
            onClick={() => { setActiveView('notice'); }}
            className={`relative w-full flex items-center gap-3 pl-5 pr-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${activeView === 'notice' ? 'bg-[#21376d]/20 text-amber-400 font-semibold shadow-sm' : 'text-slate-350 hover:bg-[#21376d]/10 hover:text-white'
              }`}
          >
            {activeView === 'notice' && (
              <div className="absolute left-0 top-2 bottom-2 w-[3px] bg-amber-500 rounded-r"></div>
            )}
            <FileText className={`w-4 h-4 ${activeView === 'notice' ? 'text-amber-400' : 'text-slate-400'}`} />
            Notice Generator
          </button>
          <button
            onClick={() => { setActiveView('evidence'); }}
            className={`relative w-full flex items-center gap-3 pl-5 pr-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${activeView === 'evidence' ? 'bg-[#21376d]/20 text-amber-400 font-semibold shadow-sm' : 'text-slate-350 hover:bg-[#21376d]/10 hover:text-white'
              }`}
          >
            {activeView === 'evidence' && (
              <div className="absolute left-0 top-2 bottom-2 w-[3px] bg-amber-500 rounded-r"></div>
            )}
            <Check className={`w-4 h-4 ${activeView === 'evidence' ? 'text-amber-400' : 'text-slate-400'}`} />
            Evidence Checklist
          </button>
          <button
            onClick={() => { setActiveView('rti'); }}
            className={`relative w-full flex items-center gap-3 pl-5 pr-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${activeView === 'rti' ? 'bg-[#21376d]/20 text-amber-400 font-semibold shadow-sm' : 'text-slate-350 hover:bg-[#21376d]/10 hover:text-white'
              }`}
          >
            {activeView === 'rti' && (
              <div className="absolute left-0 top-2 bottom-2 w-[3px] bg-amber-500 rounded-r"></div>
            )}
            <Zap className={`w-4 h-4 ${activeView === 'rti' ? 'text-amber-400' : 'text-slate-400'}`} />
            RTI Application
          </button>
          <button
            onClick={() => { setActiveView('deadline'); }}
            className={`relative w-full flex items-center gap-3 pl-5 pr-4 py-2.5 rounded-lg text-sm font-medium transition cursor-pointer ${activeView === 'deadline' ? 'bg-[#21376d]/20 text-amber-400 font-semibold shadow-sm' : 'text-slate-350 hover:bg-[#21376d]/10 hover:text-white'
              }`}
          >
            {activeView === 'deadline' && (
              <div className="absolute left-0 top-2 bottom-2 w-[3px] bg-amber-500 rounded-r"></div>
            )}
            <Clock className={`w-4 h-4 ${activeView === 'deadline' ? 'text-amber-400' : 'text-slate-400'}`} />
            Deadline Tracker
          </button>
        </div>

        {/* Sidebar Content (PDF List Preview) */}
        <div className="flex-1 overflow-y-auto p-4 border-t border-[#21376d]/20">
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-2 py-2 mb-2">My Generated PDFs</div>
          <div className="space-y-1">
            {userPDFs.slice(0, 5).map((pdf, idx) => (
              <button
                key={idx}
                onClick={() => downloadPDF(pdf._id, pdf.filename)}
                className="w-full flex items-center justify-between text-left p-2 rounded hover:bg-slate-700/50 transition text-slate-300 text-xs truncate"
              >
                <span className="truncate flex-1">{pdf.filename}</span>
                <Download className="w-3. h-3 text-amber-500 ml-2 flex-shrink-0" />
              </button>
            ))}
            {userPDFs.length > 5 && (
              <button onClick={() => setShowPDFModal(true)} className="text-[11px] text-amber-400 hover:underline px-2 py-1">
                View all {userPDFs.length} documents...
              </button>
            )}
          </div>
        </div>

        {/* Sidebar Footer - Profile Card */}
        <div className="border-t border-[#21376d]/20 p-3.5 relative">
          {/* Dropdown Popover */}
          {profileMenuOpen && (
            <div className="absolute bottom-16 left-3.5 right-3.5 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl p-2.5 z-30 space-y-1">
              <button
                onClick={() => {
                  setShowPDFModal(true);
                  setProfileMenuOpen(false);
                }}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-slate-300 hover:bg-[#21376d]/20 transition text-xs font-semibold cursor-pointer"
              >
                <Download className="w-4 h-4 text-amber-500" />
                All PDFs ({userPDFs.length})
              </button>
              <button
                onClick={auth.logout}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-red-400 hover:bg-red-500/10 transition text-xs font-semibold cursor-pointer"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          )}

          {/* Profile Row */}
          <div className="flex items-center justify-between bg-slate-950 border border-[#21376d]/10 rounded-xl p-2 hover:bg-slate-900/60 transition">
            <div className="flex items-center gap-2.5 truncate">
              {/* Avatar circle */}
              <div className="w-9 h-9 rounded-full bg-[#e6dfd5] text-slate-900 flex items-center justify-center font-bold text-sm shadow-inner flex-shrink-0 select-none">
                {auth.userName?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="text-left truncate">
                <p className="text-xs font-bold text-white leading-tight truncate">{auth.userName || 'User'}</p>
                <p className="text-[10px] text-slate-400 leading-normal mt-0.5">Free plan</p>
              </div>
            </div>

            {/* Action buttons on the right */}
            <div className="flex items-center gap-1">
              <button
                onClick={() => setShowPDFModal(true)}
                className="p-1.5 hover:bg-slate-800 rounded-md text-slate-400 hover:text-white transition cursor-pointer border border-slate-800/80 bg-slate-900"
                title="Generated PDFs"
              >
                <Download className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setProfileMenuOpen(!profileMenuOpen)}
                className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition cursor-pointer"
              >
                <ChevronDown className={`w-3.5 h-3.5 transform transition-transform duration-200 ${profileMenuOpen ? 'rotate-180' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Container */}
      <div className="flex-1 flex flex-col overflow-hidden bg-slate-950">
        {/* Top Header */}
        <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 py-3 flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1.5 rounded-lg border border-slate-700 bg-slate-900/60 text-slate-400 hover:text-white hover:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition flex items-center justify-center cursor-pointer"
                title="Open sidebar"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
              </button>
            )}
            {!sidebarOpen && (
              <div className="flex items-center gap-2">
                <Scale className="w-5 h-5 text-amber-500" />
                <span className="font-bold text-white text-base">court<span className="text-amber-400">Room</span>.ai</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <p className="text-xs font-bold text-white">{auth.userName || 'User'}</p>
              {auth.email && <p className="text-[10px] text-slate-400">{auth.email}</p>}
            </div>
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center text-slate-950 text-xs font-extrabold shadow-md">
              {auth.userName?.charAt(0).toUpperCase() || 'U'}
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
                <div className="h-full flex flex-col justify-between items-center pt-10 px-6 pb-0 bg-slate-950 overflow-hidden select-none">
                  {/* Top Spacer */}
                  <div className="h-2"></div>

                  {/* Centered Search Layout */}
                  <div className="max-w-xl w-full flex flex-col items-center flex-1 justify-center">
                    {/* Brand Greeting with courtRoom.ai Logo */}
                    <div className="flex items-center gap-3 text-3xl font-normal text-slate-100 mb-6 tracking-tight">
                      <Scale className="w-8 h-8 text-amber-500 animate-pulse" />
                      <span>{getGreeting()}</span>
                    </div>

                    {/* Unified Search Input Card */}
                    <div className="w-full bg-slate-900 border border-slate-800 rounded-2xl px-4.5 py-3 shadow-2xl relative focus-within:border-amber-500/80 focus-within:ring-1 focus-within:ring-amber-500/25 transition-all">
                      <textarea
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        placeholder="Ask courtRoom.ai anything about Indian laws..."
                        className="w-full bg-transparent border-0 outline-none focus:ring-0 text-white placeholder-slate-500 text-sm md:text-[15px] resize-none min-h-[48px] focus:outline-none focus:border-0 leading-relaxed pr-2"
                      />

                      {/* Search Card Footer Row */}
                      <div className="flex items-center justify-between pt-2">
                        {/* Plus button (Left side) */}
                        <button
                          onClick={() => setActiveView('notice')}
                          className="-ml-1.5 p-1.5 hover:bg-slate-850 rounded-lg transition text-slate-400 hover:text-white cursor-pointer"
                          title="Draft New Document"
                        >
                          <Plus className="w-5 h-5" />
                        </button>

                        {/* Controls (Right side) */}
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800 font-mono">
                            Qwen 2.5 Law
                          </span>
                          <button
                            onClick={handleSendMessage}
                            disabled={loading || !inputValue.trim()}
                            className="p-1.5 rounded-full bg-amber-500 hover:bg-amber-400 text-slate-950 transition disabled:opacity-30 disabled:hover:bg-amber-500 flex items-center justify-center cursor-pointer"
                          >
                            <Send className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>



                    {/* Category Selection Filter Chips */}
                    <div className="flex flex-wrap gap-2 justify-center w-full pt-8">
                      {[
                        { label: '📄 Notice Generator', view: 'notice' },
                        { label: '📋 Evidence Checklist', view: 'evidence' },
                        { label: '📝 RTI Builder', view: 'rti' },
                        { label: '⏰ Deadline Tracker', view: 'deadline' }
                      ].map((item, idx) => (
                        <button
                          key={idx}
                          onClick={() => setActiveView(item.view as any)}
                          className="px-3.5 py-1.5 bg-slate-900 hover:bg-slate-850 border border-slate-800 hover:border-amber-500/30 rounded-full transition text-[11px] font-medium text-slate-300 flex items-center gap-1.5 cursor-pointer shadow-sm"
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Pinned Disclaimer Footer */}
                  <p className="text-[10px] text-slate-500 text-center mt-0.5 pb-0.5">
                    ⚖️ Powered by local Qwen RAG pipeline • Multi-language enabled • Verified Indian Law Statutes
                  </p>
                </div>
              ) : (
                /* Results / Messages State (Scrollable) */
                <div className="flex-1 flex flex-col overflow-hidden bg-slate-950">
                  {/* Messages list */}
                  <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl mx-auto w-full">
                    {messages.map((msg) => (
                      <div key={msg.id} className={`flex gap-4 ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                        {msg.type === 'assistant' && (
                          <div className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center flex-shrink-0 shadow-md">
                            <Scale className="w-4.5 h-4.5 text-amber-400" />
                          </div>
                        )}

                        <div
                          className={`max-w-2xl rounded-2xl px-5 py-3.5 shadow-md ${msg.type === 'user'
                            ? 'bg-amber-500 text-slate-950 font-medium'
                            : 'bg-slate-800/80 border border-slate-700 text-slate-100'
                            }`}
                        >
                          <p className="text-sm md:text-[15px] whitespace-pre-wrap leading-relaxed">{msg.content}</p>

                          {msg.type === 'assistant' && msg.domain && (
                            <div className="mt-3 flex items-center gap-4 text-xs font-semibold text-amber-400/90 bg-amber-500/5 p-2 rounded-lg border border-amber-500/10">
                              <span>📂 DOMAIN: {msg.domain.toUpperCase()}</span>
                              {msg.confidence !== undefined && (
                                <span>🎯 CONFIDENCE: {(msg.confidence * 100).toFixed(0)}%</span>
                              )}
                            </div>
                          )}

                          {msg.type === 'assistant' && msg.results && msg.results.length > 0 && (
                            <div className="mt-4 space-y-2 border-t border-slate-700 pt-3">
                              <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Applicable Reference Laws</p>
                              <div className="grid grid-cols-1 gap-2">
                                {msg.results.map((result, idx) => (
                                  <div key={idx} className="p-3 rounded bg-slate-900/60 border border-slate-700/50 text-xs">
                                    <div className="flex justify-between items-start mb-1">
                                      <div className="font-semibold text-amber-400">{result.section}</div>
                                      <div className="text-[10px] text-slate-400 font-semibold uppercase">{result.source_act}</div>
                                    </div>
                                    <div className="text-slate-300 mb-1"><strong>Topic:</strong> {result.topic}</div>
                                    <div className="text-slate-400 italic font-mono text-[11px]">"{result.content_preview}"</div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {msg.type === 'user' && (
                          <div className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center flex-shrink-0 shadow-md">
                            <span className="text-amber-400 text-sm font-bold">{auth.userName?.charAt(0).toUpperCase()}</span>
                          </div>
                        )}
                      </div>
                    ))}

                    {loading && (
                      <div className="flex gap-4 justify-start">
                        <div className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center flex-shrink-0">
                          <Scale className="w-4.5 h-4.5 text-amber-400 animate-pulse" />
                        </div>
                        <div className="bg-slate-800/80 border border-slate-700 rounded-2xl px-5 py-4">
                          <div className="flex gap-2">
                            <div className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-bounce"></div>
                            <div className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-bounce [animation-delay:0.2s]"></div>
                            <div className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-bounce [animation-delay:0.4s]"></div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>

                  {/* Search Bar under messages directly on main display background, bg-transparent, no borders */}
                  <div className="p-4 md:p-6 max-w-4xl mx-auto w-full bg-transparent">
                    <div className="relative flex items-center bg-slate-900 border border-slate-850 focus-within:border-amber-500 focus-within:ring-1 focus-within:ring-amber-500/20 rounded-full px-5 py-2.5 transition-all shadow-lg">
                      <textarea
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        placeholder="Describe your legal issue... (Enter to send, Shift+Enter for new line)"
                        className="flex-1 bg-transparent border-0 outline-none focus:ring-0 text-white placeholder-slate-500 text-sm md:text-base resize-none pr-12 pl-1 py-1 leading-relaxed focus:outline-none focus:border-0"
                        rows={1}
                      />
                      <button
                        onClick={handleSendMessage}
                        disabled={loading || !inputValue.trim()}
                        className="absolute right-2.5 p-2 rounded-full bg-amber-500 hover:bg-amber-400 text-slate-950 transition disabled:opacity-30 disabled:hover:bg-amber-500 flex items-center justify-center flex-shrink-0 cursor-pointer"
                      >
                        {loading ? (
                          <Loader className="w-4 h-4 animate-spin" />
                        ) : (
                          <Send className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Notice Generator View */}
          {activeView === 'notice' && (
            <div className="p-6 md:p-8 max-w-4xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-white mb-1">📄 Notice Generator</h2>
                <p className="text-slate-400 text-sm">Create and download customized ReportLab PDF notices with custom names.</p>
              </div>

              <form onSubmit={handleNoticeSubmit} className="bg-slate-800/40 border border-slate-700/60 p-6 rounded-2xl space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Sender Name</label>
                    <input type="text" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={noticeData.sender_name} onChange={(e) => setNoticeData({ ...noticeData, sender_name: e.target.value })} required />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Recipient Name</label>
                    <input type="text" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={noticeData.recipient_name} onChange={(e) => setNoticeData({ ...noticeData, recipient_name: e.target.value })} required />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Sender Address</label>
                    <textarea rows={2} className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition resize-none" value={noticeData.sender_address} onChange={(e) => setNoticeData({ ...noticeData, sender_address: e.target.value })} required />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Recipient Address</label>
                    <textarea rows={2} className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition resize-none" value={noticeData.recipient_address} onChange={(e) => setNoticeData({ ...noticeData, recipient_address: e.target.value })} required />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Issue Type</label>
                    <select className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={noticeData.issue_type} onChange={(e) => setNoticeData({ ...noticeData, issue_type: e.target.value })}>
                      <option>Defective Product</option>
                      <option>Salary Delay</option>
                      <option>Illegal Eviction</option>
                      <option>Other Breach of Contract</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Claim Amount (Rs.)</label>
                    <input type="text" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={noticeData.demand_amount} onChange={(e) => setNoticeData({ ...noticeData, demand_amount: e.target.value })} required />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Detailed Description</label>
                  <textarea rows={3} className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={noticeData.issue_description} onChange={(e) => setNoticeData({ ...noticeData, issue_description: e.target.value })} required />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Applicable Law Section</label>
                  <input type="text" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" placeholder="e.g. Section 12 of Consumer Protection Act" value={noticeData.applicable_section} onChange={(e) => setNoticeData({ ...noticeData, applicable_section: e.target.value })} required />
                </div>

                <button type="submit" disabled={noticeLoading} className="w-full py-3 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 disabled:opacity-50 text-slate-900 font-bold transition flex items-center justify-center gap-2">
                  {noticeLoading ? <Loader className="w-5 h-5 animate-spin" /> : 'Compile Legal Notice PDF'}
                </button>
              </form>

              {noticeError && <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-300 text-sm rounded-lg">{noticeError}</div>}
              {noticeResult && (
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 rounded-xl flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-sm">✅ Notice Created successfully!</p>
                    <p className="text-xs text-slate-400">{noticeResult.filename}</p>
                  </div>
                  <button onClick={handleDownloadNotice} className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold rounded-lg flex items-center gap-2 transition text-sm">
                    <Download className="w-4 h-4" /> Download PDF
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Evidence Checklist View */}
          {activeView === 'evidence' && (
            <div className="p-6 md:p-8 max-w-4xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-white mb-1">📋 Evidence Checklist</h2>
                <p className="text-slate-400 text-sm">Generate lists of legal documents and evidence required by target tribunals.</p>
              </div>

              <form onSubmit={handleFetchChecklist} className="flex gap-4">
                <select className="flex-1 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-white outline-none focus:border-amber-500" value={evidenceDomain} onChange={(e) => setEvidenceDomain(e.target.value)}>
                  <option value="consumer">Consumer Court</option>
                  <option value="labour">Labour Tribunal</option>
                  <option value="rent">Rent Control Board</option>
                  <option value="rti">RTI Appeals Commission</option>
                  <option value="criminal">Criminal Court</option>
                  <option value="cyber">Cyber Cell</option>
                </select>
                <button type="submit" disabled={evidenceLoading} className="px-6 py-3 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-bold rounded-xl transition">
                  {evidenceLoading ? 'Fetching...' : 'Get checklist'}
                </button>
              </form>

              {evidenceError && <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-300 text-sm rounded-lg">{evidenceError}</div>}
              {evidenceChecklist && (
                <div className="bg-slate-800/50 border border-slate-700 p-6 rounded-2xl space-y-4">
                  <div className="p-3 bg-amber-500/5 border border-amber-500/20 text-amber-400 rounded-lg text-sm italic font-medium">
                    ⚠️ {evidenceChecklist.instruction}
                  </div>
                  <div className="space-y-3">
                    {evidenceChecklist.items.map((item: string, i: number) => (
                      <label key={i} className="flex items-center gap-3 p-3 rounded-lg bg-slate-900/30 hover:bg-slate-900/60 border border-slate-850 cursor-pointer text-slate-200 text-sm">
                        <input type="checkbox" className="w-5 h-5 rounded border-slate-600 text-amber-500 focus:ring-amber-500 bg-slate-900" />
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
            <div className="p-6 md:p-8 max-w-4xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-white mb-1">📝 RTI Application Draft</h2>
                <p className="text-slate-400 text-sm">Create and print Right to Information applications targeting PIO officers.</p>
              </div>

              <form onSubmit={handleRtiSubmit} className="bg-slate-800/40 border border-slate-700/60 p-6 rounded-2xl space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Applicant Name</label>
                    <input type="text" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={rtiData.applicant_name} onChange={(e) => setRtiData({ ...rtiData, applicant_name: e.target.value })} required />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Email</label>
                    <input type="email" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={rtiData.applicant_email} onChange={(e) => setRtiData({ ...rtiData, applicant_email: e.target.value })} required />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Phone</label>
                    <input type="text" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={rtiData.applicant_phone} onChange={(e) => setRtiData({ ...rtiData, applicant_phone: e.target.value })} required />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Address</label>
                    <textarea rows={2} className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition resize-none" value={rtiData.applicant_address} onChange={(e) => setRtiData({ ...rtiData, applicant_address: e.target.value })} required />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">PIO Office Name</label>
                    <input type="text" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" value={rtiData.pio_office} onChange={(e) => setRtiData({ ...rtiData, pio_office: e.target.value })} required />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">PIO Address</label>
                    <textarea rows={2} className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition resize-none" value={rtiData.pio_address} onChange={(e) => setRtiData({ ...rtiData, pio_address: e.target.value })} required />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Information Sought</label>
                  <textarea rows={3} className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500 transition" placeholder="State clearly what information you seek..." value={rtiData.information_sought} onChange={(e) => setRtiData({ ...rtiData, information_sought: e.target.value })} required />
                </div>

                <button type="submit" disabled={rtiLoading} className="w-full py-3 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-bold rounded-xl transition">
                  {rtiLoading ? 'Compiling Draft...' : 'Generate RTI draft'}
                </button>
              </form>

              {rtiError && <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-300 text-sm rounded-lg">{rtiError}</div>}
              {rtiResult && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-slate-350">RTI Draft Preview</h4>
                  <textarea readOnly value={rtiResult.application} className="w-full h-80 p-4 bg-slate-900 border border-slate-700 rounded-xl text-emerald-450 font-mono text-sm leading-relaxed outline-none" />
                  <p className="text-xs text-slate-400">💡 Send the draft copy with a court stamp/postal order of Rs. 10 to PIO.</p>
                </div>
              )}
            </div>
          )}

          {/* Deadline View */}
          {activeView === 'deadline' && (
            <div className="p-6 md:p-8 max-w-4xl mx-auto w-full space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-white mb-1">⏰ Deadline Limitation Calculator</h2>
                <p className="text-slate-400 text-sm">Calculate dates of expiration based on the Limitation Act 1963.</p>
              </div>

              <form onSubmit={handleDeadlineSubmit} className="bg-slate-800/40 border border-slate-700/60 p-6 rounded-2xl space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Case Category</label>
                    <select className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500" value={deadlineData.case_type} onChange={(e) => setDeadlineData({ ...deadlineData, case_type: e.target.value })}>
                      <option value="consumer">Consumer Court Complaints</option>
                      <option value="labour_salary">Labour Wage Dispute</option>
                      <option value="rti_appeal">RTI First Appeal</option>
                      <option value="rti_second_appeal">RTI Second Appeal</option>
                      <option value="rent_eviction">Tenant Eviction Order Appeal</option>
                      <option value="criminal_fir">Criminal FIR Limitation</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Occurrence Date</label>
                    <input type="date" className="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-white outline-none focus:border-amber-500" value={deadlineData.incident_date} onChange={(e) => setDeadlineData({ ...deadlineData, incident_date: e.target.value })} required />
                  </div>
                </div>

                <button type="submit" disabled={deadlineLoading} className="w-full py-3 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-bold rounded-xl transition">
                  {deadlineLoading ? 'Calculating...' : 'Run Calculator'}
                </button>
              </form>

              {deadlineError && <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-300 text-sm rounded-lg">{deadlineError}</div>}
              {deadlineResult && (
                <div className="bg-slate-800/50 border border-slate-700 p-6 rounded-2xl space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-700/60">
                      <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider font-semibold">Limitation Deadline</div>
                      <div className="text-xl font-bold text-white">{deadlineResult.deadline_date}</div>
                    </div>
                    <div className={`p-4 rounded-xl border ${deadlineResult.status === '✅ OK' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-450' : 'bg-red-500/10 border-red-500/30 text-red-400'
                      }`}>
                      <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider font-semibold">Limitation Status</div>
                      <div className="text-xl font-bold">{deadlineResult.status}</div>
                    </div>
                  </div>

                  {deadlineResult.days_remaining !== null && (
                    <div className="text-sm font-semibold text-slate-300">
                      Days Remaining to file: <span className={deadlineResult.days_remaining > 30 ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>{deadlineResult.days_remaining} Days</span>
                    </div>
                  )}

                  <div className="text-xs text-slate-400 leading-relaxed pt-2 border-t border-slate-700">
                    <strong>Rule description:</strong> <br />
                    {deadlineResult.description}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* PDF Modal */}
      {showPDFModal && (
        <div className="fixed inset-0 bg-black/55 z-50 flex items-center justify-center p-4">
          <div className="bg-slate-800 border border-slate-700 rounded-2xl max-w-2xl w-full max-h-[500px] overflow-y-auto p-6">
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-700">
              <h3 className="text-lg font-bold text-white">Generated Legal Notice PDFs</h3>
              <button
                onClick={() => setShowPDFModal(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition text-slate-300"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {userPDFs.length === 0 ? (
              <p className="text-slate-400 text-center py-6">No PDFs generated yet</p>
            ) : (
              <div className="space-y-3">
                {userPDFs.map((pdf, i) => (
                  <div key={i} className="p-4 rounded-xl bg-slate-900/40 border border-slate-700 flex items-center justify-between">
                    <div className="flex-1">
                      <p className="font-semibold text-white text-sm">{pdf.filename}</p>
                      <p className="text-xs text-slate-400 mt-1">🏷️ compensation claim: Rs. {pdf.demand_amount || 'N/A'}</p>
                    </div>
                    <button
                      onClick={() => downloadPDF(pdf._id, pdf.filename)}
                      className="px-3.5 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold flex items-center gap-2 transition text-sm shadow-md"
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
    </div>
  );
}

// ============ MAIN APP WITH PROVIDER ============

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
