/**
 * api.ts - FastAPI client
 * All API calls to your courtRoom.ai backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============ HELPER FUNCTIONS ============

function getAuthToken(): string | null {
  return localStorage.getItem('auth_token');
}

function setAuthToken(token: string): void {
  localStorage.setItem('auth_token', token);
}

async function fetchAPI(endpoint: string, options: RequestInit = {}): Promise<any> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || `API error: ${response.status}`);
  }

  return response.json();
}

// ============ AUTHENTICATION ============

export async function registerUser(
  email: string,
  password: string,
  name: string
): Promise<any> {
  return fetchAPI('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, name }),
  });
}

export async function loginUser(
  email: string,
  password: string
): Promise<any> {
  const result = await fetchAPI('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });

  if (result.access_token) {
    setAuthToken(result.access_token);
    localStorage.setItem('userId', result.user_id);
    localStorage.setItem('userName', result.name);
    localStorage.setItem('userEmail', result.email);
  }

  return result;
}

export function logoutUser(): void {
  localStorage.removeItem('auth_token');
  localStorage.removeItem('userId');
  localStorage.removeItem('userName');
  localStorage.removeItem('userEmail');
}

// ============ QUERIES ============

export async function submitQuery(
  query: string,
  userId: string = 'anonymous',
  language: string = 'en'
): Promise<any> {
  return fetchAPI(`/query?user_id=${userId}`, {
    method: 'POST',
    body: JSON.stringify({ query, language }),
  });
}

// ============ STREAMING QUERY (SSE) ============

export interface StreamCallbacks {
  onStatus?: (step: string, message: string) => void;
  onToken?: (text: string) => void;
  onFinal?: (data: any) => void;
  onError?: (message: string) => void;
}

export async function streamQuery(
  query: string,
  language: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, language }),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    callbacks.onError?.(text || `API error: ${response.status}`);
    return;
  }
  if (!response.body) {
    callbacks.onError?.('Streaming is not supported by this browser');
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const dispatch = () => {
    const frames = buffer.split('\n\n');
    buffer = frames.pop() || '';
    for (const frame of frames) {
      let event = 'message';
      const dataLines: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length === 0) continue;
      let payload: any;
      try {
        payload = JSON.parse(dataLines.join('\n'));
      } catch {
        continue;
      }
      switch (event) {
        case 'status':
          callbacks.onStatus?.(payload.step, payload.message);
          break;
        case 'token':
          callbacks.onToken?.(payload.text ?? '');
          break;
        case 'final':
          callbacks.onFinal?.(payload);
          break;
        case 'error':
          callbacks.onError?.(payload.detail ?? 'Unknown error');
          break;
        default:
          break;
      }
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    dispatch();
  }
  dispatch();
}

export async function getQueryHistory(userId: string): Promise<any[]> {
  return fetchAPI(`/user/${userId}/queries`);
}

// ============ PDF GENERATION ============

export async function generatePDFNotice(
  data: {
    sender_name: string;
    sender_address: string;
    recipient_name: string;
    recipient_address: string;
    issue_type: string;
    issue_description: string;
    applicable_section: string;
    demand_amount: string;
  },
  userId: string = 'anonymous'
): Promise<any> {
  return fetchAPI(`/generate-notice?user_id=${userId}`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function downloadPDF(
  pdfId: string,
  filename: string
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/pdf/${pdfId}/download`);
    if (!response.ok) throw new Error('Download failed');

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    console.error('Download error:', error);
    throw error;
  }
}

export async function getUserPDFs(userId: string): Promise<any[]> {
  return fetchAPI(`/user/${userId}/pdfs`);
}

// ============ EVIDENCE ============

export async function getEvidenceChecklist(domain: string): Promise<any> {
  return fetchAPI(`/evidence/${domain}`);
}

// ============ DEADLINE ============

export async function checkDeadline(
  caseType: string,
  incidentDate: string
): Promise<any> {
  return fetchAPI('/deadline', {
    method: 'POST',
    body: JSON.stringify({ case_type: caseType, incident_date: incidentDate }),
  });
}

// ============ RTI ============

export async function generateRTIApplication(data: Record<string, string>): Promise<any> {
  return fetchAPI('/rti-application', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ============ HEALTH CHECK ============

export async function healthCheck(): Promise<any> {
  return fetchAPI('/health');
}

// ============ TRANSLATION (Phase 10) ============

export async function translateText(
  text: string,
  sourceLang: string = 'en',
  targetLang: string = 'hi'
): Promise<string> {
  const result = await fetchAPI('/translate', {
    method: 'POST',
    body: JSON.stringify({ text, source_lang: sourceLang, target_lang: targetLang }),
  });
  return result.translated;
}
