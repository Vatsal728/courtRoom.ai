export type LegalCategory = 'consumer' | 'labour' | 'rent' | 'rti' | 'criminal' | 'cyber' | 'civil' | 'general';

export type Language =
  | 'en'
  | 'hi'
  | 'gu'
  | 'mr'
  | 'ta'
  | 'te'
  | 'kn'
  | 'bn'
  | 'ml'
  | 'pa'
  | 'ur';

export interface LanguageInfo {
  code: Language;
  name: string;
  native: string;
}

export interface LegalSection {
  id: string;
  document_id: string;
  section_number: string;
  title: string;
  content: string;
  plain_language: string;
  keywords: string[];
  category: LegalCategory;
  next_steps?: string;
  jurisdiction?: string;
  document_name?: string;
  document_short_name?: string;
}

export interface RetrievalResult {
  section: LegalSection;
  score: number;
  matchedTerms: string[];
}

// ── Agent tool artifacts (Phase 3/4) ────────────────────────────────────

export type ToolName = 'legal_notice' | 'rti_application' | 'case_strategy' | 'document_audit';
export type ToolStatus = 'ready' | 'needs_input' | 'error';

export interface ToolCard {
  type: ToolName;
  status: ToolStatus;
  title: string;
  message: string;
  pdf_id?: string;
  filename?: string;
  download_url?: string;
  application?: string;
  strategy?: CaseStrategyResult;
  audit?: DocumentAuditResult;
  view?: string;
  disclaimer?: string;
  missing_fields?: string[];
  params?: Record<string, any>;
}

export interface CompensationEstimate {
  domain: string;
  min_amount: number;
  max_amount: number;
  currency: string;
  basis: string;
  notes: string[];
  disclaimer: string;
}

export interface CaseStrategyResult {
  domain: string;
  summary: string;
  assessment: { strengths: string[]; weaknesses: string[] };
  legal_route: { criminal: boolean; civil: boolean; forums: string[] };
  compensation_estimate: CompensationEstimate;
  evidence_checklist: string[];
  deadline: null | {
    case_type: string;
    incident_date: string;
    deadline_date: string;
    days_remaining: number | null;
    status: string;
    description: string;
  };
  action_plan: string[];
  disclaimer: string;
}

export interface AuditCheck {
  id: string;
  label: string;
  severity: string;
  hint?: string;
  issue?: string;
}

export interface DocumentAuditResult {
  domain: string;
  document_type: string;
  audit_intro: string;
  score: number;
  risk: 'LOW' | 'MEDIUM' | 'HIGH';
  present_count: number;
  missing_count: number;
  total_checks: number;
  present: AuditCheck[];
  issues: AuditCheck[];
  missing_fields: string[];
  suggestions: string[];
  disclaimer: string;
}
