/**
 * retrieval.ts - Updated to use FastAPI backend
 * Keeps your existing tokenization logic but connects to FastAPI RAG
 */

import type { LegalSection, RetrievalResult, LegalCategory } from './types';
import { detectCategory } from './language';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/** Stopwords excluded from tokenization. */
const STOPWORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'to', 'of',
  'in', 'on', 'at', 'for', 'and', 'or', 'but', 'not', 'no', 'my', 'me', 'i', 'we',
  'they', 'he', 'she', 'it', 'this', 'that', 'these', 'those', 'with', 'from', 'by',
  'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should',
  'may', 'might', 'must', 'shall', 'if', 'then', 'so', 'as', 'than', 'too', 'very',
  'just', 'about', 'into', 'out', 'up', 'down', 'over', 'under', 'again', 'more',
  'most', 'some', 'any', 'all', 'each', 'every', 'other', 'such', 'only', 'own',
  'same', 'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
  'nathi', 'nathi', 'che', 'mali', 'aapyo', 'aape', 'madyo', 'kartu', 'gayi', 'thai',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));
}

/**
 * Retrieve legal sections from FastAPI backend
 * FastAPI handles the RAG pipeline and returns rich metadata
 */
export async function retrieveSections(
  query: string,
  topK = 4
): Promise<RetrievalResult[]> {
  try {
    const userId = localStorage.getItem('userId') || 'anonymous';
    
    // Call FastAPI backend
    const response = await fetch(`${API_BASE_URL}/query?user_id=${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        language: 'en'
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();

    // Store query ID for PDF generation
    if (result.query_id) {
      localStorage.setItem('lastQueryId', result.query_id);
    }

    // Transform FastAPI response to your RetrievalResult format
    const results: RetrievalResult[] = (result.sources || []).map((source: any, index: number) => {
      const section: LegalSection = {
        id: `${index}`,
        document_id: source.source_act || 'Unknown',
        section_number: source.section || 'General',
        title: source.topic || 'Legal Information',
        content: source.content_preview || '',
        plain_language: source.content_preview || '',
        keywords: source.keywords || [],
        category: result.domain as LegalCategory || 'general',
        next_steps: '',
        jurisdiction: source.courts?.[0] || 'District Court',
        document_name: source.source_act || 'Indian Law',
        document_short_name: source.source_act?.split(' ')[0] || 'LAW',
      };

      return {
        section,
        score: 100 - (index * 10),
        matchedTerms: source.keywords || []
      };
    });

    return results.slice(0, topK);
  } catch (error) {
    console.error('Retrieval error:', error);
    // Fallback: return empty array
    return [];
  }
}

/**
 * Get query history for user
 */
export async function getUserQueries(): Promise<any[]> {
  try {
    const userId = localStorage.getItem('userId');
    if (!userId) return [];

    const response = await fetch(`${API_BASE_URL}/user/${userId}/queries`);
    if (!response.ok) throw new Error('Failed to fetch queries');
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching queries:', error);
    return [];
  }
}
